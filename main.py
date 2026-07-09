import torch
import argparse
import copy
import time
import pickle
import random
import os 
import numpy as np 

import sys
sys.path.append('./models')

# 수정된 import
from utils import init_logger, init_seed, shuffle_clients, compute_accuracy, avg_last_n, init_net
from data_utils import get_global_dataset, partition_data, get_client_datasets, get_global_dataloader, get_client_dataloaders
import importlib
import torch.optim as optim
import torch.nn.functional as F
from algs.sam import SAM
from utils import update_ema_model

from models.resnet_byot import multi_resnet18_kd
from models.resnet_cifar import ResNet18_cifar10
from models.mobilenet_v2 import MobileNetV2, MobileNetV2BYOT

import fl_utils 

# python main.py --seed 0 --model mobilenet --last_fc --alg fedavg

def compute_client_update_drift(old_w, nets_this_round, fed_avg_freqs, eps=1e-12):
    """
    Measure how far local client updates deviate from the weighted mean update.
    This is an update-space proxy for client drift, computed after local training
    and before aggregation.
    """
    client_ids = list(nets_this_round.keys())
    if not client_ids:
        return {}

    float_keys = [
        key for key, value in old_w.items()
        if torch.is_floating_point(value)
    ]
    if not float_keys:
        return {}

    with torch.no_grad():
        mean_update = {}
        for key in float_keys:
            mean = torch.zeros_like(old_w[key], device='cpu')
            old_value = old_w[key].detach().cpu()
            for idx, client_id in enumerate(client_ids):
                local_value = nets_this_round[client_id].state_dict()[key].detach().cpu()
                mean += float(fed_avg_freqs[idx]) * (local_value - old_value)
            mean_update[key] = mean

        update_norm = 0.0
        update_norm_sq = 0.0
        divergence = 0.0
        cosine_sum = 0.0
        mean_norm_sq = 0.0

        for key in float_keys:
            mean_norm_sq += float(torch.sum(mean_update[key] * mean_update[key]).item())
        mean_norm = mean_norm_sq ** 0.5

        for idx, client_id in enumerate(client_ids):
            weight = float(fed_avg_freqs[idx])
            client_norm_sq = 0.0
            client_dot_mean = 0.0
            client_divergence = 0.0
            state = nets_this_round[client_id].state_dict()

            for key in float_keys:
                update = state[key].detach().cpu() - old_w[key].detach().cpu()
                centered = update - mean_update[key]
                client_norm_sq += float(torch.sum(update * update).item())
                client_dot_mean += float(torch.sum(update * mean_update[key]).item())
                client_divergence += float(torch.sum(centered * centered).item())

            client_norm = client_norm_sq ** 0.5
            update_norm += weight * client_norm
            update_norm_sq += weight * client_norm_sq
            divergence += weight * client_divergence
            if client_norm > 0.0 and mean_norm > 0.0:
                cosine_sum += weight * (client_dot_mean / (client_norm * mean_norm + eps))

        return {
            "client_update_norm": update_norm,
            "client_update_norm_sq": update_norm_sq,
            "client_mean_update_norm": mean_norm,
            "client_update_divergence": divergence,
            "client_relative_drift": divergence / (mean_norm_sq + eps),
            "client_update_cosine": cosine_sum,
        }

def _flatten_current_grads(model):
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param, device='cpu').flatten())
        else:
            grads.append(param.grad.detach().cpu().flatten())
    if not grads:
        return torch.empty(0)
    return torch.cat(grads)

def _gradient_probe_losses(model, x, target, temperature):
    out = model(x)
    if isinstance(out, tuple) and len(out) == 8:
        output, m1, m2, m3, _, _, _, _ = out
        ce_loss = (
            F.cross_entropy(output, target)
            + F.cross_entropy(m1, target)
            + F.cross_entropy(m2, target)
            + F.cross_entropy(m3, target)
        )
        with torch.no_grad():
            teacher_prob = F.softmax(output / temperature, dim=1)
        kd_loss = (
            F.kl_div(F.log_softmax(m1 / temperature, dim=1), teacher_prob, reduction='batchmean')
            + F.kl_div(F.log_softmax(m2 / temperature, dim=1), teacher_prob, reduction='batchmean')
            + F.kl_div(F.log_softmax(m3 / temperature, dim=1), teacher_prob, reduction='batchmean')
        ) * (temperature ** 2)
        return ce_loss, kd_loss

    if isinstance(out, tuple):
        output = out[-1]
    else:
        output = out
    return F.cross_entropy(output, target), None

def _average_kd_info_probe(model, dataloader, device, args, max_batches):
    metrics = {
        "teacher_entropy": 0.0,
        "teacher_entropy_norm": 0.0,
        "teacher_true_label_prob": 0.0,
        "teacher_non_target_mass": 0.0,
        "teacher_top2_margin": 0.0,
        "teacher_confidence": 0.0,
    }
    total_count = 0
    temperature = float(getattr(args, 'temperature', 0.5))

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            x, target = x.to(device), target.to(device).long()
            out = model(x)
            if isinstance(out, tuple) and len(out) == 8:
                output = out[0]
            elif isinstance(out, tuple):
                output = out[-1]
            else:
                output = out

            prob = F.softmax(output / temperature, dim=1)
            batch_size = int(target.numel())
            num_classes = max(int(prob.size(1)), 2)
            entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)
            true_label_prob = prob.gather(1, target.view(-1, 1)).squeeze(1).clamp(0.0, 1.0)
            top2 = prob.topk(k=min(2, num_classes), dim=1).values
            if top2.size(1) == 1:
                margin = torch.ones_like(top2[:, 0])
            else:
                margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
            confidence = top2[:, 0]

            metrics["teacher_entropy"] += float(entropy.sum().item())
            metrics["teacher_entropy_norm"] += float((entropy / math.log(num_classes)).sum().item())
            metrics["teacher_true_label_prob"] += float(true_label_prob.sum().item())
            metrics["teacher_non_target_mass"] += float((1.0 - true_label_prob).sum().item())
            metrics["teacher_top2_margin"] += float(margin.sum().item())
            metrics["teacher_confidence"] += float(confidence.sum().item())
            total_count += batch_size

    if total_count == 0:
        return None
    return {key: value / total_count for key, value in metrics.items()}

def _average_probe_gradient(model, dataloader, device, args, loss_kind, max_batches):
    grads = []
    temperature = float(getattr(args, 'temperature', 0.5))
    for batch_idx, (x, target) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x, target = x.to(device), target.to(device).long()
        model.zero_grad(set_to_none=True)
        ce_loss, kd_loss = _gradient_probe_losses(model, x, target, temperature)
        if loss_kind == 'ce':
            loss = ce_loss
        elif loss_kind == 'kd':
            if kd_loss is None:
                continue
            loss = kd_loss
        else:
            raise ValueError(f"Unknown probe loss kind: {loss_kind}")
        loss.backward()
        grads.append(_flatten_current_grads(model))
    model.zero_grad(set_to_none=True)
    if not grads:
        return None
    return torch.stack(grads, dim=0).mean(dim=0)

def _weighted_gradient_stats(gradients, weights, eps=1e-12):
    valid = [(grad, float(weight)) for grad, weight in zip(gradients, weights) if grad is not None]
    if not valid:
        return None

    total_weight = sum(weight for _, weight in valid)
    if total_weight <= 0:
        return None
    normalized = [(grad, weight / total_weight) for grad, weight in valid]

    mean_grad = torch.zeros_like(normalized[0][0])
    for grad, weight in normalized:
        mean_grad += weight * grad

    divergence = 0.0
    norm = 0.0
    norm_sq = 0.0
    cosine = 0.0
    mean_norm_sq = float(torch.sum(mean_grad * mean_grad).item())
    mean_norm = mean_norm_sq ** 0.5

    for grad, weight in normalized:
        centered = grad - mean_grad
        grad_norm_sq = float(torch.sum(grad * grad).item())
        grad_norm = grad_norm_sq ** 0.5
        divergence += weight * float(torch.sum(centered * centered).item())
        norm += weight * grad_norm
        norm_sq += weight * grad_norm_sq
        if grad_norm > 0.0 and mean_norm > 0.0:
            dot = float(torch.sum(grad * mean_grad).item())
            cosine += weight * (dot / (grad_norm * mean_norm + eps))

    return {
        "divergence": divergence,
        "relative": divergence / (mean_norm_sq + eps),
        "norm": norm,
        "norm_sq": norm_sq,
        "mean_norm": mean_norm,
        "cosine": cosine,
    }

def _weighted_centered_cross_stats(left_gradients, right_gradients, weights, eps=1e-12):
    valid = [
        (left, right, float(weight))
        for left, right, weight in zip(left_gradients, right_gradients, weights)
        if left is not None and right is not None
    ]
    if not valid:
        return None

    total_weight = sum(weight for _, _, weight in valid)
    if total_weight <= 0:
        return None
    normalized = [(left, right, weight / total_weight) for left, right, weight in valid]

    mean_left = torch.zeros_like(normalized[0][0])
    mean_right = torch.zeros_like(normalized[0][1])
    for left, right, weight in normalized:
        mean_left += weight * left
        mean_right += weight * right

    cross = 0.0
    left_divergence = 0.0
    right_divergence = 0.0
    for left, right, weight in normalized:
        centered_left = left - mean_left
        centered_right = right - mean_right
        cross += weight * float(torch.sum(centered_left * centered_right).item())
        left_divergence += weight * float(torch.sum(centered_left * centered_left).item())
        right_divergence += weight * float(torch.sum(centered_right * centered_right).item())

    corr = cross / ((left_divergence * right_divergence) ** 0.5 + eps)
    return {
        "cross": cross,
        "corr": corr,
    }

def compute_gradient_drift_probe(nets_this_round, dataloaders_this_round, fed_avg_freqs, device, args):
    """
    Probe gradient dissimilarity at the round-start global model.
    CE and KD gradients are measured separately, then combined as CE + alpha * KD
    to match the theorem-level FedSD objective decomposition.
    """
    max_batches = max(1, int(getattr(args, 'gradient_probe_batches', 1)))
    alpha = float(getattr(args, 'byot_alpha', 0.0))
    ce_gradients = []
    kd_gradients = []
    kd_info_sums = {}
    used_weights = []

    for idx, client_id in enumerate(nets_this_round.keys()):
        dataloader = dataloaders_this_round.get(client_id)
        if dataloader is None:
            continue
        model = nets_this_round[client_id]
        was_training = model.training
        model.eval()
        kd_info = _average_kd_info_probe(model, dataloader, device, args, max_batches)
        ce_grad = _average_probe_gradient(model, dataloader, device, args, 'ce', max_batches)
        kd_grad = _average_probe_gradient(model, dataloader, device, args, 'kd', max_batches)
        if was_training:
            model.train()
        if ce_grad is None:
            continue
        if kd_grad is None:
            kd_grad = torch.zeros_like(ce_grad)
        ce_gradients.append(ce_grad)
        kd_gradients.append(kd_grad)
        weight = float(fed_avg_freqs[idx])
        used_weights.append(weight)
        if kd_info is not None:
            for key, value in kd_info.items():
                kd_info_sums[key] = kd_info_sums.get(key, 0.0) + weight * value

    ce_stats = _weighted_gradient_stats(ce_gradients, used_weights)
    kd_stats = _weighted_gradient_stats(kd_gradients, used_weights)
    combined = [ce_grad + alpha * kd_grad for ce_grad, kd_grad in zip(ce_gradients, kd_gradients)]
    combined_stats = _weighted_gradient_stats(combined, used_weights)
    cross_stats = _weighted_centered_cross_stats(ce_gradients, kd_gradients, used_weights)

    if ce_stats is None or kd_stats is None or combined_stats is None:
        return {}

    metrics = {"gradient_probe_clients": len(ce_gradients)}
    for prefix, stats in [
        ("gradient_ce", ce_stats),
        ("gradient_kd", kd_stats),
        ("gradient_combined", combined_stats),
    ]:
        for key, value in stats.items():
            metrics[f"{prefix}_{key}"] = value
    if cross_stats is not None:
        metrics["gradient_ce_kd_cross"] = cross_stats["cross"]
        metrics["gradient_ce_kd_corr"] = cross_stats["corr"]
    for key, value in kd_info_sums.items():
        metrics[f"kd_info_{key}"] = value

    ce_kd_cosine = 0.0
    ce_kd_distance = 0.0
    kd_ce_norm_ratio = 0.0
    weight_total = sum(used_weights)
    if weight_total > 0:
        for ce_grad, kd_grad, weight in zip(ce_gradients, kd_gradients, used_weights):
            weight = weight / weight_total
            ce_norm = torch.norm(ce_grad)
            kd_norm = torch.norm(kd_grad)
            denom = ce_norm * kd_norm + 1e-12
            ce_kd_cosine += weight * float(torch.sum(ce_grad * kd_grad).item() / denom.item())
            ce_kd_distance += weight * float(torch.norm(ce_grad - kd_grad).item() / (ce_norm.item() + kd_norm.item() + 1e-12))
            kd_ce_norm_ratio += weight * float(kd_norm.item() / (ce_norm.item() + 1e-12))
        metrics["gradient_ce_kd_cosine"] = ce_kd_cosine
        metrics["gradient_ce_kd_distance"] = ce_kd_distance
        metrics["gradient_kd_ce_norm_ratio"] = kd_ce_norm_ratio

    return metrics

def norm_based_classwise_aggregation(global_model, nets_this_round, clients_this_round, client_data_sizes=None):
    """
    특징 추출기(Body)는 데이터 장수 비례 FedAvg로 합치고,
    분류기(Classifier)는 각 클래스별 가중치 변화량(Norm) 비례로 합치는 함수
    """
    global_w = global_model.state_dict()
    new_global_w = copy.deepcopy(global_w)
    
    # 클라이언트별 데이터 수 비율 계산 (기본 FedAvg용)
    # 데이터 수를 모른다면 동일한 비율(1/N)로 설정
    num_clients = len(nets_this_round)
    if client_data_sizes is None:
        weights = {k: 1.0 / num_clients for k in nets_this_round.keys()}
    else:
        total_data = sum([client_data_sizes[k] for k in nets_this_round.keys()])
        weights = {k: client_data_sizes[k] / total_data for k in nets_this_round.keys()}

    # 1. Body (특징 추출기) 집계: 기존 FedAvg 방식 적용
    # 'fc'나 'classifier'가 이름에 없는 레이어들만 단순 가중 평균
    for key in global_w.keys():
        if 'fc' not in key and 'classifier' not in key:
            new_global_w[key] = sum(weights[k] * nets_this_round[k].state_dict()[key] for k in nets_this_round.keys())

    # 2. Classifier (분류기) 집계: 가중치 변화량(Norm) 기반 Class-wise 가중 평균
    # ResNet18 기반이므로 마지막 분류기 이름은 보통 'fc.weight'와 'fc.bias'임
    fc_weight_key = 'fc.weight'
    fc_bias_key = 'fc.bias'
    
    if fc_weight_key in global_w:
        num_classes = global_w[fc_weight_key].shape[0]
        
        # 클라이언트별, 클래스별 Norm을 저장할 딕셔너리
        class_norms = {k: torch.zeros(num_classes, device=global_w[fc_weight_key].device) for k in nets_this_round.keys()}
        
        # [Norm 계산] 각 클라이언트가 특정 클래스를 얼마나 학습했는지 추정
        for k, net in nets_this_round.items():
            local_w = net.state_dict()
            # (local - global) 변화량 계산
            delta_w = local_w[fc_weight_key] - global_w[fc_weight_key] 
            # row(클래스) 단위로 L2 Norm 크기 계산
            norms = torch.norm(delta_w, p=2, dim=1) 
            class_norms[k] = norms
            
        # 클래스별 총 Norm 합 (나중에 비율을 구하기 위해)
        sum_norms = torch.zeros(num_classes, device=global_w[fc_weight_key].device)
        for k in nets_this_round.keys():
            sum_norms += class_norms[k]
            
        # 0으로 나누는 것 방지 (업데이트가 아예 없는 클래스의 경우)
        sum_norms[sum_norms == 0] = 1e-8
        
        # 새로운 분류기 가중치 초기화
        new_fc_weight = torch.zeros_like(global_w[fc_weight_key])
        has_bias = fc_bias_key in global_w
        if has_bias:
            new_fc_bias = torch.zeros_like(global_w[fc_bias_key])
            
        # [클래스 단위 집계] Norm이 큰 클라이언트의 가중치를 더 많이 반영
        for c in range(num_classes):
            for k, net in nets_this_round.items():
                local_w = net.state_dict()
                # 현재 클래스 c에 대한 클라이언트 k의 기여도 비율
                weight_c = class_norms[k][c] / sum_norms[c]
                
                new_fc_weight[c] += weight_c * local_w[fc_weight_key][c]
                if has_bias:
                    new_fc_bias[c] += weight_c * local_w[fc_bias_key][c]
                    
        # 업데이트된 분류기 덮어쓰기
        new_global_w[fc_weight_key] = new_fc_weight
        if has_bias:
            new_global_w[fc_bias_key] = new_fc_bias

    return new_global_w

# ==============================================================================
# [새로 추가] ECE (Expected Calibration Error) 계산 함수
# ==============================================================================
def compute_ece(preds, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences, predictions = torch.max(preds, 1)
    accuracies = predictions.eq(labels)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

# ==============================================================================
# [최종 수정] 튜플 처리 + 차원 문제 해결된 test 함수
# ==============================================================================
def test(model, test_loader, device, args): # args 파라미터 추가!
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    total_loss = 0.0
    total = 0
    correct = 0
    
    # ECE 계산용
    all_preds = []
    all_targets = []
    
    # Intermediate Accuracy (Branch별) 계산용
    correct_branches = None
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            final_output = None
            branch_outputs = []

            if isinstance(outputs, tuple):
                if len(outputs) == 8:
                    branch_outputs = [outputs[1], outputs[2], outputs[3], outputs[0]]
                    
                    # [전략 A] 앙상블 적용 여부
                    if getattr(args, 'use_ensemble', False):
                        final_output = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
                    else:
                        final_output = outputs[0]
                        
                elif isinstance(outputs[0], list):
                    branch_outputs = outputs[0]
                    if getattr(args, 'use_ensemble', False):
                        final_output = sum(branch_outputs) / len(branch_outputs)
                    else:
                        final_output = branch_outputs[-1]
                else:
                    final_output = outputs[1]
                    branch_outputs = [final_output]
            elif isinstance(outputs, list):
                branch_outputs = outputs
                if getattr(args, 'use_ensemble', False):
                    final_output = sum(branch_outputs) / len(branch_outputs)
                else:
                    final_output = branch_outputs[-1]
            else:
                final_output = outputs
                branch_outputs = [final_output]

            # 차원 강제 변환 (4D -> 2D)
            if final_output.dim() > 2:
                final_output = final_output.view(final_output.size(0), -1)
            for i in range(len(branch_outputs)):
                if branch_outputs[i].dim() > 2:
                    branch_outputs[i] = branch_outputs[i].view(branch_outputs[i].size(0), -1)

            # 첫 배치에서 Branch 개수 확인 및 초기화
            if correct_branches is None:
                correct_branches = [0] * len(branch_outputs)

            # 1. Test Loss 계산 (Final Output 기준)
            loss = criterion(final_output, target)
            total_loss += loss.item() * data.size(0)
            
            # 2. Final Accuracy (또는 앙상블) 계산
            _, pred = final_output.max(1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

            # 3. Intermediate Accuracy (Branch별 개별 성적) 집계
            for i, out in enumerate(branch_outputs):
                _, b_pred = out.max(1)
                correct_branches[i] += b_pred.eq(target).sum().item()

            # 4. ECE용 데이터 수집
            prob = torch.nn.functional.softmax(final_output, dim=1)
            all_preds.append(prob)
            all_targets.append(target)

    # --- 결과 집계 ---
    avg_loss = total_loss / total
    final_acc = 100. * correct / total
    
    branch_accs = [100. * c / total for c in correct_branches]
    
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        ece_score = compute_ece(all_preds, all_targets)
    else:
        ece_score = 0.0

    return avg_loss, final_acc, ece_score, branch_accs

def get_args():
    parser = argparse.ArgumentParser()
    ####
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    # 시드 설정. 여러번 실험해서 평균넬때 사용하시면 됩니다
    parser.add_argument('--datadir', default="./data/", help="Data directory")
    # 데이터를 다운로드하거나, 이미 있는 경로 (서버마다 공유 데이터 경로가 있음)  v100: /mnt/data3/ 8000: /mnt/data1/ or /mnt/data3/ , 4000: /data/  ,  2080: /mnt/data3/
    parser.add_argument('--logdir', default="./logs/", help='Log directory path')
    parser.add_argument('--log_file_name', default=None, help='log file name')
    parser.add_argument('--device', default='cuda:0', help='The device to run the program')
    # gpu 번호 설정
    parser.add_argument('--num_workers', default=0, type=int, help='the number of workers for each dataloader')
    # 0 추천

    #### model
    parser.add_argument('--dataset', default='cifar10', help='dataset used for training')
    parser.add_argument('--model', default='resnet50', help='neural network used in training')
    parser.add_argument('--group_norm', action='store_true', help='replace batch_norm with group_norm')
    # batchnorm -> groupnorm. 켜도되고 안켜도 되는데, groupnorm이 더 좋다는 연구결과가 있는데 꼭 그렇지는 않더라구요
    parser.add_argument('--num_groups', type=int, default=8, help='num of groups in group_norm')
    # group_norm의 groups
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes; inferred from dataset when omitted')
    parser.add_argument('--in_channels', type=int, default=3)
    # image channels. 1 channel 데이터셋이도 3 channel로 하면 gray scale -> RGB scale 채널이 복사되서 작동합니다
    parser.add_argument('--last_fc', action='store_true', help='For mobilenet, classifier is fc_layer if True, otherwise, 1x1 conv')
    # mobilenet에서만 작동. 기본값은 마지막 classifier가 1x1 conv
    

    #### Hyperparameters for training
    parser.add_argument('--optimizer', default='sgd', help='the optimizer')
    # adam, amsgrad 도 가능하고, 필요하다면 train.py에서 수정 가능합니다
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch size for validation or test')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--scheduler', default='round', help='scheduler for rounds [round, linear, cosine]')
    # round: 일정 round마다 lr 감소, cosine: cosine annealing lr 참조, 안쓰고 싶은떈 아무거나 막입력하면 됩니다 train.py의 adjust_lr 참조
    parser.add_argument('--schedule_round', type=int, default=1)
    # 몇 라운드마다 lr 감소할지
    parser.add_argument('--lr_gamma', type=float, default=0.998)
    # 얼마나 감소 new_lr = cur_lr * lr_gamma
    parser.add_argument('--eta_min', type=float, default=0.0, help='minimum learning rate for cosine')
    # cosine lr 에서만 동작합니다.
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--reg', type=float, default=1e-3, help="L2 regularization strength")

    #### Federated learning settings
    parser.add_argument('--partition', default='noniid', help='iid, noniid, noniid-balanced, noniid-grouping')
    # noniid는 클래스 뿐만 아니라 데이터 수도 불균일한 설정입니다. 논문마다 달라서 noniid-balanced써도 됩니다. data_utils의 partition_data 참조
    parser.add_argument('--group_ratios', nargs='+', type=float, default=[0.2, 0.5, 0.3], help='Ratios for Group A, B, C')
    parser.add_argument('--group_betas', nargs='+', type=float, default=[5.0, 0.5, 0.1], help='Betas for Group A, B, C')
    
    parser.add_argument('--alg', default='fedavg', help='communication strategy: fedavg/fedprox/...')
    parser.add_argument('--round', type=int, default=500, help='number of maximum communication round')
    parser.add_argument('--n_clients', type=int, default=100, help='number of workers in a distributed cluster')
    # 데이터는 클라이언트들에게 겹치지않게 분배됩니다
    parser.add_argument('--beta', type=float, default=0.3, help='The parameter for the dirichlet distribution for data partitioning')
    # noniid 조절, 작을수록 noniid
    parser.add_argument('--min_require_size', type=int, default=64, help='the minimum number of data for each client')
    # 클라이언트마다 최소 데이터 요구량. 데이터를 분배할때 noniid 세팅에선 클라이언트마다 데이터수가 다르기 때문에, 모든 클라이언트가 최소 사이즈를 가지도록 partition을 반복. 2000번 반복해보고 안되면 임의로 분배. partition_data 참조
    parser.add_argument('--unavailability', default='stationary', help='stationary, non-stationary, non-stationary-failure')
    # 클라이언트의 FL참여를 조절. stationary가 일반적, non-stationary는 클라이언트들이 라운드마다 불규칙하게 참여
    parser.add_argument('--sample_fraction', type=float, default=0.1, help='how many clients are sampled in each round')
    # round 당 클라이언트 참여비율
    parser.add_argument('--time', type=int, default=0)
    parser.add_argument('--no_init', action='store_true', help='Kaiming init is not performed')
    # 기본은 kaiming init. no_init은 별도의 initialization 없이 torch 기본 설정으로 초기화가 이루어집니다. 써도되고 안써도되요
    parser.add_argument('--init', default='normal', help='normal, kaiming_uniform, kaiming_orthogonal, orthogonal')
    # initialization 방법입니다 normal 추천
    parser.add_argument('--fan', default='fan_out')
    # weight init을 위한 fan은 fan_in, fan_out 이 있습니다. kaiming init 참조
    parser.add_argument('--linit', action='store_true')
    # last layer 의 init을 어떻게 할지 정합니다. init할때 gain 이라는 것을 정해야하는데, 보통 relu의 gain (아마도 sqrt(2))로 씁니다.
    # 그런데 마지막 layer는 relu가 없기 때문에 gain=1 을 쓰는게 맞는데, linit을 키면 마지막 layer의 gain을 1로 사용합니다. (안키면 relu의 gain을 사용)
    # 정답은 없어서 하고싶은대로 하시면 됩니다
    
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='dxfl')
    parser.add_argument('--wandb_entity', default=None)
    # wandb
    

    # 설정한 alg에 따라서 아래 하이퍼파라미터를 조절하시면 됩니다 👍
    ##################################################### Hypereparameters for other algs
    ## Fedrs
    parser.add_argument('--fedrs_alpha', type=float, default=0.5, help='FedRS scaling alpha for missing classes')
    ## Fedlogtical
    parser.add_argument('--calibration_temp', type=float, default=1.0, help='FedLC margin calibration temperature')
    ## Fedprox or MOON
    parser.add_argument('--mu', type=float, default=1.0, help='the mu parameter for fedprox, moon, or fedrcl')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    ## FedavgM
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    ## FedAG, FedAdam
    parser.add_argument('--server_momentum_second', type=float, default=0, help='the server momentum (FedAG, FedAdam)')
    parser.add_argument('--tau_optim', type=float, default=0.001, help='the server momentum (FedAG, FedAdam)')
    parser.add_argument('--server_lr', type=float, default=1)
    ## FedDecorr
    parser.add_argument('--feddecorr', action='store_true')
    parser.add_argument('--feddecorr_coef', type=float, default=0.1)
    ## FedRCL
    parser.add_argument('--tau_rcl', default=0.05, type=float)
    parser.add_argument('--threshold_rcl', default=0.7, type=float)
    parser.add_argument('--beta_rcl', default=1, type=float)
    ## FedSOL
    parser.add_argument('--rho', default=0.5, type=float)
    parser.add_argument('--adaptive', help='adaptive rho', action='store_true')
    ## FedACG
    parser.add_argument('--lambda_acg', type=float, default=0.85, help='lookahead gradient coefficient')
    parser.add_argument('--beta_acg', type=float, default=0.01)
    ## FedEXP
    parser.add_argument('--eps_exp', type=float, default=0.1)
    ## FLoCoRA
    parser.add_argument('--lora_r', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA')
    ## BYOT
    parser.add_argument('--byot_alpha', type=float, default=0.15, help='Weight for KD Loss in BYOT')
    parser.add_argument('--byot_branch_alphas', type=str, default='',
                        help='Optional comma-separated KD weights for BYOT branches B1,B2,B3. '
                             'Example: "1.0,0.7,0.5". Overrides --byot_alpha inside fedbyot.')
    parser.add_argument('--byot_active_branches', type=str, default='1,2,3',
                        help='Comma-separated BYOT branches whose CE/KD/feature losses are used. '
                             'Use 1-based branch ids, e.g. "3", "2,3", or "1,2,3".')
    parser.add_argument('--byot_branch_loss_reduction', default='sum',
                        choices=['sum', 'mean'],
                        help='How active BYOT branch losses are reduced. '
                             'sum preserves the original behavior; mean divides by the number of active branches.')
    parser.add_argument('--byot_branch_objective', default='blend',
                        choices=['blend', 'kd_only'],
                        help='BYOT branch objective. blend uses (1-alpha)*CE + alpha*KD; '
                             'kd_only removes branch CE and uses alpha as an unrestricted KD coefficient.')
    parser.add_argument('--byot_branch_gate', default='none',
                        choices=['none', 'entropy_3stage', 'entropy_no_off'],
                        help='Client-label-entropy based BYOT branch gate. '
                             'entropy_3stage: off/B1/all, entropy_no_off: B1/all.')
    parser.add_argument('--branch_entropy_off_threshold', type=float, default=0.2,
                        help='Normalized label entropy threshold below which entropy_3stage disables branch losses.')
    parser.add_argument('--branch_entropy_b1_threshold', type=float, default=0.5,
                        help='Normalized label entropy threshold below which branch gate uses B1 only.')
    parser.add_argument('--byot_beta', type=float, default=0.05, help='Weight for Feature Loss in BYOT')
    parser.add_argument('--beta_aware_byot_alpha', action='store_true',
                        help='Scale BYOT KD alpha by global Dirichlet beta for non-IID partitions.')
    parser.add_argument('--alpha_beta_ref', type=float, default=0.5,
                        help='Dirichlet beta where beta-aware alpha reaches the base byot_alpha.')
    parser.add_argument('--adaptive_byot_alpha', action='store_true',
                        help='Scale BYOT KD alpha by each client label entropy.')
    parser.add_argument('--alpha_min_scale', type=float, default=0.2,
                        help='Minimum alpha scale when client label entropy is near zero.')
    parser.add_argument('--alpha_entropy_power', type=float, default=1.0,
                        help='Power applied to normalized label entropy for adaptive alpha.')
    parser.add_argument('--byot_alpha_proxy', default='none',
                        choices=['none', 'teacher_conf', 'teacher_entropy', 'branch_agreement',
                                 'branch_soft_kl', 'branch_js', 'teacher_correctness'],
                        help='Batch-level proxy used to scale BYOT KD alpha.')
    parser.add_argument('--byot_sample_proxy', default='none',
                        choices=['none', 'teacher_conf', 'teacher_entropy', 'teacher_margin',
                                 'teacher_label_prob', 'teacher_correctness', 'branch_agreement',
                                 'branch_soft_kl', 'branch_js', 'teacher_label_prob_entropy',
                                 'teacher_label_prob_branch_js',
                                 'teacher_label_prob_entropy_branch_js'],
                        help='Sample-level reliability proxy used to weight BYOT KD loss.')
    parser.add_argument('--byot_client_proxy', default='none',
                        choices=['none', 'teacher_conf', 'teacher_entropy', 'teacher_margin',
                                 'teacher_label_prob', 'teacher_correctness', 'branch_agreement',
                                 'branch_soft_kl', 'branch_js', 'teacher_label_prob_entropy',
                                 'teacher_label_prob_branch_js',
                                 'teacher_label_prob_entropy_branch_js'],
                        help='Client-level reliability proxy used to choose one BYOT KD alpha per client.')
    parser.add_argument('--byot_client_alpha_min', type=float, default=0.01,
                        help='Minimum client-wise BYOT alpha when --byot_client_proxy is enabled.')
    parser.add_argument('--byot_client_alpha_max', type=float, default=0.30,
                        help='Maximum client-wise BYOT alpha when --byot_client_proxy is enabled.')
    parser.add_argument('--byot_client_alpha_mode', default='map',
                        choices=['map', 'multiply'],
                        help='How client reliability is applied. map: alpha_min+(alpha_max-alpha_min)*r. '
                             'multiply: fallback_alpha*(alpha_min+(alpha_max-alpha_min)*r), useful for round*client schedules.')
    parser.add_argument('--byot_client_skew_proxy', default='none',
                        choices=['none', 'label_entropy', 'max_concentration'],
                        help='Client-local label-skew proxy used to downscale BYOT alpha/lambda. '
                             'Computed only from each client local labels and never sent to the server.')
    parser.add_argument('--byot_client_skew_min_scale', type=float, default=0.0,
                        help='Minimum skew-penalty scale when client labels are extremely concentrated.')
    parser.add_argument('--byot_client_skew_power', type=float, default=1.0,
                        help='Power applied to the client skew reliability before scaling BYOT alpha/lambda.')
    parser.add_argument('--byot_class_proxy', default='none',
                        choices=['none', 'label_count', 'teacher_label_prob',
                                 'teacher_correctness', 'teacher_label_prob_count'],
                        help='Client-class-level reliability proxy used to choose one BYOT KD alpha per class inside each client.')
    parser.add_argument('--byot_class_alpha_min', type=float, default=0.0,
                        help='Minimum class-wise BYOT alpha when --byot_class_proxy is enabled.')
    parser.add_argument('--byot_class_alpha_max', type=float, default=1.0,
                        help='Maximum class-wise BYOT alpha when --byot_class_proxy is enabled.')
    parser.add_argument('--byot_class_alpha_mode', default='map',
                        choices=['map', 'multiply'],
                        help='How class reliability is applied. map: alpha_min+(alpha_max-alpha_min)*r. '
                             'multiply: fallback_alpha*(alpha_min+(alpha_max-alpha_min)*r).')
    parser.add_argument('--byot_class_count_smoothing', type=float, default=1.0,
                        help='Smoothing added to per-class counts for count-based class-wise BYOT alpha.')
    parser.add_argument('--byot_round_lambda_schedule', default='none',
                        choices=['none', 'linear', 'cosine'],
                        help='Round-wise schedule for BYOT alpha/lambda. '
                             'Useful with --byot_branch_objective kd_only, where byot_alpha is lambda_max.')
    parser.add_argument('--byot_round_lambda_min', type=float, default=0.0,
                        help='Minimum round-wise BYOT alpha/lambda at the beginning of training.')
    parser.add_argument('--byot_round_lambda_warmup', type=int, default=0,
                        help='Number of communication rounds used to ramp BYOT alpha/lambda from min to max. '
                             'If 0, args.round is used.')
    
    parser.add_argument('--warmup_rounds', type=int, default=0, help='Number of rounds for warmup (Teacher only training)')
    parser.add_argument('--amp', action='store_true', help='Use PyTorch AMP (mixed precision)')

    parser.add_argument('--kd_conf_threshold', type=float, default=0.0,
                    help='Selective KD: keep samples with teacher confidence >= threshold (0 disables)')
    parser.add_argument('--kd_min_keep_ratio', type=float, default=0.0,
                        help='Selective KD fallback: if none kept, keep top-k ratio (0 disables)')

    parser.add_argument('--use_ema_teacher', action='store_true', help='Maintain EMA of global model and save it')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay for server teacher')
    parser.add_argument('--save_best_ckpt', action='store_true', help='Save best checkpoint (global + ema if enabled)')
    parser.add_argument('--ckpt_dir', default=None, help='Checkpoint directory (default: logdir)')
    parser.add_argument(
        '--best_by_ema',
        action='store_true',
        help='If set, use EMA test accuracy as the metric for best/acc curves (requires --use_ema_teacher).'
    )
    
    parser.add_argument('--use_sd', action='store_true', help='Combine Self-Distillation with the selected algorithm')
    
    parser.add_argument('--train_file', type=str, default='train', 
                        help='python file name for training (e.g., train, train_feat)')
    
    parser.add_argument('--use_adaptive', action='store_true', help='Use Truth-based Adaptive KD (Correctness Gating)')
    parser.add_argument('--use_orthogonal', action='store_true', help='Use Orthogonal Feature Loss instead of MSE')

    parser.add_argument('--use_ensemble', action='store_true', help='Use Ensemble of all branches for prediction')
    parser.add_argument('--use_cosine', action='store_true', help='Use Cosine Similarity for Feature Distillation instead of MSE')
    
    parser.add_argument('--use_norm_agg', action='store_true', help='Use norm-based class-wise aggregation')
    
    parser.add_argument('--tau', type=float, default=1.0, help='tau for logit adjustment')
    parser.add_argument('--alpha_t', type=float, default=2.0, help='sensitivity for dynamic temperature')
    parser.add_argument('--gamma', type=float, default=2.0, help='gamma for focal loss')
    
    parser.add_argument('--use_fedprox', action='store_true', help='BYOT 학습 시 FedProx 근접항 추가')
    parser.add_argument('--use_moon', action='store_true', help='BYOT 학습 시 MOON 대조 학습 추가')
    parser.add_argument('--use_fedrcl', action='store_true', help='BYOT 학습 시 FedRCL 추가')
    parser.add_argument('--log_client_drift', action='store_true',
                        help='Log client update drift before aggregation.')
    parser.add_argument('--drift_log_interval', type=int, default=1,
                        help='Log client drift every N rounds when --log_client_drift is enabled.')
    parser.add_argument('--log_gradient_probe', action='store_true',
                        help='Probe CE/KD/combined gradient dissimilarity before local training.')
    parser.add_argument('--gradient_probe_interval', type=int, default=50,
                        help='Probe gradient dissimilarity every N rounds when --log_gradient_probe is enabled.')
    parser.add_argument('--gradient_probe_batches', type=int, default=1,
                        help='Number of local batches per sampled client used for gradient probing.')
    
    
    parser.add_argument('--min_threshold', type=float, default=0.8, help='시작 임계값 (Dynamic Threshold용)')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='전체 데이터를 학습할 워밍업 에폭 수')
    
    parser.add_argument('--partition_groups', type=int, default=8, help='Number of groups for noniid_grouping')
    parser.add_argument('--imbalance_factor', type=float, default=100.0, help='Imbalance factor for noniid_longtail')

    args = parser.parse_args()
    return args

def main():
    # --- 1. 초기 설정 (Initialization) ---
    args = get_args()
    
    if args.time > 0:
        time.sleep(args.time)
    device = torch.device(args.device)
    logger, log_file_name = init_logger(args)
    
    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=log_file_name.replace("/", "__"),
            group=os.path.dirname(log_file_name),
            job_type=os.path.basename(log_file_name),
            config=vars(args),
            dir=args.logdir,
        )
    
    init_seed(args)
    
    try:
        train_module = importlib.import_module(args.train_file)
        logger.info(f"Loaded training module: {args.train_file}.py")
    except ImportError:
        logger.error(f"Error: Could not import module '{args.train_file}'. Make sure the file exists.")
        return
    
    best_accuracy = 0.0
    
    # 1. 데이터셋 로드
    global_train_dataset, global_val_dataset, global_test_dataset = get_global_dataset(args)
    
    # 2. 데이터 파티셔닝 (Client들에게 데이터 나누기)
    client_data_map = partition_data(global_train_dataset, args, logger)
    
    # 3. 클라이언트별 데이터셋 생성
    client_datasets = get_client_datasets(global_train_dataset, client_data_map, args)
    
    # 4. [핵심] 클라이언트별 데이터로더 생성 (NameError 해결!)
    client_dataloaders = get_client_dataloaders(client_datasets, args)
    
    # 5. 글로벌 데이터로더 생성
    global_train_dataloader, global_val_dataloader, global_test_dataloader = get_global_dataloader(global_train_dataset, global_val_dataset, global_test_dataset, args)

    # ==========================================================================
    # [통합 수정] 1. 데이터셋별 클래스 수 & 채널 수 자동 설정
    # ==========================================================================
    if args.num_classes is None:
        if args.dataset == 'tinyimagenet':
            args.num_classes = 200
        elif args.dataset == 'cifar100':
            args.num_classes = 100
        elif args.dataset == 'emnist':
            args.num_classes = 47
            args.in_channels = 1  # EMNIST는 흑백
        elif args.dataset == 'cifar10':
            args.num_classes = 10
        elif hasattr(global_train_dataset, 'num_classes'):
            args.num_classes = int(global_train_dataset.num_classes)
        else:
            raise ValueError(f"Cannot infer num_classes for dataset={args.dataset}. Pass --num_classes explicitly.")
    
    print(f"🔥 [Auto Setup] Dataset: {args.dataset} | Classes: {args.num_classes} | Channels: {args.in_channels}")

    # ==========================================================================
    # [통합 수정] 2. 모델 선택 로직 (Base vs Ours)
    # args.model이 'resnet18'이면 Base(resnet_cifar.py) 로드
    # args.model이 'resnet18_byot'면 Ours(resnet_byot.py) 로드
    # ==========================================================================
    
    if args.model == 'resnet18':
        # [Base] SD 없이 순수 성능 측정용 (Adapted ResNet)
        print(f"🛡️ Loading Base Model: ResNet18_Cifar (Stride=1, No MaxPool)")
        base_global_model = ResNet18_cifar10(num_classes=args.num_classes, in_channels=args.in_channels).to(device)
        global_model = ResNet18_cifar10(num_classes=args.num_classes, in_channels=args.in_channels).to(device)

    elif args.model == 'resnet18_byot':
        # [Ours] SD 적용 모델 (Branch 포함)
        print(f"🚀 Loading Ours Model: ResNet18_BYOT (With Branches)")
        base_global_model = multi_resnet18_kd(num_classes=args.num_classes, in_channels=args.in_channels).to(device)
        global_model = multi_resnet18_kd(num_classes=args.num_classes, in_channels=args.in_channels).to(device)

    elif args.model == 'mobilenet':
        print(f"🛡️ Loading Base Model: MobileNetV2")
        base_global_model = MobileNetV2(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            last_fc=args.last_fc,
            no_init=args.no_init,
        ).to(device)
        global_model = MobileNetV2(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            last_fc=args.last_fc,
            no_init=True,
        ).to(device)

    elif args.model == 'mobilenet_byot':
        print(f"🚀 Loading Ours Model: MobileNetV2_BYOT (With Branches)")
        base_global_model = MobileNetV2BYOT(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            no_init=args.no_init,
        ).to(device)
        global_model = MobileNetV2BYOT(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            no_init=True,
        ).to(device)
        
    else:
        # 그 외 모델(MobileNet 등)은 기존 init_net 사용
        base_global_model = init_net(global_train_dataset, 1, args, device, True)[0]
        global_model = init_net(global_train_dataset, 1, args, device)[0]

    # 가중치 복사 (Base -> Global)
    global_model.load_state_dict(base_global_model.state_dict())
    
    ema_model = None
    if getattr(args, "use_ema_teacher", False):
        ema_model = copy.deepcopy(global_model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    client_nets = {i: copy.deepcopy(global_model).to('cpu') for i in range(args.n_clients)}

    moment_first, moment_second = fl_utils.init_server_optimizers(global_model)

    # [수정] 효율성(efficiency) 저장을 위한 리스트 추가
    pkl_dict = {
        'args': vars(args),
        'avg_train_loss': [],
        'test_loss': [],
        'efficiency': [],
        'acc': [],          
        'acc_global': [],   
        'acc_ema': [], 
        'ece': [],          
        'branch_acc': [],   
        'round_time': [],   # 이 줄을 새로 추가
        'client_update_norm': [],
        'client_update_norm_sq': [],
        'client_mean_update_norm': [],
        'client_update_divergence': [],
        'client_relative_drift': [],
        'client_update_cosine': [],
        'gradient_probe_clients': [],
        'gradient_ce_divergence': [],
        'gradient_ce_relative': [],
        'gradient_ce_norm': [],
        'gradient_ce_norm_sq': [],
        'gradient_ce_mean_norm': [],
        'gradient_ce_cosine': [],
        'gradient_kd_divergence': [],
        'gradient_kd_relative': [],
        'gradient_kd_norm': [],
        'gradient_kd_norm_sq': [],
        'gradient_kd_mean_norm': [],
        'gradient_kd_cosine': [],
        'gradient_combined_divergence': [],
        'gradient_combined_relative': [],
        'gradient_combined_norm': [],
        'gradient_combined_norm_sq': [],
        'gradient_combined_mean_norm': [],
        'gradient_combined_cosine': [],
        'gradient_ce_kd_cross': [],
        'gradient_ce_kd_corr': [],
        'gradient_ce_kd_cosine': [],
        'gradient_ce_kd_distance': [],
        'gradient_kd_ce_norm_ratio': [],
        'kd_info_teacher_entropy': [],
        'kd_info_teacher_entropy_norm': [],
        'kd_info_teacher_true_label_prob': [],
        'kd_info_teacher_non_target_mass': [],
        'kd_info_teacher_top2_margin': [],
        'kd_info_teacher_confidence': [],
        'byot_effective_alpha_mean': [],
        'byot_effective_alpha_min': [],
        'byot_effective_alpha_max': [],
        'max': 0, 'avg_10': 0, 'avg_30': 0, 'avg_50': 0
    }
    last_10 = []
    lr = args.lr

    # --- 4. 메인 학습 루프 ---
    m = max(int(args.sample_fraction * args.n_clients), 1) # 참여 클라이언트 수

    for round in range(args.round):
        args.current_round = round
        logger.info(f'round:{round}')
        t0 = time.time()

        # 1. 클라이언트 랜덤 선택 (이중 루프 제거함)
        clients_this_round = np.random.choice(range(args.n_clients), m, replace=False)
        
        # 2. 선택된 클라이언트의 모델과 데이터로더 가져오기
        nets_this_round = {i: client_nets[i].to(device) for i in clients_this_round}
        # [수정] get_client_dataloaders 함수가 아니라, 위에서 만든 변수(client_dataloaders)를 써야 함
        dataloaders_this_round = {i: client_dataloaders[i] for i in clients_this_round}

        # 3. 글로벌 모델 가중치 고정 (학습 안 되게)
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False
        global_w = global_model.state_dict()
        
        # (이후 로직 유지...)
        prev_nets = None
        prev_global_model = None
        if args.alg == 'fedacg':
            prev_global_model = copy.deepcopy(global_model)
        if args.alg == 'moon' or getattr(args, 'use_moon', False):
            prev_nets = copy.deepcopy(nets_this_round)
            for _, net in prev_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False

        # 클라이언트 모델 초기화 (글로벌 모델로 덮어쓰기)
        for client_idx, net in nets_this_round.items():
            net.load_state_dict(global_w)

        # Gradient dissimilarity probe at the shared round-start model.
        total_batches = sum([len(dataloaders_this_round[j]) for j in dataloaders_this_round if dataloaders_this_round[j] is not None])
        fed_avg_freqs = [len(dataloaders_this_round[j]) / total_batches if dataloaders_this_round[j] is not None else 0.0 for j in dataloaders_this_round]
        gradient_probe_metrics = {}
        should_probe_gradient = (
            getattr(args, 'log_gradient_probe', False)
            and getattr(args, 'gradient_probe_interval', 1) > 0
            and round % getattr(args, 'gradient_probe_interval', 1) == 0
        )
        if should_probe_gradient:
            gradient_probe_metrics = compute_gradient_drift_probe(
                nets_this_round, dataloaders_this_round, fed_avg_freqs, device, args
            )
            if gradient_probe_metrics:
                logger.info(
                    "Gradient probe: "
                    f"CE={gradient_probe_metrics['gradient_ce_divergence']:.6f}, "
                    f"KD={gradient_probe_metrics['gradient_kd_divergence']:.6f}, "
                    f"Combined={gradient_probe_metrics['gradient_combined_divergence']:.6f}, "
                    f"RelCombined={gradient_probe_metrics['gradient_combined_relative']:.6f}"
                )

        old_w = copy.deepcopy(global_model.state_dict())
        
        args.current_round = round + 1
        
        local_results = train_module.train_local_net(
            dataloaders=dataloaders_this_round, nets=nets_this_round, 
            global_model=global_model, prev_nets=prev_nets, prev_global_model=prev_global_model, 
            device=device, round=round, lr=lr, args=args, logger=logger,
        )

        # 결과 처리
        if len(local_results) == 8:
            (
                avg_loss, lr, avg_ratio, avg_rfd, avg_feat_ratio,
                avg_byot_alpha_mean, avg_byot_alpha_min, avg_byot_alpha_max,
            ) = local_results
        elif len(local_results) == 5:
            avg_loss, lr, avg_ratio, avg_rfd, avg_feat_ratio = local_results
            avg_byot_alpha_mean = float(getattr(args, "byot_alpha", 0.0))
            avg_byot_alpha_min = avg_byot_alpha_mean
            avg_byot_alpha_max = avg_byot_alpha_mean
        elif len(local_results) == 3:
            avg_loss, lr, avg_ratio = local_results
            avg_rfd, avg_feat_ratio = 0.0, 1.0
            avg_byot_alpha_mean = float(getattr(args, "byot_alpha", 0.0))
            avg_byot_alpha_min = avg_byot_alpha_mean
            avg_byot_alpha_max = avg_byot_alpha_mean
        else:
            avg_loss, lr = local_results
            avg_ratio, avg_rfd, avg_feat_ratio = 1.0, 0.0, 1.0
            avg_byot_alpha_mean = float(getattr(args, "byot_alpha", 0.0))
            avg_byot_alpha_min = avg_byot_alpha_mean
            avg_byot_alpha_max = avg_byot_alpha_mean

        # 로그 저장
        pkl_dict['avg_train_loss'].append(avg_loss)
        pkl_dict['efficiency'].append(avg_ratio)
        if 'rfd' not in pkl_dict: pkl_dict['rfd'] = []
        pkl_dict['rfd'].append(avg_rfd)
        if 'feat_ratio' not in pkl_dict: pkl_dict['feat_ratio'] = []
        pkl_dict['feat_ratio'].append(avg_feat_ratio)
        pkl_dict['byot_effective_alpha_mean'].append(avg_byot_alpha_mean)
        pkl_dict['byot_effective_alpha_min'].append(avg_byot_alpha_min)
        pkl_dict['byot_effective_alpha_max'].append(avg_byot_alpha_max)
        for gradient_key in [
            'gradient_probe_clients',
            'gradient_ce_divergence',
            'gradient_ce_relative',
            'gradient_ce_norm',
            'gradient_ce_norm_sq',
            'gradient_ce_mean_norm',
            'gradient_ce_cosine',
            'gradient_kd_divergence',
            'gradient_kd_relative',
            'gradient_kd_norm',
            'gradient_kd_norm_sq',
            'gradient_kd_mean_norm',
            'gradient_kd_cosine',
            'gradient_combined_divergence',
            'gradient_combined_relative',
            'gradient_combined_norm',
            'gradient_combined_norm_sq',
            'gradient_combined_mean_norm',
            'gradient_combined_cosine',
            'gradient_ce_kd_cross',
            'gradient_ce_kd_corr',
            'gradient_ce_kd_cosine',
            'gradient_ce_kd_distance',
            'gradient_kd_ce_norm_ratio',
            'kd_info_teacher_entropy',
            'kd_info_teacher_entropy_norm',
            'kd_info_teacher_true_label_prob',
            'kd_info_teacher_non_target_mass',
            'kd_info_teacher_top2_margin',
            'kd_info_teacher_confidence',
        ]:
            pkl_dict[gradient_key].append(gradient_probe_metrics.get(gradient_key))

        # 모델 집계 (Aggregation)
        drift_metrics = {}
        should_log_drift = (
            getattr(args, 'log_client_drift', False)
            and getattr(args, 'drift_log_interval', 1) > 0
            and round % getattr(args, 'drift_log_interval', 1) == 0
        )
        if should_log_drift:
            drift_metrics = compute_client_update_drift(old_w, nets_this_round, fed_avg_freqs)

        for drift_key in [
            'client_update_norm',
            'client_update_norm_sq',
            'client_mean_update_norm',
            'client_update_divergence',
            'client_relative_drift',
            'client_update_cosine',
        ]:
            pkl_dict[drift_key].append(drift_metrics.get(drift_key))

        if drift_metrics:
            logger.info(
                "Client drift: "
                f"div={drift_metrics['client_update_divergence']:.6f}, "
                f"rel={drift_metrics['client_relative_drift']:.6f}, "
                f"norm={drift_metrics['client_update_norm']:.6f}, "
                f"cos={drift_metrics['client_update_cosine']:.6f}"
            )

        if getattr(args, 'use_norm_agg', False):
            global_w = norm_based_classwise_aggregation(global_model, nets_this_round, fed_avg_freqs)
        else:
            global_w = fl_utils.aggregate_models(args, nets_this_round, fed_avg_freqs, global_w)

        # 서버 최적화 (Server Momentum 등)
        global_w, moment_first, moment_second = fl_utils.apply_server_side_optimization(
            args, global_w, old_w, nets_this_round, fed_avg_freqs,
            moment_first, moment_second
        )

        t1 = time.time()
        logger.info(f'1 Round train time: {t1 -t0} | Efficiency: {avg_ratio:.4f}')
        pkl_dict['round_time'].append(t1 - t0)

        # 글로벌 모델 업데이트 및 평가
        global_model.load_state_dict(global_w)

        if ema_model is not None:
            update_ema_model(ema_model, global_model, decay=float(args.ema_decay))

        # Test
        test_loss, test_acc_global, test_ece, test_branches = test(global_model, global_test_dataloader, device, args)
        
        pkl_dict['acc_global'].append(test_acc_global)
        pkl_dict['test_loss'].append(test_loss)
        pkl_dict['ece'].append(test_ece)
        pkl_dict['branch_acc'].append(test_branches)

        # Best Accuracy 갱신 및 저장
        if test_acc_global > best_accuracy:
             best_accuracy = test_acc_global
             if getattr(args, "save_best_ckpt", False):
                ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else args.logdir
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt = {
                    "round": round,
                    "best_accuracy": best_accuracy,
                    "args": vars(args),
                    "global_model": global_model.state_dict(),
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"{log_file_name}_best.pt"))
                
        if wandb_run is not None:
            metrics = {
                "round": round,
                "train/avg_loss": avg_loss,
                "train/efficiency": avg_ratio,
                "train/rfd": avg_rfd,
                "train/feat_ratio": avg_feat_ratio,
                "train/byot_effective_alpha_mean": avg_byot_alpha_mean,
                "train/byot_effective_alpha_min": avg_byot_alpha_min,
                "train/byot_effective_alpha_max": avg_byot_alpha_max,
                "test/loss": test_loss,
                "test/acc_global": test_acc_global,
                "test/ece": test_ece,
                "test/best_acc": best_accuracy,
                "time/round_sec": t1 - t0,
                "lr": lr,
            }

            for i, acc in enumerate(test_branches):
                metrics[f"test/branch_acc_{i}"] = acc
            for drift_key, drift_value in drift_metrics.items():
                metrics[f"drift/{drift_key}"] = drift_value
            for gradient_key, gradient_value in gradient_probe_metrics.items():
                metrics[f"gradient_probe/{gradient_key}"] = gradient_value

            wandb.log(metrics, step=round)

        # 로그 출력
        logger.info(f"Round {round} result: Acc={test_acc_global:.2f}, Loss={test_loss:.4f}, Best={best_accuracy:.2f}")
        
        if len(test_branches) > 1:
            logger.info(f" └─ [Branch Acc] B1(Shallow):{test_branches[0]:.2f}% | B2:{test_branches[1]:.2f}% | B3:{test_branches[2]:.2f}% | Teacher:{test_branches[3]:.2f}%")

        for i in clients_this_round:
            client_nets[i].to('cpu') # 다시 CPU로 돌려보냄
            
        if prev_nets is not None:
            del prev_nets # MOON 등에서 복사했던 이전 모델 메모리 해제
            
        torch.cuda.empty_cache() # 찌꺼기 메모리 완전 삭제

    # --- 5. 최종 결과 저장 ---
    # (기존 코드 유지)
    with open(os.path.join(args.logdir, log_file_name + '.pkl'), 'wb') as f:
        pickle.dump(pkl_dict, f)
        
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    main()
