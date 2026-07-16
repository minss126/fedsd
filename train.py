import torch
import torch.optim as optim
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import copy
import time
from torch.cuda.amp import autocast, GradScaler

from algs.decorr import FedDecorrLoss
from algs.fedrcl import RCLloss
from algs.fedsol import *

from torch.cuda.amp import autocast, GradScaler

def unpack_model_output(out):
    if isinstance(out, tuple) and len(out) == 8:
        # BYOT Model: (output, m1, m2, m3, final_fea, f1, f2, f3)
        # 0: logits, 4: final_features
        return out[4], out[0] 
    elif isinstance(out, tuple) and len(out) == 2:
        # Standard Model: (features, logits)
        return out[0], out[1]
    else:
        # Logits only or unexpected structure
        return None, out

def compute_sd_loss(out, target, device, args):
    """
    [Modified Self-Distillation Loss]
    1. Naive KD: 기존 방식 (Soft Target 모방, Feature 모방)
    2. Adaptive KD (--use_adaptive): 선생님이 정답을 맞췄을 때만 KD Loss 반영 (Logit 필터링)
    3. Orthogonal Feature (--use_orthogonal): 선생님 특징과 직교하도록 학습 (Feature 다양화)
    """
    if not getattr(args, 'use_sd', False): 
        return 0.0

    # 설정값 로드
    temperature = getattr(args, 'temperature', 3.0) 
    alpha = getattr(args, 'byot_alpha', 0.3)  # KD(Logit) 가중치
    beta = getattr(args, 'byot_beta', 0.1)    # Feature 가중치
    
    # [NEW] main.py에서 추가한 인자들
    use_adaptive = getattr(args, 'use_adaptive', False)
    use_orthogonal = getattr(args, 'use_orthogonal', False)
    
    criterion_ce = nn.CrossEntropyLoss().to(device)
    # Adaptive 적용을 위해 reduction='none'으로 설정 (샘플별 Loss 계산)
    criterion_kl = nn.KLDivLoss(reduction='none').to(device) 
    criterion_mse = nn.MSELoss().to(device)

    # BYOT 모델 (8개 리턴)
    if isinstance(out, tuple) and len(out) == 8:
        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
        
        # 1. Student CE Loss (학생 본연의 의무: 정답 맞추기)
        loss_ce = (criterion_ce(m1, target) + 
                   criterion_ce(m2, target) + 
                   criterion_ce(m3, target))
        
        # 2. KD Loss (Logit: 선생님 답안지 참고)
        loss_kd = 0.0
        if alpha > 0:
            with torch.no_grad():
                teacher_prob = F.softmax(output / temperature, dim=1)
                # 선생님의 예측값 (Adaptive 판단용)
                teacher_pred = output.argmax(dim=1)

            # 각 Branch별 KL Divergence 계산 (Batch, Class) -> (Batch,) 합산
            kl_1 = criterion_kl(F.log_softmax(m1 / temperature, dim=1), teacher_prob).sum(dim=1)
            kl_2 = criterion_kl(F.log_softmax(m2 / temperature, dim=1), teacher_prob).sum(dim=1)
            kl_3 = criterion_kl(F.log_softmax(m3 / temperature, dim=1), teacher_prob).sum(dim=1)
            
            raw_kd_loss = (kl_1 + kl_2 + kl_3) * (temperature ** 2)

            if use_adaptive:
                # [Adaptive] 선생님이 정답(Target)을 맞춘 샘플만 학습 (True=1, False=0)
                correct_mask = (teacher_pred == target).float()
                
                # 정답 맞춘 샘플이 하나라도 있으면 평균, 없으면 0
                if correct_mask.sum() > 0:
                    loss_kd = (raw_kd_loss * correct_mask).sum() / correct_mask.sum()
                else:
                    loss_kd = 0.0
            else:
                # [Naive] 그냥 전체 평균
                loss_kd = raw_kd_loss.mean()

        # 3. Feature Loss (Feature: 선생님 생각 훔치기)
        loss_feat = 0.0
        if beta > 0:
            student_feats = [f1, f2, f3]
            teacher_feat = final_fea.detach()
            
            # [NEW] main.py에서 추가한 인자 로드
            use_cosine = getattr(args, 'use_cosine', False)
            
            if use_orthogonal:
                # [Orthogonal] 선생님이랑 달라져라! (Cosine Similarity 절댓값 최소화)
                t_flat = teacher_feat.view(teacher_feat.size(0), -1)
                for s_feat in student_feats:
                    s_flat = s_feat.view(s_feat.size(0), -1)
                    cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)
                    loss_feat += torch.abs(cos_sim).mean()
                    
            elif use_cosine:
                # [NEW: Cosine Imitation] 선생님의 방향성만 유연하게 따라가라!
                # 두 특징 벡터의 코사인 유사도가 1이 되도록 (방향이 같아지도록) 학습합니다.
                t_flat = teacher_feat.view(teacher_feat.size(0), -1)
                for s_feat in student_feats:
                    s_flat = s_feat.view(s_feat.size(0), -1)
                    cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)
                    # 코사인 유사도의 최댓값은 1이므로, (1 - cos_sim)을 최소화합니다.
                    loss_feat += (1.0 - cos_sim).mean()
                    
            else:
                # [Naive] 선생님이랑 크기와 방향 모두 똑같아져라! (MSE)
                for s_feat in student_feats:
                    loss_feat += criterion_mse(s_feat, teacher_feat)
        # 최종 합산
        return (1 - alpha) * loss_ce + alpha * loss_kd + beta * loss_feat
    
    return 0.0

def compute_fl_regularization(net, global_model, prev_net, x, final_fea, device, args, target=None):
    """
    여러 연합학습 베이스라인(FedProx, MOON, FedRCL, FedDecorr)의 글로벌 규제항(Loss)을 통합 계산합니다.
    """
    reg_loss = 0.0
    
    # 1. FedProx 결합
    if getattr(args, 'use_fedprox', False) and global_model is not None:
        proximal_term = 0.0
        for w_local, w_global in zip(net.parameters(), global_model.parameters()):
            proximal_term += (w_local - w_global).pow(2).sum()
        reg_loss += (args.mu / 2) * proximal_term
        
    # 2. MOON 결합
    if getattr(args, 'use_moon', False) and global_model is not None and prev_net is not None:
        cos = torch.nn.CosineSimilarity(dim=-1)
        feat_s = final_fea.view(final_fea.size(0), -1)
        with torch.no_grad():
            out_g = global_model(x)
            feat_g = out_g[4].view(out_g[4].size(0), -1) if isinstance(out_g, tuple) and len(out_g) == 8 else (out_g[0].view(out_g[0].size(0), -1) if isinstance(out_g, tuple) else out_g.view(out_g.size(0), -1))
            out_p = prev_net(x)
            feat_p = out_p[4].view(out_p[4].size(0), -1) if isinstance(out_p, tuple) and len(out_p) == 8 else (out_p[0].view(out_p[0].size(0), -1) if isinstance(out_p, tuple) else out_p.view(out_p.size(0), -1))
        posi = cos(feat_s, feat_g).reshape(-1, 1)
        nega = cos(feat_s, feat_p).reshape(-1, 1)
        logits_moon = torch.cat((posi, nega), dim=1) / getattr(args, "temperature", 0.5)
        labels_moon = torch.zeros(x.size(0)).long().to(device)
        import torch.nn as nn
        reg_loss += args.mu * nn.CrossEntropyLoss()(logits_moon, labels_moon)
        
    # 3. FedRCL 결합 (새로 추가됨)
    if getattr(args, 'use_fedrcl', False) and final_fea is not None and target is not None:
        from algs.fedrcl import RCLloss
        criterion_rcl = RCLloss()
        features_norm = F.normalize(final_fea.view(final_fea.size(0), -1), p=2, dim=1)
        cosine_sim_matrix = torch.mm(features_norm, features_norm.t())
        reg_loss += args.mu * criterion_rcl(cosine_sim_matrix=cosine_sim_matrix, x=x, target=target, args=args)
        
    # 4. FedDecorr 결합 (새로 추가됨)
    if getattr(args, 'feddecorr', False) and final_fea is not None:
        from algs.decorr import FedDecorrLoss
        feddecorr = FedDecorrLoss()
        feat_flat = final_fea.view(final_fea.size(0), -1)
        reg_loss += args.feddecorr_coef * feddecorr(feat_flat)
        
    return reg_loss

def _extract_dataset_targets(dataset):
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_targets = _extract_dataset_targets(dataset.dataset)
        if parent_targets is not None:
            return parent_targets[dataset.indices]

    for attr in ("targets", "target", "labels"):
        if hasattr(dataset, attr):
            targets = getattr(dataset, attr)
            if isinstance(targets, torch.Tensor):
                return targets.detach().cpu().numpy()
            return np.asarray(targets)

    if hasattr(dataset, "samples"):
        return np.asarray([sample[1] for sample in dataset.samples])

    return None

def _clamp_unit(value):
    return min(max(float(value), 0.0), 1.0)

def _infer_num_classes_from_targets(targets, args):
    configured = getattr(args, "num_classes", None)
    if configured is not None:
        try:
            configured = int(configured)
            if configured > 0:
                return configured
        except (TypeError, ValueError):
            pass

    if targets is None or len(targets) == 0:
        return 1
    return max(int(np.max(targets)) + 1, 1)

def get_effective_byot_alpha(train_dataloader, args):
    base_alpha = float(getattr(args, "byot_alpha", 0.15))
    alpha = base_alpha

    schedule = getattr(args, "byot_round_lambda_schedule", "none")
    if schedule != "none":
        total_rounds = max(int(getattr(args, "round", 1)), 1)
        warmup_rounds = int(getattr(args, "byot_round_lambda_warmup", 0))
        if warmup_rounds <= 0:
            warmup_rounds = total_rounds

        current_round = int(getattr(args, "current_round", 0))
        progress = _clamp_unit(current_round / max(1, warmup_rounds))

        if schedule == "linear":
            schedule_scale = progress
        elif schedule == "cosine":
            schedule_scale = 0.5 - 0.5 * np.cos(np.pi * progress)
        else:
            schedule_scale = 1.0

        min_alpha = float(getattr(args, "byot_round_lambda_min", 0.0))
        alpha = min_alpha + (base_alpha - min_alpha) * schedule_scale

    if getattr(args, "beta_aware_byot_alpha", False):
        partition = getattr(args, "partition", "")
        if partition == "iid":
            beta_scale = 1.0
        elif partition == "noniid":
            beta = max(float(getattr(args, "beta", 0.0)), 0.0)
            beta_ref = max(float(getattr(args, "alpha_beta_ref", 0.5)), 1e-8)
            min_scale = _clamp_unit(getattr(args, "alpha_min_scale", 0.2))
            beta_scale = max(min_scale, min(1.0, beta / beta_ref))
        else:
            beta_scale = 1.0
        alpha *= beta_scale

    if getattr(args, "adaptive_byot_alpha", False):
        targets = _extract_dataset_targets(train_dataloader.dataset)
        if targets is None or len(targets) == 0:
            return alpha

        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        num_classes = _infer_num_classes_from_targets(targets, args)
        counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
        probs = counts[counts > 0] / max(1.0, counts.sum())

        if num_classes <= 1 or len(probs) == 0:
            entropy_norm = 0.0
        else:
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            entropy_norm = float(entropy / np.log(num_classes))
            entropy_norm = _clamp_unit(entropy_norm)

        min_scale = _clamp_unit(getattr(args, "alpha_min_scale", 0.2))
        power = max(float(getattr(args, "alpha_entropy_power", 1.0)), 1e-8)
        entropy_scale = min_scale + (1.0 - min_scale) * (entropy_norm ** power)
        alpha *= entropy_scale

    alpha *= float(getattr(args, "byot_server_lambda_scale", 1.0))

    return alpha

def get_client_skew_scale(net, train_dataloader, device, args):
    proxy = getattr(args, "byot_client_skew_proxy", "none")
    if proxy == "none":
        return 1.0

    if proxy == "prediction_entropy":
        temperature = float(getattr(args, "temperature", 0.5))
        was_training = net.training
        net.eval()
        prob_sum = None
        total = 0
        with torch.no_grad():
            for x, _ in train_dataloader:
                x = x.to(device, non_blocking=True)
                out = net(x)
                if isinstance(out, tuple) and len(out) == 8:
                    logits = out[0]
                elif isinstance(out, tuple):
                    logits = out[-1]
                else:
                    logits = out
                probs = F.softmax(logits / temperature, dim=1)
                batch_sum = probs.sum(dim=0)
                prob_sum = batch_sum if prob_sum is None else prob_sum + batch_sum
                total += int(probs.size(0))
        if was_training:
            net.train()
        if prob_sum is None or total <= 0:
            return 1.0

        mean_prob = (prob_sum / float(total)).detach().cpu().numpy().astype(np.float64)
        mean_prob = mean_prob[mean_prob > 0]
        num_classes = max(int(prob_sum.numel()), 2)
        entropy = -np.sum(mean_prob * np.log(mean_prob + 1e-12))
        reliability = entropy / np.log(num_classes)
    else:
        targets = _extract_dataset_targets(train_dataloader.dataset)
        if targets is None or len(targets) == 0:
            return 1.0

        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        num_classes = _infer_num_classes_from_targets(targets, args)
        counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
        total = counts.sum()
        if total <= 0.0:
            return 1.0

        probs = counts[counts > 0] / total
        if proxy == "label_entropy":
            if num_classes <= 1 or len(probs) == 0:
                reliability = 0.0
            else:
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                reliability = entropy / np.log(num_classes)
        elif proxy == "max_concentration":
            reliability = 1.0 - float(counts.max() / total)
            if num_classes > 1:
                reliability = reliability / (1.0 - 1.0 / num_classes)
        else:
            return 1.0

    reliability = _clamp_unit(reliability)
    power = max(float(getattr(args, "byot_client_skew_power", 1.0)), 1e-8)
    min_scale = _clamp_unit(getattr(args, "byot_client_skew_min_scale", 0.0))
    return min_scale + (1.0 - min_scale) * (reliability ** power)

def get_byot_branch_alphas(args, device):
    raw = str(getattr(args, "byot_branch_alphas", "") or "").strip()
    if not raw:
        return None

    values = [v.strip() for v in raw.split(",") if v.strip()]
    if len(values) != 3:
        raise ValueError("--byot_branch_alphas must contain exactly three comma-separated values for B1,B2,B3.")

    alphas = torch.tensor([_clamp_unit(float(v)) for v in values], dtype=torch.float32, device=device)
    return alphas

def get_byot_active_branch_indices(args):
    raw = str(getattr(args, "byot_active_branches", "1,2,3") or "1,2,3").strip()
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("--byot_active_branches must contain at least one branch id from 1,2,3.")

    indices = []
    for value in values:
        branch_id = int(value)
        if branch_id < 1 or branch_id > 3:
            raise ValueError("--byot_active_branches only supports branch ids 1,2,3.")
        idx = branch_id - 1
        if idx not in indices:
            indices.append(idx)
    return indices

def get_label_entropy_norm(train_dataloader, args):
    targets = _extract_dataset_targets(train_dataloader.dataset)
    if targets is None or len(targets) == 0:
        return 1.0

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    num_classes = int(getattr(args, "num_classes", max(int(targets.max()) + 1, 1)))
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    probs = counts[counts > 0] / max(1.0, counts.sum())

    if num_classes <= 1 or len(probs) == 0:
        return 0.0

    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return _clamp_unit(entropy / np.log(num_classes))

def get_byot_gated_active_branch_indices(train_dataloader, args):
    gate = getattr(args, "byot_branch_gate", "none")
    if gate == "none":
        return get_byot_active_branch_indices(args), None

    entropy_norm = get_label_entropy_norm(train_dataloader, args)
    b1_threshold = _clamp_unit(getattr(args, "branch_entropy_b1_threshold", 0.5))

    if gate == "entropy_no_off":
        if entropy_norm < b1_threshold:
            return [0], entropy_norm
        return [0, 1, 2], entropy_norm

    if gate == "entropy_3stage":
        off_threshold = _clamp_unit(getattr(args, "branch_entropy_off_threshold", 0.2))
        if entropy_norm < off_threshold:
            return [], entropy_norm
        if entropy_norm < b1_threshold:
            return [0], entropy_norm
        return [0, 1, 2], entropy_norm

    return get_byot_active_branch_indices(args), entropy_norm

def branch_weighted_byot_loss(ce_losses, kd_losses, branch_alphas):
    loss = 0.0
    for ce_loss, kd_loss, branch_alpha in zip(ce_losses, kd_losses, branch_alphas):
        loss = loss + (1.0 - branch_alpha) * ce_loss + branch_alpha * kd_loss
    return loss

def reduce_active_branch_loss(loss, active_branch_indices, args):
    if getattr(args, "byot_branch_loss_reduction", "sum") == "mean":
        return loss / max(1, len(active_branch_indices))
    return loss

TRAIN_BRANCH_FREQ_GROUPS = ("low", "mid", "high")
TRAIN_BRANCH_FREQ_METRICS = (
    "true_label_prob",
    "entropy_norm",
    "confidence",
    "acc",
    "teacher_js",
    "prob_mass_low",
    "prob_mass_mid",
    "prob_mass_high",
    "high_low_mass_ratio",
    "local_count",
    "local_ratio",
)

def _dataset_targets_array(dataset):
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        return np.asarray(targets, dtype=np.int64)
    if hasattr(dataset, "target"):
        targets = dataset.target
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        return np.asarray(targets, dtype=np.int64)
    if hasattr(dataset, "samples"):
        return np.asarray([sample[1] for sample in dataset.samples], dtype=np.int64)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_targets = _dataset_targets_array(dataset.dataset)
        return base_targets[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, "dataset") and hasattr(dataset, "dataidxs"):
        base_targets = _dataset_targets_array(dataset.dataset)
        return base_targets[np.asarray(dataset.dataidxs, dtype=np.int64)]
    return np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)

def get_local_class_counts(train_dataloader, args, device):
    if train_dataloader is None or getattr(train_dataloader, "dataset", None) is None:
        return None, None, None
    num_classes = int(getattr(args, "num_classes", 0) or 0)
    if num_classes <= 0:
        return None, None, None
    targets = _dataset_targets_array(train_dataloader.dataset)
    if targets.size == 0:
        return None, None, None
    counts_np = np.bincount(targets, minlength=num_classes).astype(np.float32)
    counts = torch.tensor(counts_np, device=device)
    total = float(counts.sum().detach().item())
    expected = total / max(float(num_classes), 1.0)
    return counts, total, expected

def init_train_branch_freq_stats():
    stats = {}
    for group in TRAIN_BRANCH_FREQ_GROUPS:
        for branch_idx in (1, 2, 3):
            prefix = f"train_branch_freq_{group}_b{branch_idx}"
            stats[f"{prefix}_count"] = 0.0
            for metric in TRAIN_BRANCH_FREQ_METRICS:
                stats[f"{prefix}_{metric}"] = 0.0
    return stats

def update_train_branch_freq_stats(
    stats, branch_logits, teacher_prob, target, class_counts, expected_count, args
):
    if stats is None or class_counts is None or expected_count is None or expected_count <= 0.0:
        return
    low_ratio = float(getattr(args, "train_branch_freq_low_ratio", 0.5))
    high_ratio = float(getattr(args, "train_branch_freq_high_ratio", 1.5))
    temperature = float(getattr(args, "temperature", 0.5))
    num_classes = max(int(teacher_prob.size(1)), 2)
    log_c = math.log(num_classes)

    with torch.no_grad():
        target_counts = class_counts[target].float()
        local_ratio = target_counts / max(float(expected_count), 1e-12)
        group_masks = {
            "low": local_ratio < low_ratio,
            "mid": (local_ratio >= low_ratio) & (local_ratio <= high_ratio),
            "high": local_ratio > high_ratio,
        }
        class_ratio = class_counts.float() / max(float(expected_count), 1e-12)
        class_group_masks = {
            "low": class_ratio < low_ratio,
            "mid": (class_ratio >= low_ratio) & (class_ratio <= high_ratio),
            "high": class_ratio > high_ratio,
        }

        for branch_idx, logits in enumerate(branch_logits, start=1):
            branch_prob = F.softmax(logits / temperature, dim=1)
            entropy_norm = -(branch_prob * torch.log(branch_prob + 1e-8)).sum(dim=1) / log_c
            true_label_prob = branch_prob.gather(1, target.view(-1, 1)).squeeze(1).clamp(0.0, 1.0)
            confidence, pred = branch_prob.max(dim=1)
            mix = 0.5 * (teacher_prob + branch_prob)
            kl_teacher = (
                teacher_prob * (torch.log(teacher_prob + 1e-8) - torch.log(mix + 1e-8))
            ).sum(dim=1)
            kl_branch = (
                branch_prob * (torch.log(branch_prob + 1e-8) - torch.log(mix + 1e-8))
            ).sum(dim=1)
            teacher_js = 0.5 * (kl_teacher + kl_branch) / log_c
            acc = (pred == target).float()
            prob_mass = {}
            for group, class_mask in class_group_masks.items():
                if class_mask.any():
                    prob_mass[group] = branch_prob[:, class_mask].sum(dim=1)
                else:
                    prob_mass[group] = torch.zeros_like(true_label_prob)
            high_low_mass_ratio = prob_mass["high"] / (prob_mass["low"] + 1e-8)

            values = {
                "true_label_prob": true_label_prob,
                "entropy_norm": entropy_norm,
                "confidence": confidence,
                "acc": acc,
                "teacher_js": teacher_js,
                "prob_mass_low": prob_mass["low"],
                "prob_mass_mid": prob_mass["mid"],
                "prob_mass_high": prob_mass["high"],
                "high_low_mass_ratio": high_low_mass_ratio,
                "local_count": target_counts,
                "local_ratio": local_ratio,
            }
            for group, mask in group_masks.items():
                if not mask.any():
                    continue
                prefix = f"train_branch_freq_{group}_b{branch_idx}"
                count = float(mask.float().sum().detach().item())
                stats[f"{prefix}_count"] += count
                for metric, tensor in values.items():
                    stats[f"{prefix}_{metric}"] += float(tensor[mask].sum().detach().item())

def finalize_train_branch_freq_stats(stats):
    if not stats:
        return {}
    finalized = {}
    for group in TRAIN_BRANCH_FREQ_GROUPS:
        for branch_idx in (1, 2, 3):
            prefix = f"train_branch_freq_{group}_b{branch_idx}"
            count = float(stats.get(f"{prefix}_count", 0.0))
            finalized[f"{prefix}_count"] = count
            for metric in TRAIN_BRANCH_FREQ_METRICS:
                key = f"{prefix}_{metric}"
                finalized[key] = (stats.get(key, 0.0) / count) if count > 0 else None
    return finalized

def merge_train_branch_freq_stats(total, update):
    if not update:
        return
    for group in TRAIN_BRANCH_FREQ_GROUPS:
        for branch_idx in (1, 2, 3):
            prefix = f"train_branch_freq_{group}_b{branch_idx}"
            count_key = f"{prefix}_count"
            count = float(update.get(count_key, 0.0) or 0.0)
            total[count_key] = total.get(count_key, 0.0) + count
            for metric in TRAIN_BRANCH_FREQ_METRICS:
                key = f"{prefix}_{metric}"
                value = update.get(key)
                if value is not None:
                    total[key] = total.get(key, 0.0) + float(value) * count

def get_batch_byot_alpha(alpha, output, branch_outputs, target, args):
    proxy = getattr(args, "byot_alpha_proxy", "none")
    if proxy == "none":
        return alpha

    min_scale = _clamp_unit(getattr(args, "alpha_min_scale", 0.2))

    with torch.no_grad():
        teacher_prob = F.softmax(output, dim=1)
        teacher_conf, teacher_pred = teacher_prob.max(dim=1)

        if proxy == "teacher_conf":
            proxy_score = teacher_conf.mean()
        elif proxy == "teacher_entropy":
            num_classes = max(output.size(1), 2)
            entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1)
            certainty = 1.0 - entropy / math.log(num_classes)
            proxy_score = certainty.clamp(0.0, 1.0).mean()
        elif proxy == "branch_agreement":
            agreements = []
            for branch_out in branch_outputs:
                branch_pred = branch_out.argmax(dim=1)
                agreements.append((branch_pred == teacher_pred).float())
            proxy_score = torch.stack(agreements, dim=0).mean()
        elif proxy == "branch_soft_kl":
            scores = []
            log_c = math.log(max(output.size(1), 2))
            for branch_out in branch_outputs:
                branch_prob = F.softmax(branch_out, dim=1)
                kl = (teacher_prob * (torch.log(teacher_prob + 1e-8) - torch.log(branch_prob + 1e-8))).sum(dim=1)
                scores.append((1.0 - kl / log_c).clamp(0.0, 1.0))
            proxy_score = torch.stack(scores, dim=0).mean()
        elif proxy == "branch_js":
            scores = []
            log_c = math.log(max(output.size(1), 2))
            for branch_out in branch_outputs:
                branch_prob = F.softmax(branch_out, dim=1)
                mix = 0.5 * (teacher_prob + branch_prob)
                kl_teacher = (teacher_prob * (torch.log(teacher_prob + 1e-8) - torch.log(mix + 1e-8))).sum(dim=1)
                kl_branch = (branch_prob * (torch.log(branch_prob + 1e-8) - torch.log(mix + 1e-8))).sum(dim=1)
                js = 0.5 * (kl_teacher + kl_branch)
                scores.append((1.0 - js / log_c).clamp(0.0, 1.0))
            proxy_score = torch.stack(scores, dim=0).mean()
        elif proxy == "teacher_correctness":
            proxy_score = (teacher_pred == target).float().mean()
        else:
            return alpha

        scale = min_scale + (1.0 - min_scale) * proxy_score.clamp(0.0, 1.0)

    return alpha * scale

def get_sample_byot_alpha(alpha, teacher_prob, branch_outputs, target, args, proxy_override=None):
    proxy = proxy_override if proxy_override is not None else getattr(args, "byot_sample_proxy", "none")
    if proxy == "none":
        return None

    min_scale = _clamp_unit(getattr(args, "alpha_min_scale", 0.2))

    with torch.no_grad():
        teacher_conf, teacher_pred = teacher_prob.max(dim=1)

        if proxy == "teacher_conf":
            proxy_score = teacher_conf
        elif proxy == "teacher_entropy":
            num_classes = max(teacher_prob.size(1), 2)
            entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1)
            proxy_score = (1.0 - entropy / math.log(num_classes)).clamp(0.0, 1.0)
        elif proxy == "teacher_margin":
            top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1).values
            if top2.size(1) == 1:
                proxy_score = torch.ones_like(top2[:, 0])
            else:
                proxy_score = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
        elif proxy == "teacher_label_prob":
            proxy_score = teacher_prob.gather(1, target.view(-1, 1)).squeeze(1).clamp(0.0, 1.0)
        elif proxy == "teacher_label_prob_entropy":
            num_classes = max(teacher_prob.size(1), 2)
            label_prob = teacher_prob.gather(1, target.view(-1, 1)).squeeze(1).clamp(0.0, 1.0)
            entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1)
            confidence = (1.0 - entropy / math.log(num_classes)).clamp(0.0, 1.0)
            proxy_score = label_prob * confidence
        elif proxy in {"teacher_label_prob_branch_js", "teacher_label_prob_entropy_branch_js"}:
            num_classes = max(teacher_prob.size(1), 2)
            log_c = math.log(num_classes)
            label_prob = teacher_prob.gather(1, target.view(-1, 1)).squeeze(1).clamp(0.0, 1.0)
            if proxy == "teacher_label_prob_entropy_branch_js":
                entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1)
                label_prob = label_prob * (1.0 - entropy / log_c).clamp(0.0, 1.0)

            js_scores = []
            for branch_out in branch_outputs:
                branch_prob = F.softmax(branch_out, dim=1)
                mix = 0.5 * (teacher_prob + branch_prob)
                kl_teacher = (teacher_prob * (torch.log(teacher_prob + 1e-8) - torch.log(mix + 1e-8))).sum(dim=1)
                kl_branch = (branch_prob * (torch.log(branch_prob + 1e-8) - torch.log(mix + 1e-8))).sum(dim=1)
                js = 0.5 * (kl_teacher + kl_branch)
                js_scores.append((1.0 - js / log_c).clamp(0.0, 1.0))
            branch_agreement = torch.stack(js_scores, dim=0).mean(dim=0)
            proxy_score = (label_prob * branch_agreement).clamp(0.0, 1.0)
        elif proxy == "teacher_correctness":
            proxy_score = (teacher_pred == target).float()
        elif proxy == "branch_agreement":
            scores = []
            for branch_out in branch_outputs:
                scores.append((branch_out.argmax(dim=1) == teacher_pred).float())
            proxy_score = torch.stack(scores, dim=0).mean(dim=0)
        elif proxy == "branch_soft_kl":
            scores = []
            log_c = math.log(max(teacher_prob.size(1), 2))
            for branch_out in branch_outputs:
                branch_prob = F.softmax(branch_out, dim=1)
                kl = (teacher_prob * (torch.log(teacher_prob + 1e-8) - torch.log(branch_prob + 1e-8))).sum(dim=1)
                scores.append((1.0 - kl / log_c).clamp(0.0, 1.0))
            proxy_score = torch.stack(scores, dim=0).mean(dim=0)
        elif proxy == "branch_js":
            scores = []
            log_c = math.log(max(teacher_prob.size(1), 2))
            for branch_out in branch_outputs:
                branch_prob = F.softmax(branch_out, dim=1)
                mix = 0.5 * (teacher_prob + branch_prob)
                kl_teacher = (teacher_prob * (torch.log(teacher_prob + 1e-8) - torch.log(mix + 1e-8))).sum(dim=1)
                kl_branch = (branch_prob * (torch.log(branch_prob + 1e-8) - torch.log(mix + 1e-8))).sum(dim=1)
                js = 0.5 * (kl_teacher + kl_branch)
                scores.append((1.0 - js / log_c).clamp(0.0, 1.0))
            proxy_score = torch.stack(scores, dim=0).mean(dim=0)
        else:
            return None

        scale = min_scale + (1.0 - min_scale) * proxy_score.clamp(0.0, 1.0)

    return alpha * scale

def weighted_byot_kd_loss(student_logits, teacher_prob, sample_alpha, temperature):
    logp = F.log_softmax(student_logits / temperature, dim=1)
    kl_per_sample = F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)
    return (sample_alpha * kl_per_sample).mean()

def estimate_client_byot_alpha(net, train_dataloader, device, args, fallback_alpha):
    proxy = getattr(args, "byot_client_proxy", "none")
    if proxy == "none":
        return fallback_alpha

    alpha_min = float(getattr(args, "byot_client_alpha_min", 0.01))
    alpha_max = float(getattr(args, "byot_client_alpha_max", 0.30))
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min

    temperature = float(getattr(args, "temperature", 0.5))
    was_training = net.training
    net.eval()

    total_score = 0.0
    total_count = 0
    with torch.no_grad():
        for x, target in train_dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            out = net(x)
            if not (isinstance(out, tuple) and len(out) == 8):
                continue

            output, m1, m2, m3 = out[0], out[1], out[2], out[3]
            teacher_prob = F.softmax(output / temperature, dim=1)
            sample_alpha = get_sample_byot_alpha(1.0, teacher_prob, [m1, m2, m3], target, args, proxy_override=proxy)
            if sample_alpha is None:
                continue

            total_score += float(sample_alpha.sum().detach().item())
            total_count += int(sample_alpha.numel())

    if was_training:
        net.train()

    if total_count == 0:
        return fallback_alpha

    reliability = _clamp_unit(total_score / total_count)
    reliability_power = max(float(getattr(args, "byot_client_reliability_power", 1.0)), 1e-8)
    reliability = reliability ** reliability_power
    client_alpha = alpha_min + (alpha_max - alpha_min) * reliability
    if getattr(args, "byot_client_alpha_mode", "map") == "multiply":
        return fallback_alpha * client_alpha
    return client_alpha

def estimate_class_byot_alpha(net, train_dataloader, device, args, fallback_alpha):
    proxy = getattr(args, "byot_class_proxy", "none")
    if proxy == "none":
        return None

    alpha_min = float(getattr(args, "byot_class_alpha_min", 0.0))
    alpha_max = float(getattr(args, "byot_class_alpha_max", 1.0))
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min

    num_classes = int(getattr(args, "num_classes", 100))
    smoothing = max(float(getattr(args, "byot_class_count_smoothing", 1.0)), 0.0)
    temperature = float(getattr(args, "temperature", 0.5))

    counts = torch.zeros(num_classes, device=device)
    score_sums = torch.zeros(num_classes, device=device)

    was_training = net.training
    net.eval()

    with torch.no_grad():
        for x, target in train_dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            out = net(x)
            if not (isinstance(out, tuple) and len(out) == 8):
                continue

            output = out[0]
            teacher_prob = F.softmax(output / temperature, dim=1)
            bincount = torch.bincount(target, minlength=num_classes).float()
            counts += bincount

            if proxy in {"teacher_label_prob", "teacher_label_prob_count"}:
                label_prob = teacher_prob.gather(1, target.view(-1, 1)).squeeze(1).clamp(0.0, 1.0)
                score_sums.scatter_add_(0, target, label_prob)
            elif proxy == "teacher_correctness":
                teacher_pred = teacher_prob.argmax(dim=1)
                correctness = (teacher_pred == target).float()
                score_sums.scatter_add_(0, target, correctness)

    if was_training:
        net.train()

    if counts.sum().item() <= 0:
        return None

    count_reliability = (counts + smoothing) / (counts.max().clamp_min(1.0) + smoothing)
    count_reliability = count_reliability.clamp(0.0, 1.0)

    if proxy == "label_count":
        reliability = count_reliability
    elif proxy in {"teacher_label_prob", "teacher_correctness"}:
        reliability = torch.where(counts > 0, score_sums / counts.clamp_min(1.0), torch.zeros_like(counts))
        reliability = reliability.clamp(0.0, 1.0)
    elif proxy == "teacher_label_prob_count":
        label_reliability = torch.where(counts > 0, score_sums / counts.clamp_min(1.0), torch.zeros_like(counts))
        reliability = (label_reliability.clamp(0.0, 1.0) * count_reliability).clamp(0.0, 1.0)
    else:
        return None

    class_alpha = alpha_min + (alpha_max - alpha_min) * reliability
    if getattr(args, "byot_class_alpha_mode", "map") == "multiply":
        class_alpha = fallback_alpha * class_alpha
    return class_alpha.detach()

def fedavg(net, train_dataloader, optimizer, device, args):
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()

    total_loss = 0.0
    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            
            # [수정] 모델 출력 언패킹 (기존: features, out = net(x))
            raw_out = net(x)
            features, out = unpack_model_output(raw_out)

            loss = criterion(out, target)
            
            if getattr(args, 'use_sd', False):
                loss += compute_sd_loss(raw_out, target, device, args)
            
            total_loss += loss.item()
            
            # features가 None이 아닐 때만 feddecorr 적용
            if args.feddecorr and features is not None:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            loss.backward()
            optimizer.step()

    net.zero_grad()

    return total_loss / max(1, len(train_dataloader)) / args.epochs

def fedrs(net, train_dataloader, optimizer, device, args):
    criterion = nn.CrossEntropyLoss()
    from algs.decorr import FedDecorrLoss
    feddecorr = FedDecorrLoss()

    # 1. 안전한 데이터 분포 추출
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        num_classes = subset_dataset.dataset.num_classes
        subset_targets = np.array(subset_dataset.dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(subset_dataset.dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
        
    # 2. 파라미터명 동기화 및 스케일링 팩터 생성
    rs_alpha = getattr(args, 'fedrs_alpha', 0.5)
    missing_mask = (class_counts == 0).unsqueeze(0).float()
    scaling_factor = 1.0 - missing_mask * (1.0 - rs_alpha)

    total_loss = 0.0
    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device).long()
            optimizer.zero_grad()
            
            # 3. 안전한 모델 출력 언패킹
            raw_out = net(x)
            features, out = unpack_model_output(raw_out)
            
            # Logit 스케일링
            out = out * scaling_factor
            loss = criterion(out, target)
            total_loss += loss.item()
            
            if getattr(args, 'feddecorr', False) and features is not None:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            loss.backward()
            optimizer.step()

    net.zero_grad()
    return total_loss / max(1, len(train_dataloader)) / args.epochs

def fedlc(net, train_dataloader, optimizer, device, args):
    criterion = nn.CrossEntropyLoss()
    from algs.decorr import FedDecorrLoss
    feddecorr = FedDecorrLoss()

    # 1. 안전한 데이터 분포 추출 (에러 방지)
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        num_classes = subset_dataset.dataset.num_classes
        subset_targets = np.array(subset_dataset.dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(subset_dataset.dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
    
    # 2. NaN 방지 마진 계산
    tau = float(getattr(args, "calibration_temp", 1.0))
    margin = tau * (class_counts ** -0.25).unsqueeze(dim=0).to(device)
    margin[margin == float('inf')] = 0 

    total_loss = 0.0
    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device).long()
            optimizer.zero_grad()
            
            # 3. 안전한 모델 출력 언패킹
            raw_out = net(x)
            features, out = unpack_model_output(raw_out)

            # Logit 보정
            out = out - margin
            loss = criterion(out, target)
            total_loss += loss.item()
            
            if getattr(args, 'feddecorr', False) and features is not None:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            loss.backward()
            optimizer.step()

    net.zero_grad()
    return total_loss / max(1, len(train_dataloader)) / args.epochs

def fedprox(net, global_model, train_dataloader, optimizer, device, args):
    total_loss = 0.
    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    
    net.train()
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    global_weight_collector = list(global_model.parameters())

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            
            raw_out = net(x)
            features, out = unpack_model_output(raw_out)
            
            loss = criterion(out, target)
            
            # [추가] Baseline 지표 측정 (Confidence, Entropy)
            with torch.no_grad():
                prob = F.softmax(out, dim=1)
                entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean().item()
                total_entropy += entropy
                
                conf, pred = prob.max(dim=1)
                correct_mask = (pred == target)
                if correct_mask.any():
                    total_correct_conf += conf[correct_mask].mean().item()
                    valid_conf_batches += 1
            
            if getattr(args, 'use_sd', False):
                loss += compute_sd_loss(raw_out, target, device, args)

            if args.feddecorr and features is not None:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

    net.zero_grad()
    
    denom = max(1, len(train_dataloader) * getattr(args, "epochs", 1))
    avg_loss = total_loss / denom
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / denom
    
    return avg_loss, avg_correct_conf, avg_entropy

def moon(net, global_model, previous_net, train_dataloader, optimizer, device, args):
    total_loss = 0.
    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    
    net.train()
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            
            raw_out1 = net(x)
            features1, out = unpack_model_output(raw_out1)
            
            # [추가] Baseline 지표 측정 (Confidence, Entropy)
            with torch.no_grad():
                prob = F.softmax(out, dim=1)
                entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean().item()
                total_entropy += entropy
                
                conf, pred = prob.max(dim=1)
                correct_mask = (pred == target)
                if correct_mask.any():
                    total_correct_conf += conf[correct_mask].mean().item()
                    valid_conf_batches += 1

            if features1.dim() > 2:
                features1 = features1.view(features1.size(0), -1)

            with torch.no_grad():
                raw_out2 = global_model(x)
                features2, _ = unpack_model_output(raw_out2)
                if features2.dim() > 2:
                    features2 = features2.view(features2.size(0), -1)
            
            posi = cos(features1, features2)
            logits = posi.reshape(-1,1)

            with torch.no_grad():
                raw_out3 = previous_net(x)
                features3, _ = unpack_model_output(raw_out3)
                if features3.dim() > 2:
                    features3 = features3.view(features3.size(0), -1)
                
            nega = cos(features1, features3)
            logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            logits /= args.temperature
            labels = torch.zeros(x.size(0)).long().to(device)
            
            loss2 = args.mu * criterion(logits, labels)
            loss1 = criterion(out, target)
            loss = loss1 + loss2
            
            if getattr(args, 'use_sd', False):
                loss += compute_sd_loss(raw_out1, target, device, args)

            if args.feddecorr and features1 is not None:
                loss_feddecorr = feddecorr(features1)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

    net.zero_grad()
    
    denom = max(1, len(train_dataloader) * getattr(args, "epochs", 1))
    avg_loss = total_loss / denom
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / denom
    
    return avg_loss, avg_correct_conf, avg_entropy

def fedrcl(net, train_dataloader, optimizer, device, args):
    total_loss = 0.
    net.train()
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    criterion_rcl = RCLloss()

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()
            features, out = net(x)
            CEloss = criterion(out, target)

            features_norm = F.normalize(features, p=2, dim=1)
            cosine_sim_matrix = torch.mm(features_norm, features_norm.t())
            
            rcl_loss = criterion_rcl(cosine_sim_matrix=cosine_sim_matrix, x=x, target=target, args=args)
            loss = CEloss + args.mu * rcl_loss
            
            total_loss += loss.item()

            if args.feddecorr:
                loss_feddecorr = feddecorr(features)
                loss = loss + args.feddecorr_coef * loss_feddecorr

            loss.backward()
            optimizer.step()

    net.zero_grad()
    return total_loss / max(1, len(train_dataloader)) / args.epochs

def adjust_lr(round, current_lr, args):
    if args.scheduler == 'linear':
        new_lr = args.eta_min + (args.lr - args.eta_min) * (1 - round / args.round)
    elif args.scheduler == 'cosine':
        new_lr = args.eta_min + (args.lr - args.eta_min) * 0.5 * (1 + math.cos(math.pi * round / args.round))
    elif args.scheduler == 'round':
        if (round + 1) % args.schedule_round == 0:
            new_lr = current_lr * args.lr_gamma
        else:
            new_lr = current_lr
    # elif args.scheduler == 'step':
    #     if (round + 1) in args.schedule_round:
    #         new_lr = current_lr * args.lr_gamma
    #     else:
    #         new_lr = current_lr
    else:
        new_lr = current_lr
    return new_lr

def fedsol(net, global_model, train_dataloader, optimizer, device, args):
    total_loss = 0.
    net.train()
    criterion = nn.CrossEntropyLoss()
    feddecorr = FedDecorrLoss()
    KLDiv = nn.KLDivLoss(reduction="batchmean")
    perturb_head = True
    perturb_body = True

    dg_model = copy.deepcopy(global_model)
    dg_model.to(device)
    for params in dg_model.parameters():
        params.requires_grad = False

    if args.adaptive:
        sam_optimizer = get_sam_optimizer_adaptive(net, dg_model, optimizer, args)
    else:
        sam_optimizer = get_sam_optimizer_fixed(net, optimizer, args)

    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            target = target.long()

            enable_running_stats(net)

            if not perturb_head:
                freeze_head(net)
            
            if not perturb_body:
                freeze_body(net)

            _, logits = net(x)
            _, dg_logits = dg_model(x)

            with torch.no_grad():
                dg_probs = torch.softmax(dg_logits / 3, dim=1)
            pred_probs = F.log_softmax(logits / 3, dim=1)

            loss = KLDiv(pred_probs, dg_probs)
            grads = torch.autograd.grad(
                loss,                         
                net.parameters(),             
                create_graph=True             
            )
            for p, g in zip(net.parameters(), grads):
                p.grad = g

            if not perturb_head:
                zerograd_head(net)

            if not perturb_body:
                zerograd_body(net)

            sam_optimizer.first_step(zero_grad=True)

            unfreeze(net)

            # second forward-backward pass
            disable_running_stats(net)
            _, logits_perturbed = net(x)
            sam_loss = criterion(logits_perturbed, target)
            sam_loss.backward()  # make sure to do a full forward pass
            sam_optimizer.second_step(zero_grad=True)
            
            total_loss += (loss.item() + sam_loss.item())

    net.zero_grad()
    return total_loss / max(1, len(train_dataloader)) / args.epochs

def fedacg(net, global_model, prev_global_model, train_dataloader, optimizer, device, args):
    total_loss = 0.0
    net.train()
    criterion = nn.CrossEntropyLoss()

    lookahead = copy.deepcopy(global_model)
    if prev_global_model is not None:
        for p_la, p_old in zip(lookahead.parameters(), prev_global_model.parameters()):
            p_la.data = p_la.data + args.lambda_acg * (p_la.data - p_old.data)

    for epoch in range(args.epochs):
        for x, target in train_dataloader:
            x, target = x.to(device), target.to(device).long()
            optimizer.zero_grad()

            features, out = net(x)
            loss = criterion(out, target)

            reg = 0.0
            for p, p_la in zip(net.parameters(), lookahead.parameters()):
                reg += torch.norm(p - p_la)**2
            loss = loss + args.beta_acg / 2 * reg

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
    net.zero_grad()
    return total_loss / max(1, len(train_dataloader)) / args.epochs

def fedbyot(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    temperature = args.temperature
    alpha = get_effective_byot_alpha(train_dataloader, args)
    alpha = estimate_client_byot_alpha(net, train_dataloader, device, args, alpha)
    alpha *= get_client_skew_scale(net, train_dataloader, device, args)
    class_alpha = estimate_class_byot_alpha(net, train_dataloader, device, args, alpha)
    branch_objective = getattr(args, "byot_branch_objective", "blend")
    branch_alphas = get_byot_branch_alphas(args, device)
    if branch_objective == "kd_only" and branch_alphas is not None:
        raise ValueError("--byot_branch_alphas is not supported with --byot_branch_objective kd_only.")
    active_branch_indices, branch_entropy_norm = get_byot_gated_active_branch_indices(train_dataloader, args)
    beta = args.byot_beta
    
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    criterion_mse = nn.MSELoss().to(device)
    
    total_loss = 0.0
    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    total_effective_alpha = 0.0
    total_alpha_batches = 0
    min_effective_alpha = None
    max_effective_alpha = None
    branch_freq_stats = init_train_branch_freq_stats() if getattr(args, "log_train_branch_frequency_stats", False) else None
    class_counts, _, expected_class_count = get_local_class_counts(
        train_dataloader, args, device
    ) if branch_freq_stats is not None else (None, None, None)
    net.train()
    
    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device).long()
            optimizer.zero_grad()

            out = net(x) 
            final_features = None
            
            if isinstance(out, tuple) and len(out) == 8:
                (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                final_features = final_fea
                
                # 1. Main Loss (Teacher)
                loss_main = criterion_ce(output, target)
                
                # 2. Student CE Loss
                ce_branch_losses = [
                    criterion_ce(m1, target),
                    criterion_ce(m2, target),
                    criterion_ce(m3, target),
                ]
                loss_ce_students = sum(ce_branch_losses[i] for i in active_branch_indices)
                loss_ce_students = reduce_active_branch_loss(
                    loss_ce_students, active_branch_indices, args
                )
                
                # 3. KL Divergence
                with torch.no_grad():
                    teacher_prob = F.softmax(output / temperature, dim=1)
                    
                    entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                    total_entropy += entropy
                    
                    teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                    correct_mask = (teacher_pred == target)
                    if correct_mask.any():
                        total_correct_conf += teacher_conf[correct_mask].mean().item()
                        valid_conf_batches += 1

                if branch_freq_stats is not None:
                    update_train_branch_freq_stats(
                        branch_freq_stats,
                        [m1, m2, m3],
                        teacher_prob,
                        target,
                        class_counts,
                        expected_class_count,
                        args,
                    )
                
                # 4. Feature Imitation
                feat_branch_losses = [
                    criterion_mse(f1, final_fea.detach()),
                    criterion_mse(f2, final_fea.detach()),
                    criterion_mse(f3, final_fea.detach()),
                ]
                loss_feat_students = sum(feat_branch_losses[i] for i in active_branch_indices)
                loss_feat_students = reduce_active_branch_loss(
                    loss_feat_students, active_branch_indices, args
                )

                if branch_alphas is not None:
                    sample_alpha = None
                elif class_alpha is not None:
                    sample_alpha = class_alpha[target].to(device)
                else:
                    sample_alpha = get_sample_byot_alpha(alpha, teacher_prob, [m1, m2, m3], target, args)
                if not active_branch_indices:
                    alpha_mean = 0.0
                    alpha_min = 0.0
                    alpha_max = 0.0
                    loss = loss_main
                elif sample_alpha is None:
                    kd_branch_losses = [
                        criterion_kl(F.log_softmax(m1 / temperature, dim=1), teacher_prob) * (temperature ** 2),
                        criterion_kl(F.log_softmax(m2 / temperature, dim=1), teacher_prob) * (temperature ** 2),
                        criterion_kl(F.log_softmax(m3 / temperature, dim=1), teacher_prob) * (temperature ** 2),
                    ]
                    loss_kd_students = sum(kd_branch_losses[i] for i in active_branch_indices)
                    loss_kd_students = reduce_active_branch_loss(
                        loss_kd_students, active_branch_indices, args
                    )
                    if branch_objective == "kd_only":
                        batch_alpha = get_batch_byot_alpha(alpha, output, [m1, m2, m3], target, args)
                        alpha_mean = float(batch_alpha.detach().item() if torch.is_tensor(batch_alpha) else batch_alpha)
                        alpha_min = alpha_mean
                        alpha_max = alpha_mean
                        loss = loss_main + batch_alpha * loss_kd_students + beta * loss_feat_students
                    elif branch_alphas is not None:
                        active_alphas = branch_alphas[active_branch_indices]
                        alpha_mean = float(active_alphas.mean().detach().item())
                        alpha_min = float(active_alphas.min().detach().item())
                        alpha_max = float(active_alphas.max().detach().item())
                        loss_students = branch_weighted_byot_loss(
                            [ce_branch_losses[i] for i in active_branch_indices],
                            [kd_branch_losses[i] for i in active_branch_indices],
                            active_alphas,
                        )
                        loss_students = reduce_active_branch_loss(
                            loss_students, active_branch_indices, args
                        )
                        loss = loss_main + loss_students + beta * loss_feat_students
                    else:
                        batch_alpha = get_batch_byot_alpha(alpha, output, [m1, m2, m3], target, args)
                        alpha_mean = float(batch_alpha.detach().item() if torch.is_tensor(batch_alpha) else batch_alpha)
                        alpha_min = alpha_mean
                        alpha_max = alpha_mean
                        loss = loss_main + (1 - batch_alpha) * loss_ce_students + batch_alpha * loss_kd_students + beta * loss_feat_students
                else:
                    branch_logits = [m1, m2, m3]
                    loss_kd_students = sum(
                        weighted_byot_kd_loss(branch_logits[i], teacher_prob, sample_alpha, temperature)
                        for i in active_branch_indices
                    ) * (temperature ** 2)
                    loss_kd_students = reduce_active_branch_loss(
                        loss_kd_students, active_branch_indices, args
                    )
                    mean_alpha = sample_alpha.mean()
                    alpha_mean = float(mean_alpha.detach().item())
                    alpha_min = float(sample_alpha.min().detach().item())
                    alpha_max = float(sample_alpha.max().detach().item())
                    if branch_objective == "kd_only":
                        loss = loss_main + loss_kd_students + beta * loss_feat_students
                    else:
                        loss = loss_main + (1 - mean_alpha) * loss_ce_students + loss_kd_students + beta * loss_feat_students

                total_effective_alpha += alpha_mean
                total_alpha_batches += 1
                min_effective_alpha = alpha_min if min_effective_alpha is None else min(min_effective_alpha, alpha_min)
                max_effective_alpha = alpha_max if max_effective_alpha is None else max(max_effective_alpha, alpha_max)

            else:
                if isinstance(out, tuple): _, output = out
                else: output = out
                loss = criterion_ce(output, target)
            
            loss += compute_fl_regularization(net, global_model, prev_net, x, final_features, device, args)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / max(1, len(train_dataloader))
    avg_effective_alpha = total_effective_alpha / max(1, total_alpha_batches)
    if min_effective_alpha is None:
        min_effective_alpha = alpha
    if max_effective_alpha is None:
        max_effective_alpha = alpha
    result = (
        total_loss / denom, 1.0, 0.0, 1.0, avg_correct_conf, 0, 0.0, avg_entropy,
        avg_effective_alpha, min_effective_alpha, max_effective_alpha,
    )
    if branch_freq_stats is not None:
        return result + (finalize_train_branch_freq_stats(branch_freq_stats),)
    return result

def fedbyot_lc(net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    tau = float(getattr(args, "calibration_temp", 1.0))
    
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    criterion_mse = nn.MSELoss().to(device)
    
    # --- Margin ---
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        num_classes = subset_dataset.dataset.num_classes
        subset_targets = np.array(subset_dataset.dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(subset_dataset.dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
    
    margin = tau * (class_counts ** -0.25).unsqueeze(dim=0).to(device)
    margin[margin == float('inf')] = 0 
    
    total_loss, total_correct_conf, total_entropy = 0.0, 0.0, 0.0
    valid_conf_batches = 0
    net.train()
    
    for epoch in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device), target.to(device).long()
            optimizer.zero_grad()
            out = net(x) 
            
            if isinstance(out, tuple) and len(out) == 8:
                (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                loss_main = criterion_ce(output, target)
                
                cal_output = output - margin
                cal_m1, cal_m2, cal_m3 = m1 - margin, m2 - margin, m3 - margin
                loss_ce_students = criterion_ce(cal_m1, target) + criterion_ce(cal_m2, target) + criterion_ce(cal_m3, target)
                
                with torch.no_grad():
                    teacher_prob = F.softmax(cal_output / temperature, dim=1)
                    entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                    total_entropy += entropy
                    teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                    correct_mask = (teacher_pred == target)
                    if correct_mask.any():
                        total_correct_conf += teacher_conf[correct_mask].mean().item()
                        valid_conf_batches += 1
                
                loss_kd_students = (
                    criterion_kl(F.log_softmax(cal_m1 / temperature, dim=1), teacher_prob) +
                    criterion_kl(F.log_softmax(cal_m2 / temperature, dim=1), teacher_prob) +
                    criterion_kl(F.log_softmax(cal_m3 / temperature, dim=1), teacher_prob)
                ) * (temperature ** 2)
                
                loss_feat_students = criterion_mse(f1, final_fea.detach()) + criterion_mse(f2, final_fea.detach()) + criterion_mse(f3, final_fea.detach())

                loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
            else:
                output = out[-1] if isinstance(out, tuple) else out
                loss = criterion_ce(output - margin, target)
            
            total_loss += float(loss.item())
            loss.backward()
            optimizer.step()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    return total_loss / denom, 1.0, 0.0, 1.0, total_correct_conf / max(1, valid_conf_batches), 0, 0.0, total_entropy / max(1, len(train_dataloader))


def fedbyot_rs(net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    rs_alpha = getattr(args, 'fedrs_alpha', 0.5)
    
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    criterion_mse = nn.MSELoss().to(device)
    
    # --- Scaling Factor ---
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        num_classes = subset_dataset.dataset.num_classes
        subset_targets = np.array(subset_dataset.dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(subset_dataset.dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
        
    missing_mask = (class_counts == 0).unsqueeze(0).float()
    scaling_factor = 1.0 - missing_mask * (1.0 - rs_alpha)
    
    total_loss, total_correct_conf, total_entropy = 0.0, 0.0, 0.0
    valid_conf_batches = 0
    net.train()
    
    for epoch in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device), target.to(device).long()
            optimizer.zero_grad()
            out = net(x) 
            
            if isinstance(out, tuple) and len(out) == 8:
                (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                loss_main = criterion_ce(output, target)
                
                restricted_output = output * scaling_factor
                res_m1, res_m2, res_m3 = m1 * scaling_factor, m2 * scaling_factor, m3 * scaling_factor
                loss_ce_students = criterion_ce(res_m1, target) + criterion_ce(res_m2, target) + criterion_ce(res_m3, target)
                
                with torch.no_grad():
                    teacher_prob = F.softmax(restricted_output / temperature, dim=1)
                    entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                    total_entropy += entropy
                    teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                    correct_mask = (teacher_pred == target)
                    if correct_mask.any():
                        total_correct_conf += teacher_conf[correct_mask].mean().item()
                        valid_conf_batches += 1
                
                loss_kd_students = (
                    criterion_kl(F.log_softmax(res_m1 / temperature, dim=1), teacher_prob) +
                    criterion_kl(F.log_softmax(res_m2 / temperature, dim=1), teacher_prob) +
                    criterion_kl(F.log_softmax(res_m3 / temperature, dim=1), teacher_prob)
                ) * (temperature ** 2)
                
                loss_feat_students = criterion_mse(f1, final_fea.detach()) + criterion_mse(f2, final_fea.detach()) + criterion_mse(f3, final_fea.detach())

                loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
            else:
                output = out[-1] if isinstance(out, tuple) else out
                loss = criterion_ce(output, target)
            
            total_loss += float(loss.item())
            loss.backward()
            optimizer.step()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    return total_loss / denom, 1.0, 0.0, 1.0, total_correct_conf / max(1, valid_conf_batches), 0, 0.0, total_entropy / max(1, len(train_dataloader))

def fedbyot_selective(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    threshold = float(getattr(args, "kd_conf_threshold", 0.8)) # Selective 핵심

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    total_keep_ratio = 0.0
    total_rfd = 0.0        
    total_feat_ratio = 0.0 
    total_correct_conf = 0.0 
    valid_conf_batches = 0
    total_entropy = 0.0
    class_kd_counts = torch.zeros(100, device=device)

    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                final_features = None # 헬퍼 함수에 전달할 변수 초기화
                
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    final_features = final_fea # 특징 벡터 추출
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)

                    loss_main = ce_loss(output, target)
                    loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(output / temperature, dim=1)
                        
                        # 엔트로피 계산
                        entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                        total_entropy += entropy
                        
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                        
                        # 확신도 계산
                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                        # [Selective 핵심] 마스크 생성
                        mask = (teacher_conf >= threshold)
                        if mask.any():
                            passed_targets = target[mask]
                            class_kd_counts += torch.bincount(passed_targets, minlength=100)
                    
                    total_keep_ratio += mask.float().mean().item()

                    def masked_kd(student_logits):
                        if not mask.any():
                            return torch.zeros((), device=device)
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        kl_per = F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)
                        return kl_per[mask].sum() / student_logits.size(0)

                    loss_kd_students = (masked_kd(m1) + masked_kd(m2) + masked_kd(m3)) * (temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    with torch.no_grad():
                        total_feat_ratio += 1.0 
                        rejected_mask = ~mask
                        if rejected_mask.any():
                            total_rfd += F.mse_loss(f1[rejected_mask].detach(), final_fea[rejected_mask].detach()).item()

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                else:
                    if isinstance(out, tuple):
                        final_features = out[4] if len(out) > 4 else out[0]
                        output = out[-1]
                    else:
                        output = out
                        
                    loss = ce_loss(output, target)
                    total_keep_ratio += 1.0
                    total_feat_ratio += 1.0

                # [수정] 헬퍼 함수 호출을 통해 FL 규제항(FedProx, MOON 등) 통합 추가
                loss += compute_fl_regularization(net, global_model, prev_net, x, final_features, device, args)

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / max(1, len(train_dataloader))
    zero_kd_classes = (class_kd_counts == 0).sum().item()
    kd_std = class_kd_counts.std().item()
    
    return total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom, avg_correct_conf, zero_kd_classes, kd_std, avg_entropy

def fedbyot_selective_ce_fallback(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    threshold = float(getattr(args, "kd_conf_threshold", 0.8))
    proxy = getattr(args, "byot_sample_proxy", "none")
    active_branch_indices = get_byot_active_branch_indices(args)

    ce_loss = nn.CrossEntropyLoss(reduction="none").to(device)
    main_ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    total_keep_ratio = 0.0
    total_rfd = 0.0
    total_feat_ratio = 0.0
    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    total_effective_alpha = 0.0
    total_alpha_batches = 0
    min_effective_alpha = None
    max_effective_alpha = None
    class_kd_counts = torch.zeros(int(getattr(args, "num_classes", 100)), device=device)

    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            out = net(x)
            final_features = None

            if isinstance(out, tuple) and len(out) == 8:
                output, m1, m2, m3, final_fea, f1, f2, f3 = out
                final_features = final_fea
                branch_logits = [m1, m2, m3]
                branch_features = [f1, f2, f3]
                active_branch_logits = [branch_logits[i] for i in active_branch_indices]

                loss_main = main_ce_loss(output, target)

                with torch.no_grad():
                    teacher_prob = F.softmax(output / temperature, dim=1)
                    entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                    total_entropy += entropy

                    teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                    correct_mask = teacher_pred == target
                    if correct_mask.any():
                        total_correct_conf += teacher_conf[correct_mask].mean().item()
                        valid_conf_batches += 1

                if branch_freq_stats is not None:
                    update_train_branch_freq_stats(
                        branch_freq_stats,
                        [m1, m2, m3],
                        teacher_prob,
                        target,
                        class_counts,
                        expected_class_count,
                        args,
                    )

                if proxy == "none":
                    reliability = (teacher_conf >= threshold).float()
                else:
                    reliability = get_sample_byot_alpha(1.0, teacher_prob, active_branch_logits, target, args)
                    if reliability is None:
                        reliability = (teacher_conf >= threshold).float()

                sample_alpha = (alpha * reliability).clamp(0.0, 1.0)
                hard_keep = sample_alpha > 0.5 * max(alpha, 1e-8)
                if hard_keep.any():
                    class_kd_counts += torch.bincount(target[hard_keep], minlength=class_kd_counts.numel())

                def mixed_branch_loss(student_logits):
                    ce_per = ce_loss(student_logits, target)
                    logp = F.log_softmax(student_logits / temperature, dim=1)
                    kd_per = F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1) * (temperature ** 2)
                    return ((1.0 - sample_alpha) * ce_per + sample_alpha * kd_per).mean()

                loss_students = sum(mixed_branch_loss(branch_logits[i]) for i in active_branch_indices)
                loss_feat_students = sum(feat_loss(branch_features[i], final_fea.detach()) for i in active_branch_indices)
                loss = loss_main + loss_students + beta * loss_feat_students

                alpha_mean = float(sample_alpha.mean().detach().item())
                alpha_min = float(sample_alpha.min().detach().item())
                alpha_max = float(sample_alpha.max().detach().item())
                total_effective_alpha += alpha_mean
                total_alpha_batches += 1
                min_effective_alpha = alpha_min if min_effective_alpha is None else min(min_effective_alpha, alpha_min)
                max_effective_alpha = alpha_max if max_effective_alpha is None else max(max_effective_alpha, alpha_max)
                total_keep_ratio += reliability.mean().detach().item()
                total_feat_ratio += 1.0

                with torch.no_grad():
                    rejected_mask = sample_alpha <= 0.5 * max(alpha, 1e-8)
                    if rejected_mask.any():
                        total_rfd += F.mse_loss(f1[rejected_mask].detach(), final_fea[rejected_mask].detach()).item()
            else:
                if isinstance(out, tuple):
                    final_features = out[4] if len(out) > 4 else out[0]
                    output = out[-1]
                else:
                    output = out
                loss = main_ce_loss(output, target)
                total_keep_ratio += 1.0
                total_feat_ratio += 1.0

            loss += compute_fl_regularization(net, global_model, prev_net, x, final_features, device, args)

            total_loss += float(loss.item())
            loss.backward()
            optimizer.step()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / max(1, len(train_dataloader))
    zero_kd_classes = (class_kd_counts == 0).sum().item()
    kd_std = class_kd_counts.std().item()
    avg_effective_alpha = total_effective_alpha / max(1, total_alpha_batches)
    if min_effective_alpha is None:
        min_effective_alpha = alpha
    if max_effective_alpha is None:
        max_effective_alpha = alpha

    return (
        total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom,
        avg_correct_conf, zero_kd_classes, kd_std, avg_entropy,
        avg_effective_alpha, min_effective_alpha, max_effective_alpha,
    )

def fedbyot_lc_selective(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    threshold = float(getattr(args, "kd_conf_threshold", 0.8)) # 고정 임계값
    tau = float(getattr(args, "calibration_temp", 1.0))

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    # --- [FedLC] Margin 생성 ---
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        num_classes = subset_dataset.dataset.num_classes
        subset_targets = np.array(subset_dataset.dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(subset_dataset.dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
    margin = tau * (class_counts ** -0.25).unsqueeze(dim=0).to(device)
    margin[margin == float('inf')] = 0 
    # ---------------------------

    total_loss, total_keep_ratio, total_rfd, total_feat_ratio = 0.0, 0.0, 0.0, 0.0
    total_correct_conf, valid_conf_batches, total_entropy = 0.0, 0, 0.0
    class_kd_counts = torch.zeros(100, device=device)
    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    loss_main = ce_loss(output, target) # 원본
                    
                    cal_output = output - margin
                    cal_m1, cal_m2, cal_m3 = m1 - margin, m2 - margin, m3 - margin
                    loss_ce_students = ce_loss(cal_m1, target) + ce_loss(cal_m2, target) + ce_loss(cal_m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(cal_output / temperature, dim=1)
                        entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                        total_entropy += entropy
                        
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                        mask = (teacher_conf >= threshold)
                        if mask.any():
                            class_kd_counts += torch.bincount(target[mask], minlength=100)
                    
                    total_keep_ratio += mask.float().mean().item()

                    def masked_kd(student_logits):
                        if not mask.any(): return torch.zeros((), device=device)
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].sum() / student_logits.size(0)

                    loss_kd_students = (masked_kd(cal_m1) + masked_kd(cal_m2) + masked_kd(cal_m3)) * (temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    with torch.no_grad():
                        total_feat_ratio += 1.0 
                        if (~mask).any(): total_rfd += F.mse_loss(f1[~mask].detach(), final_fea[~mask].detach()).item()

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                else:
                    output = out[-1] if isinstance(out, tuple) else out
                    loss = ce_loss(output - margin, target)
                    total_keep_ratio += 1.0
                    total_feat_ratio += 1.0

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    return total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom, total_correct_conf / max(1, valid_conf_batches), (class_kd_counts == 0).sum().item(), class_kd_counts.std().item(), total_entropy / max(1, len(train_dataloader))

def fedbyot_rs_selective(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    threshold = float(getattr(args, "kd_conf_threshold", 0.8)) # 고정 임계값
    rs_alpha = getattr(args, 'fedrs_alpha', 0.5)

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    # --- [FedRS] Scaling Factor 생성 ---
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        num_classes = subset_dataset.dataset.num_classes
        subset_targets = np.array(subset_dataset.dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(subset_dataset.dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
    missing_mask = (class_counts == 0).unsqueeze(0).float()
    scaling_factor = 1.0 - missing_mask * (1.0 - rs_alpha)
    # -----------------------------------

    total_loss, total_keep_ratio, total_rfd, total_feat_ratio = 0.0, 0.0, 0.0, 0.0
    total_correct_conf, valid_conf_batches, total_entropy = 0.0, 0, 0.0
    class_kd_counts = torch.zeros(100, device=device)
    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    loss_main = ce_loss(output, target)
                    
                    res_output = output * scaling_factor
                    res_m1, res_m2, res_m3 = m1 * scaling_factor, m2 * scaling_factor, m3 * scaling_factor
                    loss_ce_students = ce_loss(res_m1, target) + ce_loss(res_m2, target) + ce_loss(res_m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(res_output / temperature, dim=1)
                        entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                        total_entropy += entropy
                        
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                        mask = (teacher_conf >= threshold)
                        if mask.any():
                            class_kd_counts += torch.bincount(target[mask], minlength=100)
                    
                    total_keep_ratio += mask.float().mean().item()

                    def masked_kd(student_logits):
                        if not mask.any(): return torch.zeros((), device=device)
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].sum() / student_logits.size(0)

                    loss_kd_students = (masked_kd(res_m1) + masked_kd(res_m2) + masked_kd(res_m3)) * (temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    with torch.no_grad():
                        total_feat_ratio += 1.0 
                        if (~mask).any(): total_rfd += F.mse_loss(f1[~mask].detach(), final_fea[~mask].detach()).item()

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                else:
                    output = out[-1] if isinstance(out, tuple) else out
                    loss = ce_loss(output, target)
                    total_keep_ratio += 1.0
                    total_feat_ratio += 1.0

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    return total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom, total_correct_conf / max(1, valid_conf_batches), (class_kd_counts == 0).sum().item(), class_kd_counts.std().item(), total_entropy / max(1, len(train_dataloader))

def fedbyot_selective_greedy(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))

    # [1] 동적 임계값 (Dynamic Thresholding) 설정
    max_threshold = float(getattr(args, "kd_conf_threshold", 0.8))
    min_threshold = float(getattr(args, "min_threshold", 0.3))
    total_rounds = float(getattr(args, "round", 500))
    current_round = float(getattr(args, "current_round", 0))
    
    progress = current_round / max(1.0, total_rounds - 1.0)
    threshold = min_threshold + (max_threshold - min_threshold) * progress

    # [2] 워밍업 에폭 (Warm-up Epochs) 설정
    epochs = getattr(args, "epochs", 5)
    warmup_epochs = int(getattr(args, "warmup_epochs", 2))
    warmup_epochs = min(warmup_epochs, epochs - 1) 
    if warmup_epochs < 1: warmup_epochs = 1

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    total_original_samples = 0
    total_processed_samples = 0
    
    cached_x = []
    cached_t = []
    new_batches = []

    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    class_kd_counts = torch.zeros(100, device=device)

    net.train()
    bsz = getattr(train_dataloader, 'batch_size', 64)

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for x, target in train_dataloader:
                batch_size = x.size(0)
                total_original_samples += batch_size
                total_processed_samples += batch_size
                
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x)
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        loss_main = ce_loss(output, target)
                        loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                        with torch.no_grad():
                            teacher_prob = F.softmax(output / temperature, dim=1)
                            
                            entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                            total_entropy += entropy
                            
                            teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                            correct_mask = (teacher_pred == target)
                            if correct_mask.any():
                                total_correct_conf += teacher_conf[correct_mask].mean().item()
                                valid_conf_batches += 1

                            mask = (teacher_conf >= threshold)
                            if mask.any():
                                passed_targets = target[mask]
                                class_kd_counts += torch.bincount(passed_targets, minlength=100)

                        # 워밍업의 마지막 에폭일 때만 캐싱 진행
                        if epoch == (warmup_epochs - 1) and mask.any() and epochs > warmup_epochs:
                            cached_x.append(x[mask].detach().cpu())
                            cached_t.append(target[mask].detach().cpu())

                        def masked_kd(student_logits):
                            if not mask.any(): return torch.zeros((), device=device)
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            # 통과한 개수가 아닌 '전체 배치 사이즈'로 나누어 기울기 폭발 방지
                            return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].sum() / student_logits.size(0)

                        loss_kd_students = (masked_kd(m1) + masked_kd(m2) + masked_kd(m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                        loss += compute_fl_regularization(net, global_model, prev_net, x, final_fea, device, args, target=target)
                    else:
                        output = out[-1] if isinstance(out, tuple) else out
                        loss = ce_loss(output, target)
                        loss += compute_fl_regularization(net, global_model, prev_net, x, None, device, args, target=target)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            # 워밍업 마지막 에폭 종료 후 Re-batching 수행 및 BN 동결
            if epoch == (warmup_epochs - 1) and epochs > warmup_epochs:
                if len(cached_x) > 0:
                    cat_x = torch.cat(cached_x, dim=0)
                    cat_t = torch.cat(cached_t, dim=0)
                    
                    if cat_x.size(0) >= bsz:
                        indices = torch.randperm(cat_x.size(0))
                        cat_x = cat_x[indices]
                        cat_t = cat_t[indices]
                        
                        for i in range(0, cat_x.size(0), bsz):
                            bx = cat_x[i:i+bsz]
                            bt = cat_t[i:i+bsz]
                            if bx.size(0) >= 16: 
                                new_batches.append((bx, bt))

        else:
            # 남은 에폭들은 캐시된 데이터만 사용하여 학습
            if not new_batches:
                break 
                
            for x_sel, target_sel in new_batches:
                batch_size = x_sel.size(0)
                total_original_samples += bsz
                total_processed_samples += batch_size
                
                x_sel, target_sel = x_sel.to(device, non_blocking=True), target_sel.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x_sel)
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        loss_main = ce_loss(output, target_sel)
                        loss_ce_students = ce_loss(m1, target_sel) + ce_loss(m2, target_sel) + ce_loss(m3, target_sel)

                        with torch.no_grad():
                            teacher_prob = F.softmax(output / temperature, dim=1)

                        def full_kd(student_logits):
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            return F.kl_div(logp, teacher_prob, reduction="batchmean")

                        loss_kd_students = (full_kd(m1) + full_kd(m2) + full_kd(m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                        loss += compute_fl_regularization(net, global_model, prev_net, x_sel, final_fea, device, args, target=target_sel)
                    else:
                        output = out[-1] if isinstance(out, tuple) else out
                        loss = ce_loss(output, target_sel)
                        loss += compute_fl_regularization(net, global_model, prev_net, x_sel, None, device, args, target=target_sel)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    net.train()

    end_time = time.time()
    wall_clock_time = end_time - start_time
    compute_efficiency = total_processed_samples / max(1, total_original_samples)

    denom = max(1, num_batches)
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / denom
    zero_kd_classes = (class_kd_counts == 0).sum().item()
    kd_std = class_kd_counts.std().item()

    return total_loss / denom, wall_clock_time, compute_efficiency, avg_correct_conf, zero_kd_classes, kd_std, avg_entropy

def fedbyot_lc_greedy(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    tau = float(getattr(args, "calibration_temp", 1.0)) # FedLC 강도

    max_threshold = float(getattr(args, "kd_conf_threshold", 0.8))
    min_threshold = float(getattr(args, "min_threshold", 0.3))
    total_rounds = float(getattr(args, "round", 500))
    current_round = float(getattr(args, "current_round", 0))
    progress = current_round / max(1.0, total_rounds - 1.0)
    threshold = min_threshold + (max_threshold - min_threshold) * progress

    epochs = getattr(args, "epochs", 5)
    warmup_epochs = max(1, min(int(getattr(args, "warmup_epochs", 2)), epochs - 1))

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    # --- [FedLC] 데이터 분포 계산 및 Margin 생성 ---
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        original_dataset = subset_dataset.dataset
        num_classes = original_dataset.num_classes
        subset_targets = np.array(original_dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(original_dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
    
    margin = tau * (class_counts ** -0.25)
    margin = margin.unsqueeze(dim=0).to(device)
    margin[margin == float('inf')] = 0 
    # ----------------------------------------------

    total_loss, total_correct_conf, valid_conf_batches, total_entropy = 0.0, 0.0, 0, 0.0
    num_batches, total_original_samples, total_processed_samples = 0, 0, 0
    cached_x, cached_t, new_batches = [], [], []
    class_kd_counts = torch.zeros(100, device=device)

    start_time = time.time()
    net.train()
    bsz = getattr(train_dataloader, 'batch_size', 64)

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for x, target in train_dataloader:
                batch_size = x.size(0)
                total_original_samples += batch_size
                total_processed_samples += batch_size
                
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x)
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        loss_main = ce_loss(output, target) # 원본 사용
                        
                        # Student는 보정된 값 사용
                        calibrated_output = output - margin
                        cal_m1, cal_m2, cal_m3 = m1 - margin, m2 - margin, m3 - margin
                        loss_ce_students = ce_loss(cal_m1, target) + ce_loss(cal_m2, target) + ce_loss(cal_m3, target)

                        with torch.no_grad():
                            teacher_prob = F.softmax(calibrated_output / temperature, dim=1) # 보정된 지식 전달
                            entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                            total_entropy += entropy
                            
                            teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                            correct_mask = (teacher_pred == target)
                            if correct_mask.any():
                                total_correct_conf += teacher_conf[correct_mask].mean().item()
                                valid_conf_batches += 1

                            mask = (teacher_conf >= threshold)
                            if mask.any():
                                passed_targets = target[mask]
                                class_kd_counts += torch.bincount(passed_targets, minlength=100)

                        if epoch == (warmup_epochs - 1) and mask.any() and epochs > warmup_epochs:
                            cached_x.append(x[mask].detach().cpu())
                            cached_t.append(target[mask].detach().cpu())

                        def masked_kd(student_logits):
                            if not mask.any(): return torch.zeros((), device=device)
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].sum() / student_logits.size(0)

                        loss_kd_students = (masked_kd(cal_m1) + masked_kd(cal_m2) + masked_kd(cal_m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                    else:
                        output = out[-1] if isinstance(out, tuple) else out
                        loss = ce_loss(output - margin, target)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            if epoch == (warmup_epochs - 1) and epochs > warmup_epochs and len(cached_x) > 0:
                cat_x, cat_t = torch.cat(cached_x, dim=0), torch.cat(cached_t, dim=0)
                if cat_x.size(0) >= bsz:
                    indices = torch.randperm(cat_x.size(0))
                    cat_x, cat_t = cat_x[indices], cat_t[indices]
                    for i in range(0, cat_x.size(0), bsz):
                        bx, bt = cat_x[i:i+bsz], cat_t[i:i+bsz]
                        if bx.size(0) >= 16: new_batches.append((bx, bt))
        else:
            if not new_batches: break 
            for x_sel, target_sel in new_batches:
                batch_size = x_sel.size(0)
                total_original_samples += bsz
                total_processed_samples += batch_size
                
                x_sel, target_sel = x_sel.to(device, non_blocking=True), target_sel.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x_sel)
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        loss_main = ce_loss(output, target_sel)
                        
                        calibrated_output = output - margin
                        cal_m1, cal_m2, cal_m3 = m1 - margin, m2 - margin, m3 - margin
                        loss_ce_students = ce_loss(cal_m1, target_sel) + ce_loss(cal_m2, target_sel) + ce_loss(cal_m3, target_sel)

                        with torch.no_grad():
                            teacher_prob = F.softmax(calibrated_output / temperature, dim=1)

                        def full_kd(student_logits):
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            return F.kl_div(logp, teacher_prob, reduction="batchmean")

                        loss_kd_students = (full_kd(cal_m1) + full_kd(cal_m2) + full_kd(cal_m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                    else:
                        output = out[-1] if isinstance(out, tuple) else out
                        loss = ce_loss(output - margin, target_sel)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    compute_efficiency = total_processed_samples / max(1, total_original_samples)
    denom = max(1, num_batches)
    return total_loss / denom, time.time() - start_time, compute_efficiency, total_correct_conf / max(1, valid_conf_batches), (class_kd_counts == 0).sum().item(), class_kd_counts.std().item(), total_entropy / denom

def fedbyot_rs_greedy(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    rs_alpha = getattr(args, 'fedrs_alpha', 0.5)

    max_threshold = float(getattr(args, "kd_conf_threshold", 0.8))
    min_threshold = float(getattr(args, "min_threshold", 0.3))
    total_rounds = float(getattr(args, "round", 500))
    current_round = float(getattr(args, "current_round", 0))
    progress = current_round / max(1.0, total_rounds - 1.0)
    threshold = min_threshold + (max_threshold - min_threshold) * progress

    epochs = getattr(args, "epochs", 5)
    warmup_epochs = max(1, min(int(getattr(args, "warmup_epochs", 2)), epochs - 1))

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)
    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    # --- [FedRS] Missing Class 마스크 및 Scaling Factor 생성 ---
    subset_dataset = train_dataloader.dataset
    if hasattr(subset_dataset, 'dataset'):
        original_dataset = subset_dataset.dataset
        num_classes = original_dataset.num_classes
        subset_targets = np.array(original_dataset.target)[subset_dataset.indices] if hasattr(subset_dataset, 'indices') else np.array(original_dataset.target)
    else:
        num_classes = 100 
        subset_targets = np.array(subset_dataset.target)

    class_counts = torch.zeros(num_classes).to(device)
    uniq_val, uniq_count = np.unique(subset_targets, return_counts=True)
    for i, c in enumerate(uniq_val.tolist()): class_counts[c] = uniq_count[i]
        
    missing_mask = (class_counts == 0).unsqueeze(0).float()
    scaling_factor = 1.0 - missing_mask * (1.0 - rs_alpha)
    # ------------------------------------------------------------

    total_loss, total_correct_conf, valid_conf_batches, total_entropy = 0.0, 0.0, 0, 0.0
    num_batches, total_original_samples, total_processed_samples = 0, 0, 0
    cached_x, cached_t, new_batches = [], [], []
    class_kd_counts = torch.zeros(100, device=device)

    start_time = time.time()
    net.train()
    bsz = getattr(train_dataloader, 'batch_size', 64)

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for x, target in train_dataloader:
                batch_size = x.size(0)
                total_original_samples += batch_size
                total_processed_samples += batch_size
                
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x)
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        loss_main = ce_loss(output, target) # Teacher는 자유롭게 학습
                        
                        restricted_output = output * scaling_factor
                        res_m1, res_m2, res_m3 = m1 * scaling_factor, m2 * scaling_factor, m3 * scaling_factor
                        loss_ce_students = ce_loss(res_m1, target) + ce_loss(res_m2, target) + ce_loss(res_m3, target)

                        with torch.no_grad():
                            teacher_prob = F.softmax(restricted_output / temperature, dim=1) # Missing 제어된 지식
                            entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                            total_entropy += entropy
                            
                            teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                            correct_mask = (teacher_pred == target)
                            if correct_mask.any():
                                total_correct_conf += teacher_conf[correct_mask].mean().item()
                                valid_conf_batches += 1

                            mask = (teacher_conf >= threshold)
                            if mask.any():
                                passed_targets = target[mask]
                                class_kd_counts += torch.bincount(passed_targets, minlength=100)

                        if epoch == (warmup_epochs - 1) and mask.any() and epochs > warmup_epochs:
                            cached_x.append(x[mask].detach().cpu())
                            cached_t.append(target[mask].detach().cpu())

                        def masked_kd(student_logits):
                            if not mask.any(): return torch.zeros((), device=device)
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].sum() / student_logits.size(0)

                        loss_kd_students = (masked_kd(res_m1) + masked_kd(res_m2) + masked_kd(res_m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                    else:
                        output = out[-1] if isinstance(out, tuple) else out
                        loss = ce_loss(output, target)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            if epoch == (warmup_epochs - 1) and epochs > warmup_epochs and len(cached_x) > 0:
                cat_x, cat_t = torch.cat(cached_x, dim=0), torch.cat(cached_t, dim=0)
                if cat_x.size(0) >= bsz:
                    indices = torch.randperm(cat_x.size(0))
                    cat_x, cat_t = cat_x[indices], cat_t[indices]
                    for i in range(0, cat_x.size(0), bsz):
                        bx, bt = cat_x[i:i+bsz], cat_t[i:i+bsz]
                        if bx.size(0) >= 16: new_batches.append((bx, bt))
        else:
            if not new_batches: break 
            for x_sel, target_sel in new_batches:
                batch_size = x_sel.size(0)
                total_original_samples += bsz
                total_processed_samples += batch_size
                
                x_sel, target_sel = x_sel.to(device, non_blocking=True), target_sel.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x_sel)
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        loss_main = ce_loss(output, target_sel)
                        
                        restricted_output = output * scaling_factor
                        res_m1, res_m2, res_m3 = m1 * scaling_factor, m2 * scaling_factor, m3 * scaling_factor
                        loss_ce_students = ce_loss(res_m1, target_sel) + ce_loss(res_m2, target_sel) + ce_loss(res_m3, target_sel)

                        with torch.no_grad():
                            teacher_prob = F.softmax(restricted_output / temperature, dim=1)

                        def full_kd(student_logits):
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            return F.kl_div(logp, teacher_prob, reduction="batchmean")

                        loss_kd_students = (full_kd(res_m1) + full_kd(res_m2) + full_kd(res_m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                    else:
                        output = out[-1] if isinstance(out, tuple) else out
                        loss = ce_loss(output, target_sel)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    compute_efficiency = total_processed_samples / max(1, total_original_samples)
    denom = max(1, num_batches)
    return total_loss / denom, time.time() - start_time, compute_efficiency, total_correct_conf / max(1, valid_conf_batches), (class_kd_counts == 0).sum().item(), class_kd_counts.std().item(), total_entropy / denom

def dataloader_batch_size_estimate(dataloader):
    # 추정용 헬퍼 함수
    return dataloader.batch_size if hasattr(dataloader, 'batch_size') else 64

'''
def fedbyot_logit_adj(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    tau = float(getattr(args, "tau", 1.0)) 

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    class_counts = torch.zeros(num_classes, device=device)
    for _, target in train_dataloader:
        class_counts += torch.bincount(target.to(device), minlength=num_classes)
    
    class_probs = class_counts / class_counts.sum()
    class_probs = torch.clamp(class_probs, min=1e-8) 
    log_prior = torch.log(class_probs)

    total_loss = 0.0
    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                final_features = None # 헬퍼 함수에 전달할 변수 초기화
                
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    final_features = final_fea # 특징 벡터 추출
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)
                    
                    adjusted_output = output - tau * log_prior.unsqueeze(0)

                    loss_main = ce_loss(output, target) 
                    loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(adjusted_output / temperature, dim=1)
                        
                        entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                        total_entropy += entropy
                        
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                    def kd_loss(student_logits):
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        return F.kl_div(logp, teacher_prob, reduction="batchmean")

                    loss_kd_students = (kd_loss(m1) + kd_loss(m2) + kd_loss(m3)) * (temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                else:
                    if isinstance(out, tuple):
                        final_features = out[4] if len(out) > 4 else out[0]
                        output = out[-1]
                    else:
                        output = out
                    loss = ce_loss(output, target)

                # [수정] 헬퍼 함수 호출을 통해 FL 규제항(FedProx, MOON 등) 통합 추가
                loss += compute_fl_regularization(net, global_model, prev_net, x, final_features, device, args)

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / max(1, len(train_dataloader))
    return total_loss / denom, 1.0, 0.0, 1.0, avg_correct_conf, 0, 0.0, avg_entropy

def fedbyot_dynamic_temp(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    base_temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    alpha_t = float(getattr(args, "alpha_t", 2.0)) # 온도 조절 민감도

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                final_features = None # 헬퍼 함수에 전달할 변수 초기화
                
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    final_features = final_fea # 특징 벡터 추출
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)

                    loss_main = ce_loss(output, target)
                    loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(output / base_temperature, dim=1)
                        
                        entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                        total_entropy += entropy
                        
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1
                        
                        # 배치 내 각 샘플의 엔트로피 계산: H = - sum(p * log(p))
                        entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1)
                        
                        # 엔트로피가 높을수록(불확실할수록) 온도를 높임
                        T_dyn = base_temperature * (1.0 + alpha_t * entropy).view(-1, 1)

                        # 동적 온도가 적용된 교사 확률 재계산
                        teacher_prob_dyn = F.softmax(output / T_dyn, dim=1)

                    def kd_loss_dynamic(student_logits):
                        logp = F.log_softmax(student_logits / T_dyn, dim=1)
                        # reduction="none"으로 샘플별 손실을 구한 뒤 평균
                        return F.kl_div(logp, teacher_prob_dyn, reduction="none").sum(dim=1).mean()

                    # Scale 맞추기 위해 base_temperature 제곱 사용
                    loss_kd_students = (kd_loss_dynamic(m1) + kd_loss_dynamic(m2) + kd_loss_dynamic(m3)) * (base_temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                    
                else:
                    if isinstance(out, tuple):
                        final_features = out[4] if len(out) > 4 else out[0]
                        output = out[-1]
                    else:
                        output = out
                    loss = ce_loss(output, target)

                # [수정] 헬퍼 함수 호출을 통해 FL 규제항(FedProx, MOON 등) 통합 추가
                loss += compute_fl_regularization(net, global_model, prev_net, x, final_features, device, args)

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / max(1, len(train_dataloader))
    return total_loss / denom, 1.0, 0.0, 1.0, avg_correct_conf, 0, 0.0, avg_entropy

def fedbyot_logit_adj_greedy(net, global_model, prev_net, train_dataloader, optimizer, device, args):
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    tau = float(getattr(args, "tau", 1.0)) 
    threshold = float(getattr(args, "kd_conf_threshold", 0.8))

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    num_classes = 10 if args.dataset == 'cifar10' else 100
    class_counts = torch.zeros(num_classes, device=device)
    for _, target in train_dataloader:
        class_counts += torch.bincount(target.to(device), minlength=num_classes)
    class_probs = class_counts / class_counts.sum()
    class_probs = torch.clamp(class_probs, min=1e-8) 
    log_prior = torch.log(class_probs)

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    total_original_samples = 0
    total_processed_samples = 0
    cached_batches = []
    
    # [수정] 7개 반환을 위한 지표 변수 추가
    total_correct_conf = 0.0
    valid_conf_batches = 0
    total_entropy = 0.0
    class_kd_counts = torch.zeros(100, device=device)

    net.train()
    epochs = getattr(args, "epochs", 1)

    for epoch in range(epochs):
        if epoch == 0:
            for x, target in train_dataloader:
                batch_size = x.size(0)
                total_original_samples += batch_size
                total_processed_samples += batch_size
                
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x)
                    final_features = None
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        final_features = final_fea
                        
                        if output.dim() > 2: output = output.view(output.size(0), -1)
                        if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                        if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                        if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)

                        adjusted_output = output - tau * log_prior.unsqueeze(0)

                        loss_main = ce_loss(output, target)
                        loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                        with torch.no_grad():
                            teacher_prob = F.softmax(adjusted_output / temperature, dim=1)
                            
                            # [수정] 지표 계산 로직 추가
                            entropy = -(teacher_prob * torch.log(teacher_prob + 1e-8)).sum(dim=1).mean().item()
                            total_entropy += entropy
                            
                            teacher_conf, teacher_pred = teacher_prob.max(dim=1)
                            
                            correct_mask = (teacher_pred == target)
                            if correct_mask.any():
                                total_correct_conf += teacher_conf[correct_mask].mean().item()
                                valid_conf_batches += 1

                            mask = (teacher_conf >= threshold)
                            
                            if mask.any():
                                passed_targets = target[mask]
                                class_kd_counts += torch.bincount(passed_targets, minlength=100)

                        if mask.any() and epochs > 1:
                            cached_batches.append((x[mask].cpu(), target[mask].cpu()))

                        def masked_kd(student_logits):
                            if not mask.any(): return torch.zeros((), device=device)
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].mean()

                        loss_kd_students = (masked_kd(m1) + masked_kd(m2) + masked_kd(m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                    else:
                        if isinstance(out, tuple):
                            final_features = out[4] if len(out) > 4 else out[0]
                            output = out[-1]
                        else: output = out
                        loss = ce_loss(output, target)

                    loss += compute_fl_regularization(net, global_model, prev_net, x, final_features, device, args)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            # Epoch 1+ 에서는 연산 속도를 위해 지표 계산은 생략하고 학습만 수행
            for x_sel, target_sel in cached_batches:
                batch_size = x_sel.size(0)
                total_original_samples += dataloader_batch_size_estimate(train_dataloader)
                total_processed_samples += batch_size
                
                x_sel, target_sel = x_sel.to(device, non_blocking=True), target_sel.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = net(x_sel)
                    final_features = None
                    if isinstance(out, tuple) and len(out) == 8:
                        (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                        final_features = final_fea
                        
                        if output.dim() > 2: output = output.view(output.size(0), -1)
                        if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                        if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                        if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)

                        adjusted_output = output - tau * log_prior.unsqueeze(0)

                        loss_main = ce_loss(output, target_sel)
                        loss_ce_students = ce_loss(m1, target_sel) + ce_loss(m2, target_sel) + ce_loss(m3, target_sel)

                        with torch.no_grad():
                            teacher_prob = F.softmax(adjusted_output / temperature, dim=1)
                        
                        def full_kd(student_logits):
                            logp = F.log_softmax(student_logits / temperature, dim=1)
                            return F.kl_div(logp, teacher_prob, reduction="batchmean")

                        loss_kd_students = (full_kd(m1) + full_kd(m2) + full_kd(m3)) * (temperature ** 2)
                        loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                        loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                    else:
                        if isinstance(out, tuple):
                            final_features = out[4] if len(out) > 4 else out[0]
                            output = out[-1]
                        else: output = out
                        loss = ce_loss(output, target_sel)

                    loss += compute_fl_regularization(net, global_model, prev_net, x_sel, final_features, device, args)

                total_loss += float(loss.item())
                num_batches += 1
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    compute_efficiency = total_processed_samples / max(1, total_original_samples)
    wall_clock_time = time.time() - start_time
    denom = max(1, num_batches)
    
    # [수정] 7개 맞춰서 최종 반환
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    avg_entropy = total_entropy / max(1, len(train_dataloader))
    zero_kd_classes = (class_kd_counts == 0).sum().item()
    kd_std = class_kd_counts.std().item()
    
    return total_loss / denom, wall_clock_time, compute_efficiency, avg_correct_conf, zero_kd_classes, kd_std, avg_entropy


def fedbyot_focal(net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    gamma = float(getattr(args, "gamma", 2.0)) # Focal Loss의 감마 파라미터

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    # Focal Loss 정의 (내부 클래스로 간단히 구현)
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()

    criterion_main = FocalLoss(gamma=gamma).to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)

                    # CrossEntropy 대신 Focal Loss 적용
                    loss_main = criterion_main(output, target)
                    loss_ce_students = criterion_main(m1, target) + criterion_main(m2, target) + criterion_main(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(output / temperature, dim=1)

                    def kd_loss(student_logits):
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        return F.kl_div(logp, teacher_prob, reduction="batchmean")

                    loss_kd_students = (kd_loss(m1) + kd_loss(m2) + kd_loss(m3)) * (temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                else:
                    output = out[-1] if isinstance(out, tuple) else out
                    loss = criterion_main(output, target)

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    return total_loss / denom, 1.0, 0.0, 1.0, 0.0, 0, 0.0

def fedbyot_freematch(net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    total_keep_ratio = 0.0
    total_rfd = 0.0        
    total_feat_ratio = 0.0 
    total_correct_conf = 0.0 
    valid_conf_batches = 0
    
    class_kd_counts = torch.zeros(100, device=device)

    # [FreeMatch 초기화] 클라이언트별 클래스 확신도 EMA 상태 보존
    if not hasattr(net, 'class_conf_ema'):
        net.class_conf_ema = torch.ones(100, device=device) / 100.0
    
    ema_momentum = 0.9

    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)

                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)
                    
                    if final_fea.dim() > 2: final_fea = final_fea.view(final_fea.size(0), -1)
                    if f1.dim() > 2: f1 = f1.view(f1.size(0), -1)
                    if f2.dim() > 2: f2 = f2.view(f2.size(0), -1)
                    if f3.dim() > 2: f3 = f3.view(f3.size(0), -1)

                    loss_main = ce_loss(output, target)
                    loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(output / temperature, dim=1)
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)

                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                        # [FreeMatch Logic] EMA 업데이트 및 임계값 계산
                        unique_classes = teacher_pred.unique()
                        for c in unique_classes:
                            c_mask = (teacher_pred == c)
                            c_mean_conf = teacher_conf[c_mask].mean()
                            net.class_conf_ema[c] = ema_momentum * net.class_conf_ema[c] + (1 - ema_momentum) * c_mean_conf

                        global_conf_ema = net.class_conf_ema.mean()
                        max_conf_ema = net.class_conf_ema.max()
                        
                        dynamic_thresholds = global_conf_ema * (net.class_conf_ema / max_conf_ema)
                        sample_thresholds = dynamic_thresholds[teacher_pred]

                        mask = (teacher_conf >= sample_thresholds)

                        if mask.any():
                            passed_targets = target[mask]
                            class_kd_counts += torch.bincount(passed_targets, minlength=100)
                    
                    current_ratio = mask.float().mean().item()
                    total_keep_ratio += current_ratio

                    def masked_kd(student_logits):
                        if not mask.any():
                            return torch.zeros((), device=device)
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        kl_per = F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)
                        return kl_per[mask].mean()

                    loss_kd_students = (masked_kd(m1) + masked_kd(m2) + masked_kd(m3)) * (temperature ** 2)

                    loss_feat_students = (
                        feat_loss(f1, final_fea.detach()) +
                        feat_loss(f2, final_fea.detach()) +
                        feat_loss(f3, final_fea.detach())
                    )

                    with torch.no_grad():
                        total_feat_ratio += 1.0 
                        rejected_mask = ~mask
                        if rejected_mask.any():
                            s_rej = f1[rejected_mask].detach()
                            t_rej = final_fea[rejected_mask].detach()
                            total_rfd += F.mse_loss(s_rej, t_rej).item()

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students

                else:
                    if isinstance(out, tuple): output = out[-1]
                    else: output = out
                    loss = ce_loss(output, target)
                    total_keep_ratio += 1.0
                    total_feat_ratio += 1.0

            total_loss += float(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    
    zero_kd_classes = (class_kd_counts == 0).sum().item()
    kd_std = class_kd_counts.std().item()
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    
    return total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom, avg_correct_conf, zero_kd_classes, kd_std

def fedbyot_flexmatch(net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    
    # FlexMatch는 기준이 되는 기본 임계값이 필요합니다.
    base_threshold = float(getattr(args, "kd_conf_threshold", 0.8)) 

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    total_keep_ratio = 0.0
    total_rfd = 0.0        
    total_feat_ratio = 0.0 
    total_correct_conf = 0.0 
    valid_conf_batches = 0
    class_kd_counts = torch.zeros(100, device=device)

    # [FlexMatch 초기화] 클래스별 통과 '샘플 수(Count)' 상태 보존
    if not hasattr(net, 'class_pass_counts'):
        net.class_pass_counts = torch.ones(100, device=device) # 0으로 나누는 것 방지

    ema_momentum = 0.9

    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)

                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)
                    
                    if final_fea.dim() > 2: final_fea = final_fea.view(final_fea.size(0), -1)
                    if f1.dim() > 2: f1 = f1.view(f1.size(0), -1)
                    if f2.dim() > 2: f2 = f2.view(f2.size(0), -1)
                    if f3.dim() > 2: f3 = f3.view(f3.size(0), -1)

                    loss_main = ce_loss(output, target)
                    loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(output / temperature, dim=1)
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)

                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                        # [FlexMatch Logic 1] 기본 임계값을 넘는 샘플 수 카운트 추적
                        high_conf_mask = (teacher_conf >= base_threshold)
                        if high_conf_mask.any():
                            passed_classes = teacher_pred[high_conf_mask]
                            batch_counts = torch.bincount(passed_classes, minlength=100).float()
                            net.class_pass_counts = ema_momentum * net.class_pass_counts + (1 - ema_momentum) * batch_counts

                        # [FlexMatch Logic 2] 통과 횟수 비율에 맞춘 동적 임계값 계산
                        max_count = net.class_pass_counts.max()
                        beta_c = net.class_pass_counts / (max_count + 1e-8)
                        
                        # 가장 많이 통과한 클래스는 base_threshold(0.8)를 유지, 적게 통과한 클래스는 낮아짐
                        dynamic_thresholds = base_threshold * beta_c
                        sample_thresholds = dynamic_thresholds[teacher_pred]

                        # [FlexMatch Logic 3] 최종 마스크 생성
                        mask = (teacher_conf >= sample_thresholds)

                        if mask.any():
                            passed_targets = target[mask]
                            class_kd_counts += torch.bincount(passed_targets, minlength=100)
                    
                    current_ratio = mask.float().mean().item()
                    total_keep_ratio += current_ratio

                    def masked_kd(student_logits):
                        if not mask.any():
                            return torch.zeros((), device=device)
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        kl_per = F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)
                        return kl_per[mask].mean()

                    loss_kd_students = (masked_kd(m1) + masked_kd(m2) + masked_kd(m3)) * (temperature ** 2)

                    loss_feat_students = (
                        feat_loss(f1, final_fea.detach()) +
                        feat_loss(f2, final_fea.detach()) +
                        feat_loss(f3, final_fea.detach())
                    )

                    with torch.no_grad():
                        total_feat_ratio += 1.0 
                        rejected_mask = ~mask
                        if rejected_mask.any():
                            s_rej = f1[rejected_mask].detach()
                            t_rej = final_fea[rejected_mask].detach()
                            total_rfd += F.mse_loss(s_rej, t_rej).item()

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students

                else:
                    if isinstance(out, tuple): output = out[-1]
                    else: output = out
                    loss = ce_loss(output, target)
                    total_keep_ratio += 1.0
                    total_feat_ratio += 1.0

            total_loss += float(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    
    zero_kd_classes = (class_kd_counts == 0).sum().item()
    kd_std = class_kd_counts.std().item()
    avg_correct_conf = total_correct_conf / max(1, valid_conf_batches)
    
    return total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom, avg_correct_conf, zero_kd_classes, kd_std

def fedbyot_percentile(net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))
    keep_ratio = 0.5 # 상위 50%만 수용

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss, total_keep_ratio, total_rfd, total_feat_ratio, total_correct_conf = 0.0, 0.0, 0.0, 0.0, 0.0
    valid_conf_batches = 0
    class_kd_counts = torch.zeros(100, device=device)

    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)
                    if final_fea.dim() > 2: final_fea = final_fea.view(final_fea.size(0), -1)
                    if f1.dim() > 2: f1 = f1.view(f1.size(0), -1)
                    if f2.dim() > 2: f2 = f2.view(f2.size(0), -1)
                    if f3.dim() > 2: f3 = f3.view(f3.size(0), -1)

                    loss_main = ce_loss(output, target)
                    loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(output / temperature, dim=1)
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)

                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                        # [Percentile Logic] 배치 내 상위 keep_ratio(50%) 기준값 계산
                        k = max(1, int(teacher_conf.size(0) * keep_ratio))
                        threshold_val = torch.topk(teacher_conf, k)[0][-1]
                        mask = (teacher_conf >= threshold_val)

                        if mask.any():
                            passed_targets = target[mask]
                            class_kd_counts += torch.bincount(passed_targets, minlength=100)
                    
                    total_keep_ratio += mask.float().mean().item()

                    def masked_kd(student_logits):
                        if not mask.any(): return torch.zeros((), device=device)
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].mean()

                    loss_kd_students = (masked_kd(m1) + masked_kd(m2) + masked_kd(m3)) * (temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    with torch.no_grad():
                        total_feat_ratio += 1.0 
                        rejected_mask = ~mask
                        if rejected_mask.any():
                            total_rfd += F.mse_loss(f1[rejected_mask].detach(), final_fea[rejected_mask].detach()).item()

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                else:
                    output = out[-1] if isinstance(out, tuple) else out
                    loss = ce_loss(output, target)
                    total_keep_ratio += 1.0
                    total_feat_ratio += 1.0

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    return total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom, total_correct_conf / max(1, valid_conf_batches), (class_kd_counts == 0).sum().item(), class_kd_counts.std().item()

def fedbyot_round_aware(net, train_dataloader, optimizer, device, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    from torch.cuda.amp import autocast, GradScaler

    temperature = float(getattr(args, "temperature", 0.5))
    alpha = float(getattr(args, "byot_alpha", 0.15))
    beta = float(getattr(args, "byot_beta", 0.05))

    # [Round-aware Logic] 코사인 어닐링으로 현재 라운드의 임계값 계산 (0.5 -> 0.95)
    current_round = float(getattr(args, "current_round", 1))
    total_rounds = float(getattr(args, "round", 500))
    progress = min(1.0, current_round / total_rounds)
    dynamic_threshold = 0.5 + 0.45 * (1 - math.cos(math.pi * progress)) / 2

    use_amp = bool(getattr(args, "amp", False))
    scaler = GradScaler(enabled=use_amp)

    ce_loss = nn.CrossEntropyLoss().to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss, total_keep_ratio, total_rfd, total_feat_ratio, total_correct_conf = 0.0, 0.0, 0.0, 0.0, 0.0
    valid_conf_batches = 0
    class_kd_counts = torch.zeros(100, device=device)

    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                out = net(x)
                if isinstance(out, tuple) and len(out) == 8:
                    (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                    
                    if output.dim() > 2: output = output.view(output.size(0), -1)
                    if m1.dim() > 2: m1 = m1.view(m1.size(0), -1)
                    if m2.dim() > 2: m2 = m2.view(m2.size(0), -1)
                    if m3.dim() > 2: m3 = m3.view(m3.size(0), -1)
                    if final_fea.dim() > 2: final_fea = final_fea.view(final_fea.size(0), -1)
                    if f1.dim() > 2: f1 = f1.view(f1.size(0), -1)
                    if f2.dim() > 2: f2 = f2.view(f2.size(0), -1)
                    if f3.dim() > 2: f3 = f3.view(f3.size(0), -1)

                    loss_main = ce_loss(output, target)
                    loss_ce_students = ce_loss(m1, target) + ce_loss(m2, target) + ce_loss(m3, target)

                    with torch.no_grad():
                        teacher_prob = F.softmax(output / temperature, dim=1)
                        teacher_conf, teacher_pred = teacher_prob.max(dim=1)

                        correct_mask = (teacher_pred == target)
                        if correct_mask.any():
                            total_correct_conf += teacher_conf[correct_mask].mean().item()
                            valid_conf_batches += 1

                        mask = (teacher_conf >= dynamic_threshold)

                        if mask.any():
                            passed_targets = target[mask]
                            class_kd_counts += torch.bincount(passed_targets, minlength=100)
                    
                    total_keep_ratio += mask.float().mean().item()

                    def masked_kd(student_logits):
                        if not mask.any(): return torch.zeros((), device=device)
                        logp = F.log_softmax(student_logits / temperature, dim=1)
                        return F.kl_div(logp, teacher_prob, reduction="none").sum(dim=1)[mask].mean()

                    loss_kd_students = (masked_kd(m1) + masked_kd(m2) + masked_kd(m3)) * (temperature ** 2)
                    loss_feat_students = feat_loss(f1, final_fea.detach()) + feat_loss(f2, final_fea.detach()) + feat_loss(f3, final_fea.detach())

                    with torch.no_grad():
                        total_feat_ratio += 1.0 
                        rejected_mask = ~mask
                        if rejected_mask.any():
                            total_rfd += F.mse_loss(f1[rejected_mask].detach(), final_fea[rejected_mask].detach()).item()

                    loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
                else:
                    output = out[-1] if isinstance(out, tuple) else out
                    loss = ce_loss(output, target)
                    total_keep_ratio += 1.0
                    total_feat_ratio += 1.0

            total_loss += float(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    denom = max(1, len(train_dataloader) * max(1, getattr(args, "epochs", 1)))
    return total_loss / denom, total_keep_ratio / denom, total_rfd / denom, total_feat_ratio / denom, total_correct_conf / max(1, valid_conf_batches), (class_kd_counts == 0).sum().item(), class_kd_counts.std().item()

def fedbyot_spatial_temporal(net, prev_net, train_dataloader, optimizer, device, args, current_round=0):
    # 하이퍼파라미터
    alpha_temporal = args.byot_alpha 
    T_temporal = args.temperature 
    
    # [수정] 인자에서 받아온 값 사용
    threshold = args.kd_conf_threshold       # 예: 0.1
    min_keep_ratio = args.kd_min_keep_ratio  # 예: 0.5 (최소 50%는 살림)
    
    T_spatial = args.temperature
    
    GLOBAL_WARMUP_ROUNDS = 50  # 10 -> 50으로 대폭 상향

    if current_round < GLOBAL_WARMUP_ROUNDS:
        # 워밍업 기간: 모든 증류 Loss 비활성화
        alpha_temporal = 0.0  # 시간적 증류 끔
        weight_kd = 0.0       # 공간적 증류(BYOT) 끔
    else:
        # 워밍업 종료: 원래 설정대로 복구
        alpha_temporal = args.byot_alpha 
        weight_kd = 1.0

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    net.train()
    if prev_net is not None:
        prev_net.eval()
        prev_net.to(device)

    total_loss = 0.0
    total_ratio = 0.0
    
    for step, (x, target) in enumerate(train_dataloader):
        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 1. Forward Pass
        out = net(x)
        output = out[0]      # Main Head
        students = out[1:4]  # Student Heads

        prev_output = None
        if prev_net is not None:
            with torch.no_grad():
                prev_out = prev_net(x)
                if isinstance(prev_out, tuple) or isinstance(prev_out, list):
                    prev_output = prev_out[0]
                else:
                    prev_output = prev_out

        # 2. Loss Calculation
        
        # [A] Main Head 학습 (Warm-up 적용)
        loss_hard = criterion_ce(output, target)
        loss_temporal = 0.0
        
        # Warm-up: 10라운드 이후부터 Temporal KD 적용
        if prev_output is not None and current_round > 10:
            p_s = F.log_softmax(output / T_temporal, dim=1)
            p_t = F.softmax(prev_output / T_temporal, dim=1)
            loss_temporal = criterion_kl(p_s, p_t) * (T_temporal ** 2)
        
        if prev_output is not None:
            loss_main = (1 - alpha_temporal) * loss_hard + alpha_temporal * loss_temporal
        else:
            loss_main = loss_hard

        # [B] Spatial BYOT Loss (Selective + Top-K Fallback)
        loss_ce_students = 0.0
        loss_kd_students = 0.0
        
        # Teacher(Main Head)의 신뢰도 계산
        softmax_output = F.softmax(output, dim=1)
        prob, _ = torch.max(softmax_output, dim=1) # [Batch_size]
        
        # --- [핵심 수정] Selective Mask 생성 로직 ---
        # 1. 기본 마스크: 임계값 넘는 것만 1
        mask = (prob >= threshold).float()
        
        # 2. 비율 보장 로직 (Top-K)
        # 현재 배치에서 살아남은 비율 계산
        current_ratio = mask.mean().item()
        
        # 만약 살아남은게 설정한 최소 비율보다 적다면?
        if current_ratio < min_keep_ratio:
            # 배치 크기 * 최소비율 개수만큼 뽑음
            num_keep = int(x.size(0) * min_keep_ratio)
            if num_keep > 0:
                # 확신도가 높은 순서대로 상위 K개 인덱스 구함
                _, topk_indices = torch.topk(prob, k=num_keep)
                # 마스크를 0으로 초기화하고 상위 K개만 1로 채움 (Or 기존 통과한 애들과 합집합)
                mask = torch.zeros_like(prob).float()
                mask[topk_indices] = 1.0
        
        valid_ratio = mask.mean().item() # 최종 반영 비율
        
        # Soft Target for Spatial KD
        soft_target_spatial = F.softmax(output / T_spatial, dim=1).detach()

        for student_logit in students:
            # Hard Loss
            loss_ce_students += criterion_ce(student_logit, target)
            
            # Soft Loss (Selective)
            log_prob_student = F.log_softmax(student_logit / T_spatial, dim=1)
            kl_loss_per_sample = F.kl_div(log_prob_student, soft_target_spatial, reduction='none').sum(dim=1)
            
            # 마스크 적용: 선택된 샘플의 Loss만 평균냄
            # (선택된 샘플이 하나도 없으면 0)
            if mask.sum() > 0:
                loss_kd_students += (kl_loss_per_sample * mask).sum() / mask.sum() * (T_spatial ** 2)
            else:
                loss_kd_students += 0.0

        # [C] Optimization
        loss = loss_main + loss_ce_students + weight_kd * loss_kd_students
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ratio += valid_ratio

    avg_loss = total_loss / len(train_dataloader)
    avg_ratio = total_ratio / len(train_dataloader)

    return avg_loss, avg_ratio

def fedbyot_cosine(net, train_dataloader, optimizer, device, args):
    """
    FedBYOT + Cosine Similarity Loss
    - Feature Loss를 MSE 대신 Cosine Similarity로 변경
    - 목적: Teacher의 '값'이 아닌 '특징의 방향성'을 학습하여 Generalization 성능 향상
    """
    temperature = args.temperature
    alpha = args.byot_alpha   # KD Loss 가중치
    beta = args.byot_beta     # Feature Loss 가중치
    
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    # criterion_mse는 사용하지 않음
    
    # [Helper] Cosine Loss 함수 정의
    def cosine_loss(student_feat, teacher_feat):
        # 1. 차원 평탄화 (Batch, Channel, H, W) -> (Batch, Vector)
        # Spatial dimension이 있어도 벡터로 펴서 방향성을 비교합니다.
        s_flat = student_feat.view(student_feat.size(0), -1)
        t_flat = teacher_feat.detach().view(teacher_feat.size(0), -1)
        
        # 2. Cosine Similarity 계산 (1에 가까울수록 같음)
        # Loss는 작아야 하므로 (1 - similarity)를 반환 (0에 가까울수록 좋음)
        return (1 - F.cosine_similarity(s_flat, t_flat, dim=1)).mean()
    
    total_loss = 0.0
    net.train()
    
    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device).long()
            optimizer.zero_grad()

            out = net(x) 
            
            if isinstance(out, tuple) and len(out) == 8:
                (output, m1, m2, m3, final_fea, f1, f2, f3) = out
                
                # 1. Main Loss
                loss_main = criterion_ce(output, target)
                
                # 2. Student CE Loss
                loss_ce_students = (criterion_ce(m1, target) + 
                                    criterion_ce(m2, target) + 
                                    criterion_ce(m3, target))
                
                # 3. KD Loss
                with torch.no_grad():
                    teacher_prob = F.softmax(output / temperature, dim=1)
                
                loss_kd_students = (
                    criterion_kl(F.log_softmax(m1 / temperature, dim=1), teacher_prob) +
                    criterion_kl(F.log_softmax(m2 / temperature, dim=1), teacher_prob) +
                    criterion_kl(F.log_softmax(m3 / temperature, dim=1), teacher_prob)
                ) * (temperature ** 2)
                
                # 4. Feature Loss (Cosine Similarity 적용)
                # Teacher의 'Feature Map 방향'을 Student가 따라감
                loss_feat_students = (
                    cosine_loss(f1, final_fea) +
                    cosine_loss(f2, final_fea) +
                    cosine_loss(f3, final_fea)
                )

                loss = loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students

            else:
                # 일반 모델 Fallback
                if isinstance(out, tuple): _, output = out
                else: output = out
                loss = criterion_ce(output, target)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

    return total_loss / len(train_dataloader) / args.epochs

def fedbyot_sam(net, train_dataloader, optimizer, device, args):
    """
    FedBYOT + SAM Optimizer
    - SGD 대신 SAM을 사용하여 Loss Landscape의 평평한 지점(Flat Minima)을 찾음.
    - Non-IID 환경에서 Generalization 성능을 높이는 데 매우 효과적임.
    """
    temperature = args.temperature
    alpha = args.byot_alpha
    beta = args.byot_beta
    
    # SAM Optimizer로 감싸기 (기존 optimizer 설정 무시하고 새로 생성해야 함)
    base_optimizer = torch.optim.SGD
    sam_optimizer = SAM(net.parameters(), base_optimizer, rho=0.05, 
                        lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    criterion_mse = nn.MSELoss().to(device)
    
    # [Helper] BYOT Loss 계산 함수 (코드가 길어서 함수로 분리)
    def compute_loss(model, inputs, targets):
        out = model(inputs)
        
        if isinstance(out, tuple) and len(out) == 8:
            (output, m1, m2, m3, final_fea, f1, f2, f3) = out
            
            loss_main = criterion_ce(output, targets)
            
            loss_ce_students = (criterion_ce(m1, targets) + 
                                criterion_ce(m2, targets) + 
                                criterion_ce(m3, targets))
            
            with torch.no_grad():
                teacher_prob = F.softmax(output / temperature, dim=1)
            
            loss_kd_students = (
                criterion_kl(F.log_softmax(m1 / temperature, dim=1), teacher_prob) +
                criterion_kl(F.log_softmax(m2 / temperature, dim=1), teacher_prob) +
                criterion_kl(F.log_softmax(m3 / temperature, dim=1), teacher_prob)
            ) * (temperature ** 2)
            
            loss_feat_students = (
                criterion_mse(f1, final_fea.detach()) +
                criterion_mse(f2, final_fea.detach()) +
                criterion_mse(f3, final_fea.detach())
            )
            
            return loss_main + (1 - alpha) * loss_ce_students + alpha * loss_kd_students + beta * loss_feat_students
        else:
            if isinstance(out, tuple): _, output = out
            else: output = out
            return criterion_ce(output, targets)

    total_loss = 0.0
    net.train()
    
    for epoch in range(args.epochs):
        for step, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device).long()
            
            # --- SAM Step 1 ---
            # 1. 첫 번째 Forward & Backward
            loss_1 = compute_loss(net, x, target)
            loss_1.backward()
            
            # 2. w + e(w)로 이동 (Loss가 가장 높은 주변부로 이동)
            sam_optimizer.first_step(zero_grad=True)
            
            # --- SAM Step 2 ---
            # 3. 두 번째 Forward & Backward (이동한 위치에서 기울기 계산)
            loss_2 = compute_loss(net, x, target)
            loss_2.backward()
            
            # 4. 원래 위치(w)로 돌아와서 실제로 업데이트
            sam_optimizer.second_step(zero_grad=True)
            
            total_loss += loss_1.item()

    return total_loss / len(train_dataloader) / args.epochs

# train.py 파일에 추가/수정
# (v2에서 더 나아가, WARMUP_ROUNDS를 추가한 v3 최종본)

def fedflocora_byot(net, train_dataloader, optimizer, device, args, current_round):
    """
    FLoCoRA + BYOT 로컬 학습 루프 (워밍업 단계에서 teacher만 학습 → 이후 student 손실 추가)
    - BYOT forward: (output, m1, m2, m3, final_fea, f1, f2, f3) 길이 8 튜플 가정
    - 일반 모델: (features, logits) 또는 logits 단독도 안전 처리
    """
    # ---- 0) 하이퍼 파라미터/손실 정의 ----
    WARMUP_ROUNDS  = getattr(args, "warmup_rounds", 50)   # 기본 50, args에 있으면 덮어씀
    temperature    = float(getattr(args, "temperature", 4.0))
    alpha          = float(getattr(args, "byot_alpha", 0.15))
    beta           = float(getattr(args, "byot_beta", 0.05))

    ce_loss   = nn.CrossEntropyLoss().to(device)
    kd_loss   = nn.KLDivLoss(reduction="batchmean").to(device)
    feat_loss = nn.MSELoss().to(device)

    total_loss = 0.0
    total_teacher_ce_loss = 0.0
    num_batches = 0

    net.train()  # 보장

    # ---- 1) 로컬 학습 ----
    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            out = net(x)

            # ---- 2) BYOT / 일반 모델 분기 ----
            if isinstance(out, tuple):
                if len(out) == 8:
                    # BYOT: (teacher logits, m1, m2, m3, final_fea, f1, f2, f3)
                    (output,
                     middle_output1, middle_output2, middle_output3,
                     final_fea,
                     middle1_fea, middle2_fea, middle3_fea) = out

                    # 2a) 항상 teacher CE 포함 (워밍업/이후 모두)
                    loss_main = ce_loss(output, target)
                    loss_teacher_ce = loss_main.item()
                    loss = loss_main

                    # 2b) 워밍업 종료 후에만 student 손실 추가
                    if current_round >= WARMUP_ROUNDS:
                        with torch.no_grad():
                            teacher_prob = F.softmax(output / temperature, dim=1)

                        # 학생 1
                        loss_m1 = ce_loss(middle_output1, target)
                        loss_m1_kd   = (temperature ** 2) * kd_loss(F.log_softmax(middle_output1 / temperature, dim=1), teacher_prob)
                        loss_m1_feat = feat_loss(middle1_fea, final_fea.detach())

                        # 학생 2
                        loss_m2 = ce_loss(middle_output2, target)
                        loss_m2_kd   = (temperature ** 2) * kd_loss(F.log_softmax(middle_output2 / temperature, dim=1), teacher_prob)
                        loss_m2_feat = feat_loss(middle2_fea, final_fea.detach())

                        # 학생 3
                        loss_m3 = ce_loss(middle_output3, target)
                        loss_m3_kd   = (temperature ** 2) * kd_loss(F.log_softmax(middle_output3 / temperature, dim=1), teacher_prob)
                        loss_m3_feat = feat_loss(middle3_fea, final_fea.detach())

                        loss_student = ( 1 - alpha ) * (loss_m1 + loss_m2 + loss_m3) \
                                     + alpha * (loss_m1_kd + loss_m2_kd + loss_m3_kd) \
                                     + beta  * (loss_m1_feat + loss_m2_feat + loss_m3_feat)

                        loss = loss + loss_student

                elif len(out) == 2:
                    # 일반 모델: (features, logits)
                    _, logits = out
                    loss = ce_loss(logits, target)
                else:
                    raise RuntimeError(f"Unexpected forward() tuple length: {len(out)}")
            else:
                # logits 단독
                logits = out
                loss = ce_loss(logits, target)

            # ---- 3) 역전파/최적화 ----
            total_loss += float(loss.item())
            total_teacher_ce_loss += loss_teacher_ce
            num_batches += 1

            loss.backward()
            optimizer.step()

    net.zero_grad(set_to_none=True)
    # 에폭/배치 평균 손실 반환
    denom = max(1, num_batches)
    return total_loss / denom, total_teacher_ce_loss / denom

def fedflocora_byot_v1(net, train_dataloader, optimizer, device, args, current_round):
    """
    FLoCoRA + BYOT (v1 - 최초 실패 버전)
    - Fed-BYOT(Baseline 2)의 10-Loss 구조를 FLoCoRA(동결)에 그대로 적용
    - (예상: 10% [cite: 443-469, 361-377, 470-486, 990-995] 수렴 실패)
    """
    temperature    = float(getattr(args, "temperature", 4.0))
    alpha         = float(getattr(args, "byot_alpha", 0.15))
    beta          = float(getattr(args, "byot_beta", 0.05))

    ce_loss   = nn.CrossEntropyLoss().to(device)
    kd_loss   = nn.KLDivLoss(reduction="batchmean").to(device)
    feat_loss = nn.MSELoss().to(device)
    
    total_loss = 0.0
    total_teacher_ce_loss = 0.0
    num_batches = 0
    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            out = net(x)

            if isinstance(out, tuple) and len(out) == 8:
                (output,
                 middle_output1, middle_output2, middle_output3,
                 final_fea,
                 middle1_fea, middle2_fea, middle3_fea) = out

                # 1) 모든 헤드가 CE Loss 계산
                loss_main = ce_loss(output, target)
                loss_teacher_ce = loss_main.item()
                
                loss_m1 = ce_loss(middle_output1, target)
                loss_m2 = ce_loss(middle_output2, target)
                loss_m3 = ce_loss(middle_output3, target)
                total_ce = loss_m1 + loss_m2 + loss_m3 # (Fed-BYOT 원본 로직)

                # 2) 학생 KD Loss
                with torch.no_grad():
                    teacher_prob = F.softmax(output / temperature, dim=1)
                loss_m1_kd = (temperature ** 2) * kd_loss(F.log_softmax(middle_output1 / temperature, dim=1), teacher_prob)
                loss_m2_kd = (temperature ** 2) * kd_loss(F.log_softmax(middle_output2 / temperature, dim=1), teacher_prob)
                loss_m3_kd = (temperature ** 2) * kd_loss(F.log_softmax(middle_output3 / temperature, dim=1), teacher_prob)
                total_kd = loss_m1_kd + loss_m2_kd + loss_m3_kd

                # 3) 학생 Feature Loss
                loss_m1_feat = feat_loss(middle1_fea, final_fea.detach())
                loss_m2_feat = feat_loss(middle2_fea, final_fea.detach())
                loss_m3_feat = feat_loss(middle3_fea, final_fea.detach())
                total_feat = loss_m1_feat + loss_m2_feat + loss_m3_feat

                # 4) 최종 Loss (Fed-BYOT 원본과 동일)
                loss = loss_main + (1 - alpha) * total_ce + alpha * total_kd + beta * total_feat
            else:
                _, logits = unpack_forward(net, out) # (Unpack 헬퍼 사용 가정)
                loss = ce_loss(logits, target)

            total_loss += float(loss.item())
            total_teacher_ce_loss += loss_teacher_ce
            num_batches += 1
            loss.backward()
            optimizer.step()

    net.zero_grad(set_to_none=True)
    denom = max(1, num_batches)
    return total_loss / denom, total_teacher_ce_loss / denom

def fedflocora_byot_v2(net, train_dataloader, optimizer, device, args, current_round):
    """
    FLoCoRA + BYOT (v2 - 학생 CE Loss 제거)
    - v1의 실패 원인을 "학생 CE Loss"로 보고 그것만 제거
    - (예상: 10% [cite: 969-974, 992-995, 1373-1380, 1382-1390, 1391-1397]에서 이륙 시작)
    """
    temperature   = float(getattr(args, "temperature", 4.0))
    alpha         = float(getattr(args, "byot_alpha", 0.15))
    beta          = float(getattr(args, "byot_beta", 0.05))

    ce_loss   = nn.CrossEntropyLoss().to(device)
    kd_loss   = nn.KLDivLoss(reduction="batchmean").to(device)
    feat_loss = nn.MSELoss().to(device)
    
    total_loss = 0.0
    total_teacher_ce_loss = 0.0
    num_batches = 0
    net.train()

    for _ in range(getattr(args, "epochs", 1)):
        for x, target in train_dataloader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            out = net(x)

            if isinstance(out, tuple) and len(out) == 8:
                (output,
                 middle_output1, middle_output2, middle_output3,
                 final_fea,
                 middle1_fea, middle2_fea, middle3_fea) = out

                # 1) Teacher CE Loss (v2 핵심: Teacher만 CE Loss 수행)
                loss = ce_loss(output, target)
                loss_teacher_ce = loss.item()

                # 2) 학생 Loss (CE Loss 없음)
                with torch.no_grad():
                    teacher_prob = F.softmax(output / temperature, dim=1)
                
                loss_m1_kd   = (temperature ** 2) * kd_loss(F.log_softmax(middle_output1 / temperature, dim=1), teacher_prob)
                loss_m1_feat = feat_loss(middle1_fea, final_fea.detach())
                loss_m2_kd   = (temperature ** 2) * kd_loss(F.log_softmax(middle_output2 / temperature, dim=1), teacher_prob)
                loss_m2_feat = feat_loss(middle2_fea, final_fea.detach())
                loss_m3_kd   = (temperature ** 2) * kd_loss(F.log_softmax(middle_output3 / temperature, dim=1), teacher_prob)
                loss_m3_feat = feat_loss(middle3_fea, final_fea.detach())

                loss_student = alpha * (loss_m1_kd + loss_m2_kd + loss_m3_kd) \
                             + beta  * (loss_m1_feat + loss_m2_feat + loss_m3_feat)
                
                # 3) 최종 Loss (v2)
                loss = loss + loss_student
            else:
                _, logits = unpack_forward(net, out) # (Unpack 헬퍼 사용 가정)
                loss = ce_loss(logits, target)

            total_loss += float(loss.item())
            total_teacher_ce_loss += loss_teacher_ce
            num_batches += 1
            loss.backward()
            optimizer.step()

    net.zero_grad(set_to_none=True)
    denom = max(1, num_batches)
    return total_loss / denom, total_teacher_ce_loss / denom

'''

def train_local_net(dataloaders, nets, global_model, prev_nets, prev_global_model, device, round, lr, args, logger):
    total_loss = 0.0
    total_ratio = 0.0 
    total_entropy = 0.0
    total_rfd = 0.0        
    total_feat_ratio = 0.0 
    total_byot_alpha_mean = 0.0
    total_byot_alpha_min = 0.0
    total_byot_alpha_max = 0.0
    client_byot_alpha_stats = {}
    total_correct_conf = 0.0
    total_zero_kd = 0.0 
    total_kd_std = 0.0  
    total_branch_freq_stats = init_train_branch_freq_stats() if getattr(args, "log_train_branch_frequency_stats", False) else {}
    
    # [NEW] 시간 및 연산 효율 측정을 위한 누적 변수
    total_time = 0.0
    total_efficiency = 0.0

    for net_id, net in nets.items():
        start_time = time.time() # [NEW] 로컬 클라이언트 학습 시작 시간 기록
        net.train()
        
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                   lr=lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                   lr=lr, weight_decay=args.reg, amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                  lr=lr, momentum=args.momentum, weight_decay=args.reg)

        # [NEW] 기본값 설정
        ratio, rfd, feat_ratio, entropy = 1.0, 0.0, 1.0, 0.0
        byot_alpha_mean = float(getattr(args, "byot_alpha", 0.0))
        byot_alpha_min = byot_alpha_mean
        byot_alpha_max = byot_alpha_mean
        correct_conf, zero_kd_classes, kd_std = 0.0, 0, 0.0
        wall_clock_time = 0.0
        compute_efficiency = 1.0 # 기본 알고리즘들은 매 에폭 모든 데이터를 연산함 (100%)

        prev_net = None
        if prev_nets is not None and net_id in prev_nets:
            prev_net = prev_nets[net_id]

        # --- 알고리즘 분기 ---
        
        if args.alg == 'fedbyot_selective_greedy':
            if dataloaders[net_id] is None:
                print(f"[-] Client {net_id} has no data. Skipping BYOT training.")
                # 데이터가 없을 경우 기본값으로 채움
                loss, wall_clock_time, compute_efficiency, correct_conf, zero_kd_classes, kd_std, entropy = 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0
                ratio = compute_efficiency
            else:
                loss, wall_clock_time, compute_efficiency, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_selective_greedy(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
                ratio = compute_efficiency
            
        elif args.alg == 'fedbyot_logit_adj_greedy':
            loss, wall_clock_time, compute_efficiency, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_logit_adj_greedy(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            ratio = compute_efficiency
            
        elif args.alg == 'fedbyot_lc':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_lc(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time

        elif args.alg == 'fedbyot_rs':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_rs(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_selective':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_selective(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            # 순전파는 100% 수행하므로 연산 비율은 1.0 유지

        elif args.alg == 'fedbyot_selective_ce_fallback':
            result = fedbyot_selective_ce_fallback(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            (
                loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes,
                kd_std, entropy, byot_alpha_mean, byot_alpha_min, byot_alpha_max,
            ) = result
            wall_clock_time = time.time() - start_time
        
        elif args.alg == 'fedbyot_lc_selective':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_lc_selective(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time

        elif args.alg == 'fedbyot_rs_selective':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_rs_selective(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
        
        elif args.alg == 'fedbyot_lc_greedy':
            loss, wall_clock_time, compute_efficiency, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_lc_greedy(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            ratio = compute_efficiency

        elif args.alg == 'fedbyot_rs_greedy':
            loss, wall_clock_time, compute_efficiency, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_rs_greedy(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            ratio = compute_efficiency
            
        elif args.alg == 'fedavg' or args.alg == 'fedavgM' or args.alg == 'fedag' or args.alg == 'fedadam' or args.alg == 'fedexp' or args.alg == 'flocora':
            if dataloaders[net_id] is None:
                print(f"[-] Client {net_id} has no data. Skipping local training.")
                loss = 0.0  # 데이터가 없으니 학습을 건너뛰고 loss를 0으로 처리
            else:
                loss = fedavg(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedrs':
            loss = fedrs(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedlc':
            loss = fedlc(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedprox':
            loss, correct_conf, entropy = fedprox(net, global_model, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'moon':
            loss, correct_conf, entropy = moon(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedrcl':
            loss = fedrcl(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedsol':
            loss = fedsol(net, global_model, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedacg':
            loss = fedacg(net, global_model, prev_global_model, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot':
            fedbyot_result = fedbyot(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            branch_freq_stats = None
            if len(fedbyot_result) == 12:
                (
                    loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy,
                    byot_alpha_mean, byot_alpha_min, byot_alpha_max, branch_freq_stats,
                ) = fedbyot_result
            elif len(fedbyot_result) == 11:
                (
                    loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy,
                    byot_alpha_mean, byot_alpha_min, byot_alpha_max,
                ) = fedbyot_result
            else:
                loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_result
            merge_train_branch_freq_stats(total_branch_freq_stats, branch_freq_stats)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_logit_adj':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_logit_adj(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_dynamic_temp':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std, entropy = fedbyot_dynamic_temp(net, global_model, prev_net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_flexmatch':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std = fedbyot_flexmatch(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_freematch':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std = fedbyot_freematch(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_percentile':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std = fedbyot_percentile(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_round_aware':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std = fedbyot_round_aware(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        elif args.alg == 'fedbyot_focal':
            loss, ratio, rfd, feat_ratio, correct_conf, zero_kd_classes, kd_std = fedbyot_focal(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
            
        else: 
            loss = fedavg(net, dataloaders[net_id], optimizer, device, args)
            wall_clock_time = time.time() - start_time
             
        # 누적 계산
        total_loss += loss
        total_entropy += entropy
        total_ratio += ratio 
        total_rfd += rfd
        total_feat_ratio += feat_ratio 
        total_byot_alpha_mean += byot_alpha_mean
        total_byot_alpha_min += byot_alpha_min
        total_byot_alpha_max += byot_alpha_max
        client_byot_alpha_stats[int(net_id)] = {
            "mean": float(byot_alpha_mean),
            "min": float(byot_alpha_min),
            "max": float(byot_alpha_max),
        }
        total_correct_conf += correct_conf
        total_zero_kd += zero_kd_classes 
        total_kd_std += kd_std           
        
        # [NEW] 시간 및 연산 효율 누적
        total_time += wall_clock_time
        total_efficiency += compute_efficiency

    num_clients = len(nets)
    avg_loss = total_loss / num_clients
    avg_ratio = total_ratio / num_clients
    avg_correct_conf = total_correct_conf / num_clients
    avg_zero_kd = total_zero_kd / num_clients 
    avg_kd_std = total_kd_std / num_clients   
    avg_rfd = total_rfd / num_clients
    avg_feat_ratio = total_feat_ratio / num_clients
    avg_byot_alpha_mean = total_byot_alpha_mean / num_clients
    avg_byot_alpha_min = total_byot_alpha_min / num_clients
    avg_byot_alpha_max = total_byot_alpha_max / num_clients
    avg_entropy = total_entropy / num_clients
    args._last_client_byot_alpha_stats = client_byot_alpha_stats
    args._last_train_branch_frequency_stats = finalize_train_branch_freq_stats(total_branch_freq_stats)
    
    # [NEW] 시간 및 연산 효율 평균
    avg_time = total_time / num_clients
    avg_efficiency = total_efficiency / num_clients
    
    # [수정] 로그 메시지에 Time과 Processed Data 추가
    status = "[Warmup]" if getattr(args, 'alg', '') == 'fedbyot' and round < getattr(args, 'warmup_rounds', 0) else "[Distill]"
    logger.info(
        f'At round: {round}, avg_loss: {avg_loss:.4f}, KD%: {avg_ratio*100:.1f}%, '
        f'EffAlpha(mean/min/max): {avg_byot_alpha_mean:.4f}/{avg_byot_alpha_min:.4f}/{avg_byot_alpha_max:.4f}, '
        f'Conf: {avg_correct_conf:.4f}, Entropy: {avg_entropy:.4f}, ZeroKD: {avg_zero_kd:.1f}, '
        f'Time: {avg_time:.2f}s, Processed: {avg_efficiency*100:.1f}%'
    )
    
    lr = adjust_lr(round, lr, args)
    
    # [유지] main.py와 호환되도록 기존 반환 포맷 유지
    return avg_loss, lr, avg_ratio, avg_rfd, avg_feat_ratio, avg_byot_alpha_mean, avg_byot_alpha_min, avg_byot_alpha_max
