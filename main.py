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

LAYERWISE_UPDATE_GROUPS = (
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "branch_private",
    "teacher_head",
    "other",
)


def _args_snapshot(args):
    """Return a stable, serialization-safe snapshot of CLI arguments.

    Training attaches private runtime diagnostics to ``args``.  Keeping a
    live ``vars(args)`` reference in result dictionaries lets those mutable
    diagnostics leak into the final pickle and can create arbitrarily deep
    structures across rounds.  Private runtime fields are not experiment
    configuration, so exclude them from checkpoints and result metadata.
    """
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def _extract_dataset_targets(dataset):
    """Return labels for a dataset or a torch Subset-style wrapper."""
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        parent_targets = _extract_dataset_targets(dataset.dataset)
        if parent_targets is not None:
            return parent_targets[np.asarray(dataset.indices, dtype=np.int64)]

    for attr in ('targets', 'target', 'labels'):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if isinstance(values, torch.Tensor):
                return values.detach().cpu().numpy()
            return np.asarray(values)
    return None


def _layerwise_update_group(key):
    if key.startswith(("conv1.", "bn1.")):
        return "stem"
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        if key.startswith(f"{layer_name}."):
            return layer_name
    if key.startswith("fc."):
        return "teacher_head"
    if any(token in key for token in ("bottleneck", "middle_fc", "downsample")):
        return "branch_private"
    return "other"


def _compute_update_drift_for_keys(old_w, nets_this_round, fed_avg_freqs, keys, eps=1e-12):
    """Compute weighted pre-aggregation update geometry for one parameter group."""
    client_ids = list(nets_this_round.keys())
    if not client_ids or not keys:
        return None

    mean_update = {}
    for key in keys:
        mean = torch.zeros_like(old_w[key], device='cpu')
        old_value = old_w[key].detach().cpu()
        for idx, client_id in enumerate(client_ids):
            local_value = nets_this_round[client_id].state_dict()[key].detach().cpu()
            mean += float(fed_avg_freqs[idx]) * (local_value - old_value)
        mean_update[key] = mean

    mean_norm_sq = sum(
        float(torch.sum(update * update).item()) for update in mean_update.values()
    )
    mean_norm = mean_norm_sq ** 0.5
    update_norm = 0.0
    update_norm_sq = 0.0
    divergence = 0.0
    cosine_sum = 0.0

    for idx, client_id in enumerate(client_ids):
        weight = float(fed_avg_freqs[idx])
        state = nets_this_round[client_id].state_dict()
        client_norm_sq = 0.0
        client_dot_mean = 0.0
        client_divergence = 0.0
        for key in keys:
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
        "norm": update_norm,
        "norm_sq": update_norm_sq,
        "mean_norm": mean_norm,
        "divergence": divergence,
        "relative_drift": divergence / (mean_norm_sq + eps),
        "cosine": cosine_sum,
    }


def compute_client_update_drift(old_w, nets_this_round, fed_avg_freqs, layerwise=False, eps=1e-12):
    """Measure total and optionally layer-wise client update disagreement.

    The layer-wise metrics separate the shared trunk blocks from branch-private
    modules.  They are required to test whether a branch-loss intervention
    changes the actual representation update geometry, rather than merely its
    output probabilities.
    """
    float_keys = [key for key, value in old_w.items() if torch.is_floating_point(value)]
    if not float_keys:
        return {}

    with torch.no_grad():
        total_stats = _compute_update_drift_for_keys(
            old_w, nets_this_round, fed_avg_freqs, float_keys, eps
        )
        if total_stats is None:
            return {}
        metrics = {
            "client_update_norm": total_stats["norm"],
            "client_update_norm_sq": total_stats["norm_sq"],
            "client_mean_update_norm": total_stats["mean_norm"],
            "client_update_divergence": total_stats["divergence"],
            "client_relative_drift": total_stats["relative_drift"],
            "client_update_cosine": total_stats["cosine"],
        }
        if not layerwise:
            return metrics

        for group in LAYERWISE_UPDATE_GROUPS:
            group_keys = [key for key in float_keys if _layerwise_update_group(key) == group]
            stats = _compute_update_drift_for_keys(
                old_w, nets_this_round, fed_avg_freqs, group_keys, eps
            )
            for stat_name in ("norm", "norm_sq", "mean_norm", "divergence", "relative_drift", "cosine"):
                metrics[f"layer_update_{group}_{stat_name}"] = (
                    None if stats is None else stats[stat_name]
                )
        return metrics

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
    used_weights = []

    for idx, client_id in enumerate(nets_this_round.keys()):
        dataloader = dataloaders_this_round.get(client_id)
        if dataloader is None:
            continue
        model = nets_this_round[client_id]
        was_training = model.training
        model.eval()
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
        used_weights.append(float(fed_avg_freqs[idx]))

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
    return metrics


def _gradient_vector_for_prefixes(model, loss, prefixes):
    """Flatten a loss gradient over a selected shared-prefix parameter set."""
    params = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(prefixes)
    ]
    if not params:
        return None
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = [
        (torch.zeros_like(parameter) if grad is None else grad).detach().flatten()
        for parameter, grad in zip(params, grads)
    ]
    return torch.cat(flat)


def _branch_teacher_gradient_alignment_for_model(model, dataloader, device, max_batches):
    """Compare branch CE/KD and teacher CE gradients on the same client batches."""
    prefixes = {
        "b1": ("conv1.", "bn1.", "layer1."),
        "b2": ("conv1.", "bn1.", "layer1.", "layer2."),
        "b3": ("conv1.", "bn1.", "layer1.", "layer2.", "layer3."),
    }
    sums = {
        name: {
            "ce_cosine": 0.0, "ce_norm_ratio": 0.0,
            "kd_cosine": 0.0, "kd_norm_ratio": 0.0,
            "count": 0,
        }
        for name in prefixes
    }
    temperature = float(getattr(model, "_gradient_probe_temperature", 0.5))
    for batch_idx, (x, target) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x, target = x.to(device), target.to(device).long()
        out = model(x)
        if not (isinstance(out, tuple) and len(out) == 8):
            continue
        teacher_logits, m1, m2, m3, _, _, _, _ = out
        teacher_loss = F.cross_entropy(teacher_logits, target)
        with torch.no_grad():
            teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
        for name, branch_logits in zip(("b1", "b2", "b3"), (m1, m2, m3)):
            teacher_grad = _gradient_vector_for_prefixes(model, teacher_loss, prefixes[name])
            branch_ce_grad = _gradient_vector_for_prefixes(
                model, F.cross_entropy(branch_logits, target), prefixes[name]
            )
            branch_kd_loss = F.kl_div(
                F.log_softmax(branch_logits / temperature, dim=1),
                teacher_prob,
                reduction='batchmean',
            ) * (temperature ** 2)
            branch_kd_grad = _gradient_vector_for_prefixes(model, branch_kd_loss, prefixes[name])
            if teacher_grad is None or branch_ce_grad is None or branch_kd_grad is None:
                continue
            teacher_norm = float(torch.linalg.vector_norm(teacher_grad).item())
            branch_ce_norm = float(torch.linalg.vector_norm(branch_ce_grad).item())
            branch_kd_norm = float(torch.linalg.vector_norm(branch_kd_grad).item())
            ce_cosine = 0.0
            kd_cosine = 0.0
            if teacher_norm > 0.0 and branch_ce_norm > 0.0:
                ce_cosine = float(torch.dot(teacher_grad, branch_ce_grad).item() / (teacher_norm * branch_ce_norm))
            if teacher_norm > 0.0 and branch_kd_norm > 0.0:
                kd_cosine = float(torch.dot(teacher_grad, branch_kd_grad).item() / (teacher_norm * branch_kd_norm))
            sums[name]["ce_cosine"] += ce_cosine
            sums[name]["ce_norm_ratio"] += branch_ce_norm / max(teacher_norm, 1e-12)
            sums[name]["kd_cosine"] += kd_cosine
            sums[name]["kd_norm_ratio"] += branch_kd_norm / max(teacher_norm, 1e-12)
            sums[name]["count"] += 1
    return {
        name: {
            "ce_cosine": values["ce_cosine"] / values["count"],
            "ce_norm_ratio": values["ce_norm_ratio"] / values["count"],
            "kd_cosine": values["kd_cosine"] / values["count"],
            "kd_norm_ratio": values["kd_norm_ratio"] / values["count"],
        }
        for name, values in sums.items() if values["count"] > 0
    }


def compute_branch_teacher_gradient_alignment_probe(nets_this_round, dataloaders_this_round, fed_avg_freqs, device, args):
    """Client-weighted branch CE/KD versus teacher CE alignment at round start."""
    max_batches = max(1, int(getattr(args, "branch_gradient_probe_batches", 1)))
    weighted = {
        name: {
            "ce_cosine": 0.0, "ce_norm_ratio": 0.0,
            "kd_cosine": 0.0, "kd_norm_ratio": 0.0,
            "weight": 0.0,
        }
        for name in ("b1", "b2", "b3")
    }
    used_clients = 0
    for idx, client_id in enumerate(nets_this_round.keys()):
        dataloader = dataloaders_this_round.get(client_id)
        if dataloader is None:
            continue
        model = nets_this_round[client_id]
        was_training = model.training
        model._gradient_probe_temperature = float(getattr(args, "temperature", 0.5))
        model.eval()
        stats = _branch_teacher_gradient_alignment_for_model(model, dataloader, device, max_batches)
        if was_training:
            model.train()
        if not stats:
            continue
        used_clients += 1
        weight = float(fed_avg_freqs[idx])
        for name, values in stats.items():
            for metric in ("ce_cosine", "ce_norm_ratio", "kd_cosine", "kd_norm_ratio"):
                weighted[name][metric] += weight * values[metric]
            weighted[name]["weight"] += weight

    metrics = {"branch_gradient_probe_clients": used_clients}
    for name, values in weighted.items():
        if values["weight"] > 0:
            metrics[f"branch_gradient_{name}_teacher_ce_cosine"] = values["ce_cosine"] / values["weight"]
            metrics[f"branch_gradient_{name}_teacher_ce_norm_ratio"] = values["ce_norm_ratio"] / values["weight"]
            metrics[f"branch_gradient_{name}_teacher_kd_cosine"] = values["kd_cosine"] / values["weight"]
            metrics[f"branch_gradient_{name}_teacher_kd_norm_ratio"] = values["kd_norm_ratio"] / values["weight"]
    return metrics


def _active_branch_indices_for_probe(args):
    """Parse active BYOT exits without importing the training module."""
    raw = str(getattr(args, "byot_active_branches", "1,2,3") or "1,2,3").strip().lower()
    if raw in {"", "none", "off", "0"}:
        return []
    indices = []
    for token in raw.split(","):
        branch_id = int(token.strip())
        if branch_id not in (1, 2, 3):
            raise ValueError("--byot_active_branches only supports 1,2,3 for this probe.")
        if branch_id - 1 not in indices:
            indices.append(branch_id - 1)
    return indices


def _branch_shared_trunk_parameters(model):
    """Parameters that branch exits can update and that the final teacher reuses."""
    prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.")
    return [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(prefixes)
    ]


def _branch_kd_probe_target(teacher_prob, target, args):
    """Mirror the supported KD-target ablation when probing gradient geometry."""
    mode = getattr(args, "byot_branch_kd_target_mode", "full_teacher")
    if mode == "full_teacher":
        return teacher_prob
    if mode not in ("teacher_mass_uniform", "teacher_mass_uniform_batchmean"):
        raise ValueError(f"Unknown --byot_branch_kd_target_mode: {mode}")
    num_classes = teacher_prob.size(1)
    if num_classes <= 1:
        return teacher_prob
    target_prob = teacher_prob.gather(1, target.unsqueeze(1))
    if mode == "teacher_mass_uniform_batchmean":
        target_prob = target_prob.mean(dim=0, keepdim=True).expand_as(target_prob)
    uniform_non_target = (1.0 - target_prob) / float(num_classes - 1)
    target_distribution = uniform_non_target.expand_as(teacher_prob).clone()
    target_distribution.scatter_(1, target.unsqueeze(1), target_prob)
    return target_distribution


def _branch_shared_probe_loss(output, branch_logits, target, args, loss_kind):
    """Return one branch-only CE or KD loss over the configured active exits."""
    active_indices = _active_branch_indices_for_probe(args)
    if not active_indices:
        return None
    default_temperature = float(getattr(args, "temperature", 0.5))
    teacher_temperature = float(getattr(args, "byot_branch_kd_teacher_temperature", 0.0))
    student_temperature = float(getattr(args, "byot_branch_kd_student_temperature", 0.0))
    teacher_temperature = teacher_temperature if teacher_temperature > 0.0 else default_temperature
    student_temperature = student_temperature if student_temperature > 0.0 else default_temperature
    if loss_kind == "ce":
        smoothing = float(getattr(args, "byot_branch_ce_label_smoothing", 0.0))
        weight = float(getattr(args, "byot_branch_ce_weight", 1.0))
        losses = [
            weight * F.cross_entropy(branch_logits[idx], target, label_smoothing=smoothing)
            for idx in active_indices
        ]
    elif loss_kind == "kd":
        with torch.no_grad():
            teacher_prob = F.softmax(output / teacher_temperature, dim=1)
            kd_target = _branch_kd_probe_target(teacher_prob, target, args)
        scale_mode = getattr(args, "byot_branch_kd_loss_scale_mode", "native_t2")
        if scale_mode == "native_t2":
            loss_scale = student_temperature ** 2
        elif scale_mode == "gradient_prefactor_one":
            loss_scale = student_temperature
        else:
            raise ValueError(f"Unknown --byot_branch_kd_loss_scale_mode: {scale_mode}")
        losses = [
            F.kl_div(
                F.log_softmax(branch_logits[idx] / student_temperature, dim=1),
                kd_target,
                reduction="batchmean",
            ) * loss_scale
            for idx in active_indices
        ]
    else:
        raise ValueError(f"Unknown branch shared-gradient loss kind: {loss_kind}")
    total = sum(losses)
    if getattr(args, "byot_branch_loss_reduction", "sum") == "mean":
        total = total / len(active_indices)
    return total


def _average_branch_shared_gradient(model, dataloader, device, args, loss_kind, max_batches):
    """Client gradient of one branch loss with respect to shared trunk only."""
    params = _branch_shared_trunk_parameters(model)
    if not params:
        return None
    gradients = []
    for batch_idx, (x, target) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x, target = x.to(device), target.to(device).long()
        if getattr(args, "byot_branch_gradient_mode", "attached") == "detach_shared":
            # This control blocks exactly the branch-loss path into the
            # shared trunk, so record the gradient that is actually applied.
            out = model(x, detach_branch_inputs=True)
        else:
            out = model(x)
        if not (isinstance(out, tuple) and len(out) == 8):
            continue
        output, m1, m2, m3 = out[:4]
        loss = _branch_shared_probe_loss(output, (m1, m2, m3), target, args, loss_kind)
        if loss is None:
            continue
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat = [
            (torch.zeros_like(parameter) if grad is None else grad).detach().cpu().flatten()
            for parameter, grad in zip(params, grads)
        ]
        gradients.append(torch.cat(flat))
    if not gradients:
        return None
    return torch.stack(gradients, dim=0).mean(dim=0)


def compute_branch_shared_gradient_dispersion_probe(
    nets_this_round, dataloaders_this_round, fed_avg_freqs, device, args
):
    """Measure CE/KD branch-gradient disagreement before local training.

    Each vector is \nabla_{theta_shared} L_branch for one selected client at
    the identical round-start global checkpoint.  The weighted mean is the
    first-order FedAvg branch update; divergence measures how much client
    gradients cancel around that aggregated direction.
    """
    max_batches = max(1, int(getattr(args, "branch_shared_gradient_probe_batches", 1)))
    ce_gradients, kd_gradients, used_weights = [], [], []
    for idx, client_id in enumerate(nets_this_round.keys()):
        dataloader = dataloaders_this_round.get(client_id)
        if dataloader is None:
            continue
        model = nets_this_round[client_id]
        was_training = model.training
        model.eval()
        ce_grad = _average_branch_shared_gradient(model, dataloader, device, args, "ce", max_batches)
        kd_grad = _average_branch_shared_gradient(model, dataloader, device, args, "kd", max_batches)
        if was_training:
            model.train()
        if ce_grad is None and kd_grad is None:
            continue
        if ce_grad is not None:
            ce_gradients.append(ce_grad)
        else:
            ce_gradients.append(torch.zeros_like(kd_grad))
        if kd_grad is not None:
            kd_gradients.append(kd_grad)
        else:
            kd_gradients.append(torch.zeros_like(ce_grad))
        used_weights.append(float(fed_avg_freqs[idx]))

    ce_stats = _weighted_gradient_stats(ce_gradients, used_weights)
    kd_stats = _weighted_gradient_stats(kd_gradients, used_weights)
    if ce_stats is None or kd_stats is None:
        return {}
    metrics = {"branch_shared_gradient_probe_clients": len(used_weights)}
    for name, stats in (("ce", ce_stats), ("kd", kd_stats)):
        for key, value in stats.items():
            metrics[f"branch_shared_gradient_{name}_{key}"] = value
    return metrics


def _collect_reference_representations(model, dataloader, device, max_batches):
    """Collect fixed common-reference features without storing per-sample logits."""
    features = {name: [] for name in ("b1", "b2", "b3", "teacher")}
    targets = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            out = model(x)
            if not (isinstance(out, tuple) and len(out) == 8):
                continue
            _, _, _, _, final_fea, f1, f2, f3 = out
            for name, feature in zip(("b1", "b2", "b3", "teacher"), (f1, f2, f3, final_fea)):
                features[name].append(feature.detach().flatten(1).cpu())
            targets.append(target.detach().cpu().long())
    if was_training:
        model.train()
    if not targets:
        return None
    return {
        "targets": torch.cat(targets, dim=0),
        "features": {name: torch.cat(values, dim=0) for name, values in features.items() if values},
    }


def _fisher_feature_ratio(features, targets, eps=1e-12):
    """Trace(S_between)/Trace(S_within) on a common reference set."""
    classes = torch.unique(targets)
    if features.size(0) < 2 or classes.numel() < 2:
        return None
    overall_mean = features.mean(dim=0)
    between = features.new_zeros(())
    within = features.new_zeros(())
    for class_id in classes:
        selected = features[targets == class_id]
        if selected.numel() == 0:
            continue
        class_mean = selected.mean(dim=0)
        between += selected.size(0) * torch.sum((class_mean - overall_mean) ** 2)
        within += torch.sum((selected - class_mean) ** 2)
    return float((between / (within + eps)).item())


def _linear_cka(left, right, eps=1e-12):
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    cross_sq = torch.sum((left.T @ right) ** 2)
    left_sq = torch.sum((left.T @ left) ** 2)
    right_sq = torch.sum((right.T @ right) ** 2)
    return float((cross_sq / (torch.sqrt(left_sq * right_sq) + eps)).item())


def compute_post_aggregation_representation_change(pre, post, eps=1e-12):
    """Quantify how one FedAvg aggregation changes common-reference geometry."""
    if pre is None or post is None:
        return {}
    metrics = {}
    targets = pre["targets"]
    if not torch.equal(targets, post["targets"]):
        raise ValueError("Representation probe reference samples changed within a round.")
    for name in ("b1", "b2", "b3", "teacher"):
        before, after = pre["features"].get(name), post["features"].get(name)
        if before is None or after is None or before.shape != after.shape:
            continue
        cosine = F.cosine_similarity(before, after, dim=1).mean()
        relative_delta = torch.linalg.vector_norm(after - before) / (torch.linalg.vector_norm(before) + eps)
        pre_fisher = _fisher_feature_ratio(before, targets, eps)
        post_fisher = _fisher_feature_ratio(after, targets, eps)
        metrics[f"representation_{name}_cosine_pre_post"] = float(cosine.item())
        metrics[f"representation_{name}_relative_delta"] = float(relative_delta.item())
        metrics[f"representation_{name}_linear_cka"] = _linear_cka(before, after, eps)
        metrics[f"representation_{name}_pre_fisher"] = pre_fisher
        metrics[f"representation_{name}_post_fisher"] = post_fisher
        metrics[f"representation_{name}_fisher_delta"] = (
            None if pre_fisher is None or post_fisher is None else post_fisher - pre_fisher
        )
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
    parser.add_argument('--log_client_group_lambda', action='store_true',
                        help='Compatibility flag for lambda experiment scripts.')
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
    parser.add_argument('--in_channels', type=int, default=3)
    # image channels. 1 channel 데이터셋이도 3 channel로 하면 gray scale -> RGB scale 채널이 복사되서 작동합니다
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Classifier output dimension; dataset setup may override this value.')
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
                        choices=['blend', 'kd_only', 'feature_only'],
                        help='BYOT branch objective. blend uses (1-alpha)*CE + alpha*KD; '
                             'kd_only removes branch CE and uses alpha as an unrestricted KD coefficient; '
                             'feature_only removes both branch CE and KD while retaining feature imitation.')
    parser.add_argument('--byot_branch_ce_label_smoothing', type=float, default=0.0,
                        help='Label-smoothing epsilon applied only to branch CE; the final teacher CE remains hard-label CE.')
    parser.add_argument('--byot_branch_ce_weight', type=float, default=1.0,
                        help='Multiplicative coefficient applied only to branch CE; teacher CE is unchanged.')
    parser.add_argument('--byot_branch_kd_filter', default='none',
                        choices=['none', 'teacher_correct', 'teacher_correct_confident'],
                        help='Optional target-quality ablation for branch KD. teacher_correct keeps only samples '
                             'whose detached teacher prediction matches the label; teacher_correct_confident also '
                             'requires teacher confidence >= --byot_branch_kd_conf_threshold.')
    parser.add_argument('--byot_branch_kd_target_mode', default='full_teacher',
                        choices=['full_teacher', 'teacher_mass_uniform', 'teacher_mass_uniform_batchmean'],
                        help='Branch-KD target construction. full_teacher uses the complete teacher distribution; '
                             'teacher_mass_uniform preserves the teacher probability of the true label but spreads '
                             'all remaining probability uniformly over non-target classes; '
                             'teacher_mass_uniform_batchmean additionally replaces per-sample true-label mass '
                             'with its batch mean.')
    parser.add_argument('--byot_branch_kd_teacher_temperature', type=float, default=0.0,
                        help='Optional teacher temperature for branch KD. A non-positive value uses --temperature.')
    parser.add_argument('--byot_branch_kd_student_temperature', type=float, default=0.0,
                        help='Optional student temperature for branch KD. A non-positive value uses --temperature.')
    parser.add_argument('--byot_branch_kd_loss_scale_mode', default='native_t2',
                        choices=['native_t2', 'gradient_prefactor_one'],
                        help='Branch-KD KL multiplier: native_t2 uses T_student^2; '
                             'gradient_prefactor_one uses T_student to normalize the leading logit-gradient factor.')
    parser.add_argument('--byot_branch_kd_conf_threshold', type=float, default=0.8,
                        help='Confidence threshold for --byot_branch_kd_filter teacher_correct_confident. '
                             'Confidence is computed from the same temperature-scaled teacher distribution used by KD.')
    parser.add_argument('--byot_branch_gradient_mode', default='attached',
                        choices=['attached', 'detach_shared'],
                        help='Analysis-only intervention for BYOT branches. detach_shared trains branch-private '
                             'modules but blocks every branch-side loss from updating the shared trunk.')
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
                                 'teacher_label_prob', 'teacher_label_prob_entropy',
                                 'teacher_label_prob_branch_js', 'teacher_label_prob_entropy_branch_js',
                                 'teacher_correctness', 'branch_agreement', 'branch_soft_kl', 'branch_js'],
                        help='Sample-level reliability proxy used to weight BYOT KD loss.')
    parser.add_argument('--byot_client_proxy', default='none',
                        choices=['none', 'teacher_conf', 'teacher_entropy', 'teacher_margin',
                                 'teacher_label_prob', 'teacher_label_prob_entropy',
                                 'teacher_label_prob_branch_js', 'teacher_label_prob_entropy_branch_js',
                                 'teacher_correctness', 'branch_agreement', 'branch_soft_kl', 'branch_js'],
                        help='Client-level reliability proxy used to choose one BYOT KD alpha per client.')
    parser.add_argument('--byot_client_alpha_min', type=float, default=0.01,
                        help='Minimum client-wise BYOT alpha when --byot_client_proxy is enabled.')
    parser.add_argument('--byot_client_alpha_max', type=float, default=0.30,
                        help='Maximum client-wise BYOT alpha when --byot_client_proxy is enabled.')
    parser.add_argument('--byot_client_alpha_mode', default='map', choices=['map', 'multiply'],
                        help='map: use the client alpha directly; multiply: scale the round-wise lambda by it.')
    parser.add_argument('--byot_client_reliability_power', type=float, default=1.0,
                        help='Power applied to the client reliability proxy before mapping it to lambda.')
    parser.add_argument('--byot_proxy_temperature', type=float, default=1.0,
                        help='Temperature used only to measure BYOT reliability/skew proxies. '
                             'It is independent of branch-KD and MOON temperatures; 1.0 uses native logits.')
    parser.add_argument('--byot_client_skew_proxy', default='none',
                        choices=['none', 'prediction_entropy', 'prediction_mutual_info', 'prediction_js_global',
                                 'label_entropy', 'label_js_global', 'max_concentration'],
                        help='Client-distribution signal used to scale BYOT lambda.')
    parser.add_argument('--byot_log_prediction_entropy_components', action='store_true',
                        help='For prediction-entropy skew proxies, log marginal entropy b, mean sample entropy u, '
                             'and their difference d=b-u for every selected client.')
    parser.add_argument('--byot_client_skew_power', type=float, default=1.0,
                        help='Power p applied to the client skew reliability b before lambda scaling.')
    parser.add_argument('--byot_client_skew_min_scale', type=float, default=0.0,
                        help='Lower bound for the client skew lambda scale.')
    parser.add_argument('--byot_client_skew_max_scale', type=float, default=10.0,
                        help='Upper bound for the client skew lambda scale.')
    parser.add_argument('--byot_client_skew_correction_mode', default='multiply',
                        choices=['multiply', 'normalize', 'residual', 'soft_relax'],
                        help='How client skew reliability is converted to a lambda scale.')
    parser.add_argument('--byot_client_skew_norm_value', type=float, default=1.0,
                        help='Normalization constant for --byot_client_skew_correction_mode normalize.')
    parser.add_argument('--byot_client_skew_center', type=float, default=1.0,
                        help='Center value for --byot_client_skew_correction_mode residual.')
    parser.add_argument('--byot_client_skew_gamma', type=float, default=1.0,
                        help='Residual strength for --byot_client_skew_correction_mode residual.')
    parser.add_argument('--byot_client_skew_soft_tau', type=float, default=0.80,
                        help='Reliability threshold where soft_relax starts removing the skew penalty.')
    parser.add_argument('--byot_client_skew_soft_temperature', type=float, default=0.05,
                        help='Transition temperature for --byot_client_skew_correction_mode soft_relax.')
    parser.add_argument('--byot_client_skew_label_smoothing', type=float, default=0.0,
                        help='Additive class-count smoothing for label-based client skew proxies.')
    parser.add_argument('--byot_lambda_gate_mode', default='none', choices=['none', 'hard', 'soft'],
                        help='Optional gate between sample-wise and client-wise BYOT lambda.')
    parser.add_argument('--byot_lambda_gate_scope', default='client', choices=['client', 'round'],
                        help='Use an individual-client or round-level skew signal for the lambda gate.')
    parser.add_argument('--byot_lambda_gate_tau', type=float, default=0.75,
                        help='Skew-reliability threshold for the lambda gate.')
    parser.add_argument('--byot_lambda_gate_temperature', type=float, default=0.05,
                        help='Transition temperature for a soft lambda gate.')
    parser.add_argument('--byot_lambda_gate_warmup', type=int, default=0,
                        help='Optional rounds over which to ramp the lambda gate from zero.')
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
    parser.add_argument('--save_final_ckpt', action='store_true',
                        help='Save the final global-model checkpoint after the last communication round.')
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
    parser.add_argument('--log_layerwise_client_update_drift', action='store_true',
                        help='Also log pre-aggregation client update drift separately for trunk blocks, '
                             'branch-private modules, and the teacher head.')
    parser.add_argument('--drift_log_interval', type=int, default=1,
                        help='Log client drift every N rounds when --log_client_drift is enabled.')
    parser.add_argument('--log_gradient_probe', action='store_true',
                        help='Probe CE/KD/combined gradient dissimilarity before local training.')
    parser.add_argument('--gradient_probe_interval', type=int, default=50,
                        help='Probe gradient dissimilarity every N rounds when --log_gradient_probe is enabled.')
    parser.add_argument('--gradient_probe_batches', type=int, default=1,
                        help='Number of local batches per sampled client used for gradient probing.')
    parser.add_argument('--log_branch_gradient_alignment', action='store_true',
                        help='Probe cosine/norm-ratio between branch CE and teacher CE gradients on shared prefixes.')
    parser.add_argument('--branch_gradient_probe_interval', type=int, default=50,
                        help='Probe branch-to-teacher CE gradient alignment every N rounds when enabled.')
    parser.add_argument('--branch_gradient_probe_batches', type=int, default=1,
                        help='Number of local batches per client used for branch-to-teacher gradient probing.')
    parser.add_argument('--log_branch_shared_gradient_dispersion', action='store_true',
                        help='Log client dispersion of branch CE/KD gradients restricted to the shared trunk.')
    parser.add_argument('--branch_shared_gradient_probe_interval', type=int, default=50,
                        help='Round interval for --log_branch_shared_gradient_dispersion.')
    parser.add_argument('--branch_shared_gradient_probe_batches', type=int, default=1,
                        help='Local batches per selected client for shared-trunk branch-gradient dispersion.')
    parser.add_argument('--log_post_aggregation_representation', action='store_true',
                        help='Log common-reference representation change caused by each sampled FedAvg aggregation.')
    parser.add_argument('--representation_probe_interval', type=int, default=50,
                        help='Round interval for --log_post_aggregation_representation.')
    parser.add_argument('--representation_probe_batches', type=int, default=8,
                        help='Fixed test batches used before and after aggregation for representation geometry.')
    
    
    parser.add_argument('--min_threshold', type=float, default=0.8, help='시작 임계값 (Dynamic Threshold용)')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='전체 데이터를 학습할 워밍업 에폭 수')
    
    parser.add_argument('--partition_groups', type=int, default=8, help='Number of groups for noniid_grouping')
    parser.add_argument('--imbalance_factor', type=float, default=100.0, help='Imbalance factor for noniid_longtail')
    parser.add_argument('--cifar100_class_count', type=int, default=0,
                        help='For the controlled CIFAR-100 class-count study, use a deterministic nested subset '
                             'of this many classes (0 keeps all 100 classes).')
    parser.add_argument('--cifar100_subset_seed', type=int, default=0,
                        help='Seed defining the nested CIFAR-100 class subset used with --cifar100_class_count.')

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
    if args.dataset == 'tinyimagenet':
        args.num_classes = 200
    elif args.dataset == 'cifar100':
        requested_class_count = int(getattr(args, 'cifar100_class_count', 0))
        args.num_classes = requested_class_count if requested_class_count > 0 else 100
    elif args.dataset == 'emnist':
        args.num_classes = 47
        args.in_channels = 1  # EMNIST는 흑백
    elif args.dataset == 'cifar10':
        args.num_classes = 10
    # 다른 데이터셋 추가 시 여기에 작성
    
    print(f"🔥 [Auto Setup] Dataset: {args.dataset} | Classes: {args.num_classes} | Channels: {args.in_channels}")

    # Label-JS client-skew proxies compare each local label distribution with
    # this global training-distribution reference.  It is a server-side prior;
    # clients only need the resulting public vector to compute a local scalar.
    global_targets = _extract_dataset_targets(global_train_dataset)
    if global_targets is not None:
        if isinstance(global_targets, torch.Tensor):
            global_targets = global_targets.detach().cpu().numpy()
        global_targets = np.asarray(global_targets, dtype=np.int64).reshape(-1)
        global_counts = np.bincount(global_targets, minlength=args.num_classes).astype(np.float64)
        if float(global_counts.sum()) > 0.0:
            args.byot_client_skew_global_label_probs = (global_counts / global_counts.sum()).tolist()

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
        'args': _args_snapshot(args),
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
        'branch_gradient_probe_clients': [],
        'branch_gradient_b1_teacher_ce_cosine': [],
        'branch_gradient_b1_teacher_ce_norm_ratio': [],
        'branch_gradient_b1_teacher_kd_cosine': [],
        'branch_gradient_b1_teacher_kd_norm_ratio': [],
        'branch_gradient_b2_teacher_ce_cosine': [],
        'branch_gradient_b2_teacher_ce_norm_ratio': [],
        'branch_gradient_b2_teacher_kd_cosine': [],
        'branch_gradient_b2_teacher_kd_norm_ratio': [],
        'branch_gradient_b3_teacher_ce_cosine': [],
        'branch_gradient_b3_teacher_ce_norm_ratio': [],
        'branch_gradient_b3_teacher_kd_cosine': [],
        'branch_gradient_b3_teacher_kd_norm_ratio': [],
        'byot_effective_alpha_mean': [],
        'byot_effective_alpha_min': [],
        'byot_effective_alpha_max': [],
        # Optional prediction-entropy decomposition diagnostics.  Each item
        # stores the selected clients' raw b/u/d values for one round.
        'byot_prediction_entropy_client_stats': [],
        'byot_prediction_entropy_mean': [],
        'byot_sample_entropy_mean': [],
        'byot_prediction_mutual_info_mean': [],
        'byot_prediction_entropy_mutual_info_corr': [],
        'max': 0, 'avg_10': 0, 'avg_30': 0, 'avg_50': 0
    }
    for group in LAYERWISE_UPDATE_GROUPS:
        for stat_name in ("norm", "norm_sq", "mean_norm", "divergence", "relative_drift", "cosine"):
            pkl_dict[f"layer_update_{group}_{stat_name}"] = []
    for loss_name in ("ce", "kd"):
        for stat_name in ("divergence", "relative", "norm", "norm_sq", "mean_norm", "cosine"):
            pkl_dict[f"branch_shared_gradient_{loss_name}_{stat_name}"] = []
    pkl_dict["branch_shared_gradient_probe_clients"] = []
    for feature_name in ("b1", "b2", "b3", "teacher"):
        for stat_name in (
            "cosine_pre_post", "relative_delta", "linear_cka",
            "pre_fisher", "post_fisher", "fisher_delta",
        ):
            pkl_dict[f"representation_{feature_name}_{stat_name}"] = []
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

        branch_gradient_alignment_metrics = {}
        should_probe_branch_gradient = (
            getattr(args, 'log_branch_gradient_alignment', False)
            and getattr(args, 'branch_gradient_probe_interval', 1) > 0
            and round % getattr(args, 'branch_gradient_probe_interval', 1) == 0
        )
        if should_probe_branch_gradient:
            branch_gradient_alignment_metrics = compute_branch_teacher_gradient_alignment_probe(
                nets_this_round, dataloaders_this_round, fed_avg_freqs, device, args
            )
            if branch_gradient_alignment_metrics:
                logger.info(
                    "Branch gradient alignment: "
                    f"B1(CE/KD)={branch_gradient_alignment_metrics.get('branch_gradient_b1_teacher_ce_cosine', float('nan')):.4f}/"
                    f"{branch_gradient_alignment_metrics.get('branch_gradient_b1_teacher_kd_cosine', float('nan')):.4f}, "
                    f"B2(CE/KD)={branch_gradient_alignment_metrics.get('branch_gradient_b2_teacher_ce_cosine', float('nan')):.4f}/"
                    f"{branch_gradient_alignment_metrics.get('branch_gradient_b2_teacher_kd_cosine', float('nan')):.4f}, "
                    f"B3(CE/KD)={branch_gradient_alignment_metrics.get('branch_gradient_b3_teacher_ce_cosine', float('nan')):.4f}/"
                    f"{branch_gradient_alignment_metrics.get('branch_gradient_b3_teacher_kd_cosine', float('nan')):.4f}"
                )

        branch_shared_gradient_metrics = {}
        should_probe_branch_shared_gradient = (
            getattr(args, 'log_branch_shared_gradient_dispersion', False)
            and getattr(args, 'branch_shared_gradient_probe_interval', 1) > 0
            and round % getattr(args, 'branch_shared_gradient_probe_interval', 1) == 0
        )
        if should_probe_branch_shared_gradient:
            branch_shared_gradient_metrics = compute_branch_shared_gradient_dispersion_probe(
                nets_this_round, dataloaders_this_round, fed_avg_freqs, device, args
            )
            if branch_shared_gradient_metrics:
                logger.info(
                    "Branch shared-gradient dispersion: "
                    f"CE(div/rel/cos)={branch_shared_gradient_metrics['branch_shared_gradient_ce_divergence']:.6f}/"
                    f"{branch_shared_gradient_metrics['branch_shared_gradient_ce_relative']:.6f}/"
                    f"{branch_shared_gradient_metrics['branch_shared_gradient_ce_cosine']:.4f}, "
                    f"KD(div/rel/cos)={branch_shared_gradient_metrics['branch_shared_gradient_kd_divergence']:.6f}/"
                    f"{branch_shared_gradient_metrics['branch_shared_gradient_kd_relative']:.6f}/"
                    f"{branch_shared_gradient_metrics['branch_shared_gradient_kd_cosine']:.4f}"
                )

        representation_pre = None
        should_probe_representation = (
            getattr(args, 'log_post_aggregation_representation', False)
            and getattr(args, 'representation_probe_interval', 1) > 0
            and round % getattr(args, 'representation_probe_interval', 1) == 0
        )
        if should_probe_representation:
            representation_pre = _collect_reference_representations(
                global_model,
                global_test_dataloader,
                device,
                max(1, int(getattr(args, 'representation_probe_batches', 8))),
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
        client_proxy_stats = getattr(args, "_last_round_client_skew_proxy_stats", {})
        pkl_dict['byot_prediction_entropy_client_stats'].append(client_proxy_stats)
        if client_proxy_stats:
            b_values = np.asarray([
                stats['prediction_entropy'] for stats in client_proxy_stats.values()
                if 'prediction_entropy' in stats
            ], dtype=np.float64)
            u_values = np.asarray([
                stats['sample_entropy'] for stats in client_proxy_stats.values()
                if 'sample_entropy' in stats
            ], dtype=np.float64)
            d_values = np.asarray([
                stats['prediction_mutual_info'] for stats in client_proxy_stats.values()
                if 'prediction_mutual_info' in stats
            ], dtype=np.float64)
            pkl_dict['byot_prediction_entropy_mean'].append(
                float(b_values.mean()) if b_values.size else None
            )
            pkl_dict['byot_sample_entropy_mean'].append(
                float(u_values.mean()) if u_values.size else None
            )
            pkl_dict['byot_prediction_mutual_info_mean'].append(
                float(d_values.mean()) if d_values.size else None
            )
            if b_values.size >= 2 and b_values.size == d_values.size:
                pkl_dict['byot_prediction_entropy_mutual_info_corr'].append(
                    float(np.corrcoef(b_values, d_values)[0, 1])
                )
            else:
                pkl_dict['byot_prediction_entropy_mutual_info_corr'].append(None)
        else:
            pkl_dict['byot_prediction_entropy_mean'].append(None)
            pkl_dict['byot_sample_entropy_mean'].append(None)
            pkl_dict['byot_prediction_mutual_info_mean'].append(None)
            pkl_dict['byot_prediction_entropy_mutual_info_corr'].append(None)
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
            'branch_gradient_probe_clients',
            'branch_gradient_b1_teacher_ce_cosine',
            'branch_gradient_b1_teacher_ce_norm_ratio',
            'branch_gradient_b1_teacher_kd_cosine',
            'branch_gradient_b1_teacher_kd_norm_ratio',
            'branch_gradient_b2_teacher_ce_cosine',
            'branch_gradient_b2_teacher_ce_norm_ratio',
            'branch_gradient_b2_teacher_kd_cosine',
            'branch_gradient_b2_teacher_kd_norm_ratio',
            'branch_gradient_b3_teacher_ce_cosine',
            'branch_gradient_b3_teacher_ce_norm_ratio',
            'branch_gradient_b3_teacher_kd_cosine',
            'branch_gradient_b3_teacher_kd_norm_ratio',
        ]:
            pkl_dict[gradient_key].append(
                gradient_probe_metrics.get(gradient_key, branch_gradient_alignment_metrics.get(gradient_key))
            )
        for gradient_key in [
            'branch_shared_gradient_probe_clients',
            'branch_shared_gradient_ce_divergence',
            'branch_shared_gradient_ce_relative',
            'branch_shared_gradient_ce_norm',
            'branch_shared_gradient_ce_norm_sq',
            'branch_shared_gradient_ce_mean_norm',
            'branch_shared_gradient_ce_cosine',
            'branch_shared_gradient_kd_divergence',
            'branch_shared_gradient_kd_relative',
            'branch_shared_gradient_kd_norm',
            'branch_shared_gradient_kd_norm_sq',
            'branch_shared_gradient_kd_mean_norm',
            'branch_shared_gradient_kd_cosine',
        ]:
            pkl_dict[gradient_key].append(branch_shared_gradient_metrics.get(gradient_key))

        # 모델 집계 (Aggregation)
        drift_metrics = {}
        should_log_drift = (
            (
                getattr(args, 'log_client_drift', False)
                or getattr(args, 'log_layerwise_client_update_drift', False)
            )
            and getattr(args, 'drift_log_interval', 1) > 0
            and round % getattr(args, 'drift_log_interval', 1) == 0
        )
        if should_log_drift:
            drift_metrics = compute_client_update_drift(
                old_w,
                nets_this_round,
                fed_avg_freqs,
                layerwise=getattr(args, 'log_layerwise_client_update_drift', False),
            )

        for drift_key in [
            'client_update_norm',
            'client_update_norm_sq',
            'client_mean_update_norm',
            'client_update_divergence',
            'client_relative_drift',
            'client_update_cosine',
        ]:
            pkl_dict[drift_key].append(drift_metrics.get(drift_key))
        for group in LAYERWISE_UPDATE_GROUPS:
            for stat_name in ("norm", "norm_sq", "mean_norm", "divergence", "relative_drift", "cosine"):
                drift_key = f"layer_update_{group}_{stat_name}"
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

        representation_metrics = {}
        if should_probe_representation:
            representation_post = _collect_reference_representations(
                global_model,
                global_test_dataloader,
                device,
                max(1, int(getattr(args, 'representation_probe_batches', 8))),
            )
            representation_metrics = compute_post_aggregation_representation_change(
                representation_pre, representation_post
            )
            if representation_metrics:
                logger.info(
                    "Post-aggregation representation: "
                    f"teacher(cos/delta/fisher)="
                    f"{representation_metrics.get('representation_teacher_cosine_pre_post', float('nan')):.6f}/"
                    f"{representation_metrics.get('representation_teacher_relative_delta', float('nan')):.6f}/"
                    f"{representation_metrics.get('representation_teacher_fisher_delta', float('nan')):.6f}"
                )
        for feature_name in ("b1", "b2", "b3", "teacher"):
            for stat_name in (
                "cosine_pre_post", "relative_delta", "linear_cka",
                "pre_fisher", "post_fisher", "fisher_delta",
            ):
                key = f"representation_{feature_name}_{stat_name}"
                pkl_dict[key].append(representation_metrics.get(key))

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
                    "args": _args_snapshot(args),
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
    if getattr(args, "save_final_ckpt", False):
        ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else args.logdir
        final_ckpt_path = os.path.join(ckpt_dir, f"{log_file_name}_final.pt")
        os.makedirs(os.path.dirname(final_ckpt_path), exist_ok=True)
        final_ckpt = {
            "round": max(int(args.round) - 1, 0),
            "final_accuracy": float(test_acc_global) if args.round > 0 else None,
            "args": _args_snapshot(args),
            "global_model": global_model.state_dict(),
        }
        torch.save(final_ckpt, final_ckpt_path)
        logger.info(f"Saved final checkpoint: {final_ckpt_path}")

    # Serialize to a temporary file first.  A failed pickle must not leave a
    # zero-byte file that a resumable launcher mistakes for a completed run.
    result_path = os.path.join(args.logdir, log_file_name + '.pkl')
    temp_result_path = f"{result_path}.tmp.{os.getpid()}"
    try:
        with open(temp_result_path, 'wb') as f:
            pickle.dump(pkl_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_result_path, result_path)
    finally:
        if os.path.exists(temp_result_path):
            os.remove(temp_result_path)
        
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    main()
