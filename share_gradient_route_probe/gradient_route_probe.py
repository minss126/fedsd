"""Gradient-route diagnostic for federated auxiliary CE and KD.

At one unchanged model checkpoint this module measures the sample-mean
gradient of the final CE and each auxiliary branch CE/KD objective on local
client data and on a fixed global reference set. It never calls an optimizer
and never writes ``parameter.grad``.
"""

from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


Tensor = torch.Tensor
Batch = tuple[Tensor, Tensor]
Gradient = Dict[str, Tensor]
GradientRoutes = Dict[str, Gradient]
ForwardAdapter = Callable[[torch.nn.Module, Tensor], tuple[Tensor, Sequence[Tensor]]]


DEFAULT_RESNET_GROUPS: Mapping[str, tuple[str, ...]] = {
    "Stage1": ("conv1", "bn1", "layer1"),
    "Stage2": ("layer2",),
    "Stage3": ("layer3",),
}

DEFAULT_RESNET_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "B1": ("conv1", "bn1", "layer1"),
    "B2": ("conv1", "bn1", "layer1", "layer2"),
    "B3": ("conv1", "bn1", "layer1", "layer2", "layer3"),
    "All": ("conv1", "bn1", "layer1", "layer2", "layer3"),
}


@dataclass(frozen=True)
class ProbeConfig:
    """Loss conventions for the diagnostic.

    Branch indices are zero-based positions returned by the adapter.
    Individual branch routes are unit gradients. Reduction only controls
    whether the derived ``*_all`` route is their sum or mean.
    """

    branch_indices: tuple[int, ...] = (0, 1, 2)
    branch_reduction: str = "sum"
    branch_ce_label_smoothing: float = 0.0
    teacher_temperature: float = 1.0
    student_temperature: float = 1.0
    kd_scale_mode: str = "student_t_squared"
    eps: float = 1.0e-12

    def validate(self) -> None:
        if self.branch_reduction not in {"sum", "mean"}:
            raise ValueError("branch_reduction must be 'sum' or 'mean'")
        if self.teacher_temperature <= 0 or self.student_temperature <= 0:
            raise ValueError("KD temperatures must be positive")
        if self.kd_scale_mode not in {"student_t_squared", "student_t", "none"}:
            raise ValueError("Unsupported kd_scale_mode")
        if not self.branch_indices:
            raise ValueError("At least one branch is required")
        if len(set(self.branch_indices)) != len(self.branch_indices):
            raise ValueError("branch_indices contains duplicates")
        if any(index < 0 for index in self.branch_indices):
            raise ValueError("branch_indices must be non-negative")


def byot_tuple_adapter(
    model: torch.nn.Module, inputs: Tensor
) -> tuple[Tensor, Sequence[Tensor]]:
    """Adapt ``(final, b1, b2, b3, ...)`` BYOT model output."""

    output = model(inputs)
    if not isinstance(output, (tuple, list)) or len(output) < 4:
        raise TypeError(
            "Expected model(x) -> (final_logits, b1_logits, b2_logits, "
            "b3_logits, ...)."
        )
    return output[0], tuple(output[1:4])


@contextmanager
def preserve_rng():
    """Prevent a diagnostic pass from perturbing the training RNG stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def cache_loader(loader: Iterable[Batch], max_samples: int | None = None) -> list[Batch]:
    """Cache a loader on CPU; retained for standalone integrations/tests."""

    if max_samples is not None and max_samples <= 0:
        max_samples = None
    cached: list[Batch] = []
    seen = 0
    with preserve_rng():
        for inputs, targets in loader:
            if max_samples is not None and seen >= max_samples:
                break
            if max_samples is not None and seen + int(targets.numel()) > max_samples:
                keep = max_samples - seen
                inputs, targets = inputs[:keep], targets[:keep]
            cached.append((inputs.detach().cpu(), targets.detach().cpu().long()))
            seen += int(targets.numel())
    if not cached:
        raise ValueError("The diagnostic loader produced no samples")
    return cached


def _belongs_to(name: str, roots: Iterable[str]) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


def shared_named_parameters(
    model: torch.nn.Module, shared_roots: Iterable[str]
) -> list[tuple[str, torch.nn.Parameter]]:
    roots = tuple(shared_roots)
    result = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and _belongs_to(name, roots)
    ]
    if not result:
        raise ValueError(f"No trainable parameters matched shared_roots={roots}")
    return result


def _zero_gradient(named_parameters: Sequence[tuple[str, Tensor]]) -> Gradient:
    return {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in named_parameters
    }


def _accumulate_autograd(
    destination: MutableMapping[str, Tensor],
    named_parameters: Sequence[tuple[str, Tensor]],
    gradients: Sequence[Tensor | None],
    weight: float,
) -> None:
    for (name, parameter), gradient in zip(named_parameters, gradients):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        destination[name].add_(value.detach().cpu(), alpha=float(weight))


def _scale_route_(route: MutableMapping[str, Tensor], scale: float) -> None:
    for value in route.values():
        value.mul_(float(scale))


def _combine_routes(
    routes: Sequence[Mapping[str, Tensor]], scale: float = 1.0
) -> Gradient:
    if not routes:
        raise ValueError("Cannot combine an empty route list")
    combined = {name: torch.zeros_like(value) for name, value in routes[0].items()}
    for route in routes:
        if route.keys() != combined.keys():
            raise ValueError("Gradient routes use different parameter-name sets")
        for name, value in route.items():
            combined[name].add_(value)
    _scale_route_(combined, scale)
    return combined


def _kd_scale(config: ProbeConfig) -> float:
    if config.kd_scale_mode == "student_t_squared":
        return config.student_temperature**2
    if config.kd_scale_mode == "student_t":
        return config.student_temperature
    return 1.0


def _branch_labels(config: ProbeConfig) -> tuple[str, ...]:
    return tuple(f"b{index + 1}" for index in config.branch_indices)


def measure_gradient_routes(
    model: torch.nn.Module,
    batches: Iterable[Batch],
    device: torch.device | str,
    forward_adapter: ForwardAdapter,
    shared_roots: Iterable[str],
    config: ProbeConfig = ProbeConfig(),
) -> GradientRoutes:
    """Measure final CE and independent B1/B2/B3 CE/KD gradients.

    Batch gradients are weighted by sample count, making every returned route
    the gradient of the sample-mean loss over the complete supplied dataset.
    The All route is derived from independent routes, so it adds no backward.
    """

    config.validate()
    device = torch.device(device)
    labels = _branch_labels(config)
    named_parameters = shared_named_parameters(model, shared_roots)
    parameters = [parameter for _, parameter in named_parameters]
    route_names = ["main_ce"]
    route_names += [f"aux_ce_{label}" for label in labels]
    route_names += [f"aux_kd_{label}" for label in labels]
    routes = {name: _zero_gradient(named_parameters) for name in route_names}
    module_modes = {module: module.training for module in model.modules()}
    total_samples = 0

    with preserve_rng():
        model.eval()
        try:
            for inputs_cpu, targets_cpu in batches:
                inputs = inputs_cpu.to(device, non_blocking=True)
                targets = targets_cpu.to(device, non_blocking=True).long()
                batch_samples = int(targets.numel())
                if batch_samples == 0:
                    continue
                final_logits, branch_logits_all = forward_adapter(model, inputs)
                try:
                    branches = [branch_logits_all[index] for index in config.branch_indices]
                except IndexError as error:
                    raise ValueError("forward_adapter returned too few branches") from error

                losses: list[tuple[str, Tensor]] = [
                    ("main_ce", F.cross_entropy(final_logits, targets, reduction="mean"))
                ]
                losses.extend(
                    (
                        f"aux_ce_{label}",
                        F.cross_entropy(
                            logits,
                            targets,
                            reduction="mean",
                            label_smoothing=config.branch_ce_label_smoothing,
                        ),
                    )
                    for label, logits in zip(labels, branches)
                )
                teacher_probability = F.softmax(
                    final_logits.detach() / config.teacher_temperature, dim=1
                )
                losses.extend(
                    (
                        f"aux_kd_{label}",
                        F.kl_div(
                            F.log_softmax(logits / config.student_temperature, dim=1),
                            teacher_probability,
                            reduction="batchmean",
                        )
                        * _kd_scale(config),
                    )
                    for label, logits in zip(labels, branches)
                )

                for loss_index, (route_name, loss) in enumerate(losses):
                    gradients = torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=loss_index + 1 < len(losses),
                        allow_unused=True,
                    )
                    _accumulate_autograd(
                        routes[route_name], named_parameters, gradients, batch_samples
                    )
                total_samples += batch_samples
        finally:
            for module, was_training in module_modes.items():
                module.training = was_training

    if total_samples <= 0:
        raise ValueError("The diagnostic batches contained no samples")
    for route in routes.values():
        _scale_route_(route, 1.0 / total_samples)
    reduction_scale = 1.0 if config.branch_reduction == "sum" else 1.0 / len(labels)
    routes["aux_ce_all"] = _combine_routes(
        [routes[f"aux_ce_{label}"] for label in labels], reduction_scale
    )
    routes["aux_kd_all"] = _combine_routes(
        [routes[f"aux_kd_{label}"] for label in labels], reduction_scale
    )
    return routes


def gradient_stats(
    route: Mapping[str, Tensor],
    reference: Mapping[str, Tensor],
    roots: Iterable[str],
    eps: float = 1.0e-12,
) -> dict[str, float | int | bool | None]:
    """Return every inexpensive scalar derivable from two gradients."""

    dot = route_sq = reference_sq = 0.0
    parameter_count = 0
    roots = tuple(roots)
    if route.keys() != reference.keys():
        raise ValueError("Gradient routes use different parameter-name sets")
    for name, route_value in route.items():
        if not _belongs_to(name, roots):
            continue
        left = route_value.float().reshape(-1)
        right = reference[name].float().reshape(-1)
        dot += float(torch.dot(left, right).item())
        route_sq += float(torch.dot(left, left).item())
        reference_sq += float(torch.dot(right, right).item())
        parameter_count += int(left.numel())
    route_norm = math.sqrt(max(route_sq, 0.0))
    reference_norm = math.sqrt(max(reference_sq, 0.0))
    cosine = None
    if route_norm > eps and reference_norm > eps:
        cosine = max(-1.0, min(1.0, dot / (route_norm * reference_norm)))
    angle = None if cosine is None else math.degrees(math.acos(cosine))
    return {
        "parameter_count": parameter_count,
        "dot": dot,
        "route_norm": route_norm,
        "reference_norm": reference_norm,
        "cosine": cosine,
        "angle_degrees": angle,
        "negative_cosine": None if cosine is None else bool(cosine < 0.0),
        "norm_ratio": None if reference_norm <= eps else route_norm / reference_norm,
        "signed_projection_on_reference_unit": (
            None if reference_norm <= eps else dot / reference_norm
        ),
        "reference_normalized_projection": (
            None if reference_sq <= eps else dot / reference_sq
        ),
    }


def planned_comparisons(
    local_routes: GradientRoutes,
    global_routes: GradientRoutes,
    branch_prefixes: Mapping[str, Iterable[str]] = DEFAULT_RESNET_PREFIXES,
    eps: float = 1.0e-12,
) -> dict[str, dict[str, dict[str, float | int | bool | None]]]:
    """Return the requested analyses 1--3 for B1/B2/B3/All."""

    output = {}
    for branch_name, roots in branch_prefixes.items():
        suffix = branch_name.lower()
        output[branch_name] = {
            "global_aux_ce_to_main_ce": gradient_stats(
                global_routes[f"aux_ce_{suffix}"], global_routes["main_ce"], roots, eps
            ),
            "global_aux_kd_to_main_ce": gradient_stats(
                global_routes[f"aux_kd_{suffix}"], global_routes["main_ce"], roots, eps
            ),
            "local_aux_ce_to_main_ce": gradient_stats(
                local_routes[f"aux_ce_{suffix}"], local_routes["main_ce"], roots, eps
            ),
            "local_aux_kd_to_main_ce": gradient_stats(
                local_routes[f"aux_kd_{suffix}"], local_routes["main_ce"], roots, eps
            ),
            "local_to_global_aux_ce": gradient_stats(
                local_routes[f"aux_ce_{suffix}"], global_routes[f"aux_ce_{suffix}"], roots, eps
            ),
            "local_to_global_aux_kd": gradient_stats(
                local_routes[f"aux_kd_{suffix}"], global_routes[f"aux_kd_{suffix}"], roots, eps
            ),
        }
    return output


def pairwise_stats_matrix(
    local_routes: GradientRoutes,
    global_routes: GradientRoutes,
    parameter_groups: Mapping[str, Iterable[str]],
    eps: float = 1.0e-12,
) -> dict:
    """Return full pairwise stats, not only cosine, for later analyses."""

    routes = {f"local/{name}": value for name, value in local_routes.items()}
    routes.update({f"global/{name}": value for name, value in global_routes.items()})
    names = tuple(routes)
    return {
        group_name: {
            left: {
                right: gradient_stats(routes[left], routes[right], roots, eps)
                for right in names
            }
            for left in names
        }
        for group_name, roots in parameter_groups.items()
    }


def route_norms(
    routes: GradientRoutes,
    parameter_groups: Mapping[str, Iterable[str]],
    eps: float = 1.0e-12,
) -> dict:
    """Store absolute norms even when a comparison is not primary."""

    return {
        group_name: {
            route_name: gradient_stats(route, route, roots, eps)["route_norm"]
            for route_name, route in routes.items()
        }
        for group_name, roots in parameter_groups.items()
    }


def _empty_like_routes(routes: GradientRoutes) -> GradientRoutes:
    return {
        route_name: {name: torch.zeros_like(value) for name, value in route.items()}
        for route_name, route in routes.items()
    }


def _add_weighted_routes_(
    destination: GradientRoutes, source: GradientRoutes, weight: float
) -> None:
    for route_name, route in source.items():
        for name, value in route.items():
            destination[route_name][name].add_(value, alpha=float(weight))


def _weighted_scalar_summary(
    records: Sequence[tuple[float, float]],
) -> dict[str, float | int | None]:
    """Summarize client scalars with FedAvg-weighted and ordinary stats."""

    finite = [
        (float(weight), float(value))
        for weight, value in records
        if math.isfinite(float(value))
    ]
    if not finite:
        return {"count": 0}
    values = np.asarray([value for _, value in finite], dtype=np.float64)
    weights = np.asarray([max(weight, 0.0) for weight, _ in finite], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    weighted_mean = float(np.sum(weights * values))
    weighted_std = float(np.sqrt(np.sum(weights * (values - weighted_mean) ** 2)))
    ordinary_mean = float(values.mean())
    sample_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ci_half = 1.96 * sample_std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "count": int(len(values)),
        "fedavg_weighted_mean": weighted_mean,
        "fedavg_weighted_population_std": weighted_std,
        "unweighted_mean": ordinary_mean,
        "unweighted_sample_std": sample_std,
        "unweighted_normal_ci95_low": ordinary_mean - ci_half,
        "unweighted_normal_ci95_high": ordinary_mean + ci_half,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _summarize_client_comparisons(
    per_client: Mapping[str, dict], normalized_weights: Mapping[str, float]
) -> dict:
    if not per_client:
        return {}
    summary = {}
    first = next(iter(per_client.values()))
    for branch_name, pair_map in first.items():
        summary[branch_name] = {}
        for pair_name, metric_map in pair_map.items():
            summary[branch_name][pair_name] = {}
            for metric_name in metric_map:
                records = []
                for client_id, comparisons in per_client.items():
                    value = comparisons[branch_name][pair_name][metric_name]
                    if value is None or isinstance(value, bool):
                        continue
                    records.append((normalized_weights[client_id], float(value)))
                summary[branch_name][pair_name][metric_name] = _weighted_scalar_summary(records)
            cos_records = []
            for client_id, comparisons in per_client.items():
                cosine = comparisons[branch_name][pair_name]["cosine"]
                if cosine is not None:
                    cos_records.append(
                        (normalized_weights[client_id], float(cosine < 0.0))
                    )
            summary[branch_name][pair_name]["negative_cosine_rate"] = (
                _weighted_scalar_summary(cos_records)
            )
    return summary


def run_round_probe(
    model_at_checkpoint: torch.nn.Module,
    selected_client_batches: Mapping[Hashable, Iterable[Batch]],
    fedavg_client_weights: Mapping[Hashable, float],
    global_reference_batches: Iterable[Batch],
    device: torch.device | str,
    forward_adapter: ForwardAdapter,
    shared_roots: Iterable[str],
    branch_prefixes: Mapping[str, Iterable[str]] = DEFAULT_RESNET_PREFIXES,
    incremental_groups: Mapping[str, Iterable[str]] = DEFAULT_RESNET_GROUPS,
    config: ProbeConfig = ProbeConfig(),
) -> dict:
    """Measure global, per-client, and FedAvg-aggregated local routes."""

    if set(selected_client_batches) != set(fedavg_client_weights):
        raise ValueError("client batches and weights must have identical keys")
    weight_sum = sum(float(value) for value in fedavg_client_weights.values())
    if weight_sum <= 0:
        raise ValueError("FedAvg client weights must have a positive sum")
    normalized_weights = {
        client_id: float(weight) / weight_sum
        for client_id, weight in fedavg_client_weights.items()
    }

    global_routes = measure_gradient_routes(
        model_at_checkpoint,
        global_reference_batches,
        device,
        forward_adapter,
        shared_roots,
        config,
    )
    aggregate_routes = _empty_like_routes(global_routes)
    per_client_comparisons = {}
    for client_id, batches in selected_client_batches.items():
        local_routes = measure_gradient_routes(
            model_at_checkpoint,
            batches,
            device,
            forward_adapter,
            shared_roots,
            config,
        )
        _add_weighted_routes_(aggregate_routes, local_routes, normalized_weights[client_id])
        per_client_comparisons[str(client_id)] = planned_comparisons(
            local_routes, global_routes, branch_prefixes, config.eps
        )
        del local_routes

    string_weights = {str(key): value for key, value in normalized_weights.items()}
    all_groups = dict(incremental_groups)
    all_groups.update({f"Prefix/{name}": roots for name, roots in branch_prefixes.items()})
    report = {
        "definition": (
            "All gradients are sample-mean gradients at one unchanged model checkpoint; "
            "local routes are aggregated with the supplied FedAvg weights"
        ),
        "config": asdict(config),
        "selected_clients": [str(client_id) for client_id in selected_client_batches],
        "normalized_client_weights": string_weights,
        "aggregate_comparisons": planned_comparisons(
            aggregate_routes, global_routes, branch_prefixes, config.eps
        ),
        "per_client_comparisons": per_client_comparisons,
        "per_client_summary": _summarize_client_comparisons(
            per_client_comparisons, string_weights
        ),
        "aggregate_route_norms": {
            "local": route_norms(aggregate_routes, all_groups, config.eps),
            "global": route_norms(global_routes, all_groups, config.eps),
        },
        "aggregate_pairwise_stats": pairwise_stats_matrix(
            aggregate_routes, global_routes, all_groups, config.eps
        ),
    }
    del aggregate_routes, global_routes
    return report


def save_report(report: Mapping, output_path: str | Path) -> None:
    """Atomically save scalar output; raw gradient tensors are not serialized."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
