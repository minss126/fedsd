"""Trajectory-safe diagnostics for aggregation-induced functionality damage.

The analysis is deliberately evaluation-only.  It runs after local training,
uses a deterministic class-balanced subset of the official test set, and
temporarily replaces complete ResNet stages in-place before restoring their
exact local state.  No full-model deepcopy is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# The BYOT branches are attached immediately after these backbone stages.
RESNET18_BRANCH_STAGES = {
    "B1": "layer1.",
    "B2": "layer2.",
    "B3": "layer3.",
}


MODEL_LEVEL_FIELDS = (
    "method",
    "partition",
    "local_epoch",
    "seed",
    "round",
    "reference_samples",
    "reference_per_class",
    "selected_clients",
    "aggregation_weight_sum",
    "weighted_local_loss",
    "weighted_local_acc",
    "aggregated_loss",
    "aggregated_acc",
    "aggregation_loss_gap",
    "aggregation_acc_gap",
)

BLOCK_RAW_FIELDS = (
    "method",
    "partition",
    "local_epoch",
    "seed",
    "round",
    "client_id",
    "aggregation_weight",
    "replacement_source",
    "stage",
    "local_loss",
    "local_acc",
    "hybrid_loss",
    "hybrid_acc",
    "replacement_loss_damage",
    "replacement_acc_damage",
)

BLOCK_SUMMARY_FIELDS = (
    "method",
    "partition",
    "local_epoch",
    "seed",
    "round",
    "replacement_source",
    "stage",
    "client_count",
    "replacement_loss_damage_mean",
    "replacement_loss_damage_std",
    "replacement_loss_damage_weighted_mean",
    "replacement_loss_damage_weighted_std",
    "replacement_acc_damage_mean",
    "replacement_acc_damage_std",
    "replacement_acc_damage_weighted_mean",
    "replacement_acc_damage_weighted_std",
)


def parse_completed_rounds(spec: str, total_rounds: int) -> set[int]:
    """Parse human-facing, one-based completed-round indices."""
    rounds: set[int] = set()
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(
                "Aggregation-damage rounds are one-based completed-round "
                f"indices and must be >= 1; received {value}."
            )
        if value <= int(total_rounds):
            rounds.add(value)
    if not rounds:
        raise ValueError(
            "No aggregation-damage analysis round falls within the configured "
            f"training horizon of {total_rounds} rounds."
        )
    return rounds


def _dataset_targets(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent = _dataset_targets(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]
    for name in ("targets", "target", "labels"):
        if hasattr(dataset, name):
            values = getattr(dataset, name)
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            return np.asarray(values, dtype=np.int64).reshape(-1)
    raise ValueError(
        "The aggregation-damage reference dataset does not expose targets, "
        "target, or labels."
    )


def build_class_balanced_reference_loader(
    dataset,
    num_classes: int,
    per_class: int,
    batch_size: int,
    reference_seed: int,
    pin_memory: bool,
):
    """Build a deterministic balanced subset without touching training RNGs."""
    targets = _dataset_targets(dataset)
    class_indices = []
    available_counts = []
    for class_id in range(int(num_classes)):
        indices = np.flatnonzero(targets == class_id).astype(np.int64)
        if indices.size == 0:
            raise ValueError(
                f"Reference dataset has no samples for class {class_id}."
            )
        class_indices.append(indices)
        available_counts.append(int(indices.size))

    selected_per_class = (
        min(available_counts) if int(per_class) <= 0 else int(per_class)
    )
    if selected_per_class > min(available_counts):
        raise ValueError(
            f"Requested {selected_per_class} reference samples per class, but "
            f"the smallest class has {min(available_counts)}."
        )

    # A private Generator guarantees that reference construction cannot alter
    # NumPy's global RNG used for client selection.
    rng = np.random.default_rng(int(reference_seed))
    selected = []
    class_counts = {}
    for class_id, indices in enumerate(class_indices):
        chosen = rng.permutation(indices)[:selected_per_class]
        selected.extend(int(index) for index in chosen)
        class_counts[str(class_id)] = int(len(chosen))

    selected_array = np.asarray(selected, dtype=np.int64)
    reference_dataset = Subset(dataset, selected_array.tolist())
    loader = DataLoader(
        reference_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
    )
    manifest = {
        "source": "official_test_set",
        "training_samples_used": False,
        "class_balanced": True,
        "num_classes": int(num_classes),
        "samples_per_class": int(selected_per_class),
        "total_samples": int(len(selected_array)),
        "reference_seed": int(reference_seed),
        "class_counts": class_counts,
        "indices_sha256": hashlib.sha256(selected_array.tobytes()).hexdigest(),
    }
    return loader, manifest


def _main_logits(model, inputs: torch.Tensor) -> torch.Tensor:
    # BYOT-shaped ResNets can bypass all auxiliary branches.  This evaluates
    # exactly the final/main classifier requested by the experiment and avoids
    # changing private-branch BN behavior during diagnostics.
    if hasattr(model, "forward_teacher"):
        output = model.forward_teacher(inputs)
        if isinstance(output, tuple):
            return output[-1]
        return output

    output = model(inputs)
    if isinstance(output, tuple):
        if len(output) == 8:
            return output[0]
        if len(output) >= 2:
            return output[1]
        return output[0]
    if isinstance(output, list):
        return output[-1]
    return output


def evaluate_main_classifier(model, loader, device: torch.device) -> dict:
    was_training = bool(model.training)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()
                logits = _main_logits(model, inputs)
                if logits.dim() > 2:
                    logits = logits.flatten(1)
                total_loss += float(
                    F.cross_entropy(logits, targets, reduction="sum").item()
                )
                total_correct += int(
                    logits.argmax(dim=1).eq(targets).sum().item()
                )
                total_samples += int(targets.numel())
    finally:
        model.train(was_training)
    if total_samples == 0:
        raise ValueError("Aggregation-damage reference loader is empty.")
    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples,
        "samples": total_samples,
    }


def _model_device(model) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _stage_keys(state: Mapping[str, torch.Tensor], prefix: str) -> list[str]:
    keys = [key for key in state if key.startswith(prefix)]
    if not keys:
        raise ValueError(f"Model state has no entries under stage prefix {prefix!r}.")
    return keys


def _copy_state_entries(
    target_state: MutableMapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    keys: Iterable[str],
) -> None:
    with torch.no_grad():
        for key in keys:
            if key not in source_state:
                raise KeyError(f"Replacement source is missing state entry {key!r}.")
            target = target_state[key]
            source = source_state[key]
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def _evaluate_with_stage_replacement(
    model,
    replacement_state: Mapping[str, torch.Tensor],
    stage_prefix: str,
    loader,
    device: torch.device,
) -> dict:
    """Replace one complete stage, evaluate, and restore exact local state."""
    state = model.state_dict()
    keys = _stage_keys(state, stage_prefix)
    # Only one stage is retained at a time, on CPU, to bound GPU memory.
    local_stage = {
        key: state[key].detach().cpu().clone()
        for key in keys
    }
    try:
        _copy_state_entries(state, replacement_state, keys)
        return evaluate_main_classifier(model, loader, device)
    finally:
        _copy_state_entries(state, local_stage, keys)


def _append_csv(path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    if not rows:
        return
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float("nan"), float("nan")
    normalized = weights / weight_sum
    mean = float(np.sum(normalized * values))
    variance = float(np.sum(normalized * (values - mean) ** 2))
    return mean, math.sqrt(max(variance, 0.0))


def _summarize_replacement_rows(rows: Sequence[dict]) -> list[dict]:
    groups: Dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["replacement_source"], row["stage"])
        groups.setdefault(key, []).append(row)

    summaries = []
    for (source, stage), group in groups.items():
        losses = np.asarray(
            [float(row["replacement_loss_damage"]) for row in group],
            dtype=np.float64,
        )
        accuracies = np.asarray(
            [float(row["replacement_acc_damage"]) for row in group],
            dtype=np.float64,
        )
        weights = np.asarray(
            [float(row["aggregation_weight"]) for row in group],
            dtype=np.float64,
        )
        weighted_loss_mean, weighted_loss_std = _weighted_stats(losses, weights)
        weighted_acc_mean, weighted_acc_std = _weighted_stats(accuracies, weights)
        first = group[0]
        summaries.append({
            "method": first["method"],
            "partition": first["partition"],
            "local_epoch": first["local_epoch"],
            "seed": first["seed"],
            "round": first["round"],
            "replacement_source": source,
            "stage": stage,
            "client_count": len(group),
            "replacement_loss_damage_mean": float(losses.mean()),
            "replacement_loss_damage_std": float(losses.std(ddof=0)),
            "replacement_loss_damage_weighted_mean": weighted_loss_mean,
            "replacement_loss_damage_weighted_std": weighted_loss_std,
            "replacement_acc_damage_mean": float(accuracies.mean()),
            "replacement_acc_damage_std": float(accuracies.std(ddof=0)),
            "replacement_acc_damage_weighted_mean": weighted_acc_mean,
            "replacement_acc_damage_weighted_std": weighted_acc_std,
        })
    return sorted(
        summaries,
        key=lambda row: (
            row["replacement_source"],
            list(RESNET18_BRANCH_STAGES).index(row["stage"]),
        ),
    )


@dataclass
class PostLocalAnalysisContext:
    completed_round: int
    local_metrics: Dict[int, dict]
    weight_by_client: Dict[int, float]
    pre_round_rows: list[dict]


class AggregationDamageAnalyzer:
    """Sparse-round aggregation damage analysis and CSV writer."""

    def __init__(
        self,
        *,
        args,
        log_file_name: str,
        test_dataset,
        device: torch.device,
        logger,
    ) -> None:
        self.args = args
        self.device = device
        self.logger = logger
        self.analysis_rounds = parse_completed_rounds(
            getattr(args, "aggregation_damage_analysis_rounds", ""),
            int(args.round),
        )
        self.method = str(
            getattr(args, "aggregation_damage_method_label", "")
            or args.alg
        )
        self.partition = str(
            getattr(args, "aggregation_damage_partition_label", "")
            or ("iid" if args.partition == "iid" else f"beta_{args.beta}")
        )
        self.local_epoch = int(args.epochs)
        self.seed = int(args.seed)
        output_dir = str(
            getattr(args, "aggregation_damage_output_dir", "") or ""
        ).strip()
        if not output_dir:
            output_dir = os.path.join(
                args.logdir, f"{log_file_name}_aggregation_damage"
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_level_path = os.path.join(self.output_dir, "model_level.csv")
        self.block_raw_path = os.path.join(
            self.output_dir, "block_replacement_raw.csv"
        )
        self.block_summary_path = os.path.join(
            self.output_dir, "block_replacement_summary.csv"
        )
        if bool(getattr(args, "aggregation_damage_overwrite", False)):
            for path in (
                self.model_level_path,
                self.block_raw_path,
                self.block_summary_path,
            ):
                with open(path, "w", encoding="utf-8"):
                    pass

        self.reference_loader, self.reference_manifest = (
            build_class_balanced_reference_loader(
                test_dataset,
                num_classes=int(args.num_classes),
                per_class=int(
                    getattr(args, "aggregation_damage_reference_per_class", 10)
                ),
                batch_size=int(
                    getattr(
                        args,
                        "aggregation_damage_reference_batch_size",
                        args.test_batch_size,
                    )
                ),
                reference_seed=int(
                    getattr(args, "aggregation_damage_reference_seed", 1729)
                ),
                pin_memory=device.type == "cuda",
            )
        )
        manifest = {
            **self.reference_manifest,
            "analysis_rounds_completed": sorted(self.analysis_rounds),
            "round_indexing": "one_based_number_of_completed_aggregations",
            "stage_mapping": dict(RESNET18_BRANCH_STAGES),
            "replacement_state": (
                "complete stage state_dict entries, including convolution, "
                "BN affine parameters, and BN running buffers"
            ),
            "method": self.method,
            "partition": self.partition,
            "local_epoch": self.local_epoch,
            "seed": self.seed,
        }
        manifest_path = os.path.join(self.output_dir, "reference_manifest.json")
        temp_path = f"{manifest_path}.tmp.{os.getpid()}"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
        os.replace(temp_path, manifest_path)
        logger.info(
            "Aggregation-damage analysis enabled: "
            f"rounds={sorted(self.analysis_rounds)}, "
            f"reference_samples={self.reference_manifest['total_samples']}, "
            f"output={self.output_dir}"
        )

    def should_analyze(self, completed_round: int) -> bool:
        return int(completed_round) in self.analysis_rounds

    def _base_row(self, completed_round: int) -> dict:
        return {
            "method": self.method,
            "partition": self.partition,
            "local_epoch": self.local_epoch,
            "seed": self.seed,
            "round": int(completed_round),
        }

    def _measure_client_replacements(
        self,
        *,
        completed_round: int,
        client_id: int,
        model,
        local_metrics: dict,
        aggregation_weight: float,
        source_state: Mapping[str, torch.Tensor],
        source_label: str,
    ) -> list[dict]:
        original_device = _model_device(model)
        moved = original_device != self.device
        if moved:
            model.to(self.device)
        rows = []
        try:
            for stage, prefix in RESNET18_BRANCH_STAGES.items():
                hybrid = _evaluate_with_stage_replacement(
                    model,
                    source_state,
                    prefix,
                    self.reference_loader,
                    self.device,
                )
                rows.append({
                    **self._base_row(completed_round),
                    "client_id": int(client_id),
                    "aggregation_weight": float(aggregation_weight),
                    "replacement_source": source_label,
                    "stage": stage,
                    "local_loss": float(local_metrics["loss"]),
                    "local_acc": float(local_metrics["acc"]),
                    "hybrid_loss": float(hybrid["loss"]),
                    "hybrid_acc": float(hybrid["acc"]),
                    "replacement_loss_damage": float(
                        hybrid["loss"] - local_metrics["loss"]
                    ),
                    "replacement_acc_damage": float(
                        local_metrics["acc"] - hybrid["acc"]
                    ),
                })
        finally:
            if moved:
                model.to(original_device)
        return rows

    def measure_post_local(
        self,
        *,
        completed_round: int,
        nets_this_round: Mapping[int, torch.nn.Module],
        fed_avg_freqs: Sequence[float],
        pre_round_global_state: Mapping[str, torch.Tensor],
    ) -> PostLocalAnalysisContext:
        client_ids = list(nets_this_round.keys())
        if len(client_ids) != len(fed_avg_freqs):
            raise ValueError(
                "Client count and FedAvg weight count differ during aggregation "
                "damage analysis."
            )
        weights = {
            int(client_id): float(fed_avg_freqs[index])
            for index, client_id in enumerate(client_ids)
        }
        local_metrics: Dict[int, dict] = {}
        pre_rows = []
        for client_id in client_ids:
            model = nets_this_round[client_id]
            original_device = _model_device(model)
            moved = original_device != self.device
            if moved:
                model.to(self.device)
            try:
                metrics = evaluate_main_classifier(
                    model, self.reference_loader, self.device
                )
            finally:
                if moved:
                    model.to(original_device)
            local_metrics[int(client_id)] = metrics
            pre_rows.extend(self._measure_client_replacements(
                completed_round=completed_round,
                client_id=int(client_id),
                model=model,
                local_metrics=metrics,
                aggregation_weight=weights[int(client_id)],
                source_state=pre_round_global_state,
                source_label="pre_round_global",
            ))
        return PostLocalAnalysisContext(
            completed_round=int(completed_round),
            local_metrics=local_metrics,
            weight_by_client=weights,
            pre_round_rows=pre_rows,
        )

    def measure_post_aggregation(
        self,
        *,
        context: PostLocalAnalysisContext,
        nets_this_round: Mapping[int, torch.nn.Module],
        aggregated_global_model,
    ) -> None:
        completed_round = int(context.completed_round)
        aggregated_metrics = evaluate_main_classifier(
            aggregated_global_model, self.reference_loader, self.device
        )
        aggregated_state = aggregated_global_model.state_dict()
        post_rows = []
        for client_id, model in nets_this_round.items():
            client_id = int(client_id)
            post_rows.extend(self._measure_client_replacements(
                completed_round=completed_round,
                client_id=client_id,
                model=model,
                local_metrics=context.local_metrics[client_id],
                aggregation_weight=context.weight_by_client[client_id],
                source_state=aggregated_state,
                source_label="post_aggregation_global",
            ))

        weight_sum = float(sum(context.weight_by_client.values()))
        weighted_local_loss = float(sum(
            context.weight_by_client[client_id] * metrics["loss"]
            for client_id, metrics in context.local_metrics.items()
        ))
        weighted_local_acc = float(sum(
            context.weight_by_client[client_id] * metrics["acc"]
            for client_id, metrics in context.local_metrics.items()
        ))
        model_row = {
            **self._base_row(completed_round),
            "reference_samples": self.reference_manifest["total_samples"],
            "reference_per_class": self.reference_manifest["samples_per_class"],
            "selected_clients": len(context.local_metrics),
            "aggregation_weight_sum": weight_sum,
            "weighted_local_loss": weighted_local_loss,
            "weighted_local_acc": weighted_local_acc,
            "aggregated_loss": float(aggregated_metrics["loss"]),
            "aggregated_acc": float(aggregated_metrics["acc"]),
            "aggregation_loss_gap": float(
                aggregated_metrics["loss"] - weighted_local_loss
            ),
            "aggregation_acc_gap": float(
                weighted_local_acc - aggregated_metrics["acc"]
            ),
        }
        all_rows = context.pre_round_rows + post_rows
        summaries = _summarize_replacement_rows(all_rows)
        _append_csv(self.model_level_path, MODEL_LEVEL_FIELDS, [model_row])
        _append_csv(self.block_raw_path, BLOCK_RAW_FIELDS, all_rows)
        _append_csv(
            self.block_summary_path, BLOCK_SUMMARY_FIELDS, summaries
        )
        self.logger.info(
            "Aggregation damage: "
            f"round={completed_round}, "
            f"loss_gap={model_row['aggregation_loss_gap']:.6f}, "
            f"acc_gap={model_row['aggregation_acc_gap']:.4f}"
        )

