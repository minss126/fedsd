"""Evaluation-only diagnostics for branch-wise KD necessity.

The diagnostic never changes an optimizer, a parameter, or a buffer.  It
evaluates the selected clients on their own local training distributions at
the round start and again after local training/before aggregation.  Process
and DataLoader generator RNG states are restored after every diagnostic pass,
so enabling the analysis does not change the subsequent training trajectory.

The primary quantities for branch ``b`` and final teacher ``T`` are

    JS_b = JS(p_b || q_T) / log(2)
    A_b  = [q_T(y) - p_b(y)]_+
    N_b  = JS_b * A_b

``JS_b`` says that the two predictive distributions differ, ``A_b`` checks
that the teacher assigns more mass to the correct class, and ``N_b`` requires
both.  These are diagnostic candidates, not an adaptive rule in this module.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F


BRANCH_NAMES = ("B1", "B2", "B3")

CSV_FIELDS = (
    "dataset",
    "partition",
    "beta",
    "seed",
    "algorithm",
    "model",
    "communication_round",
    "completed_aggregations",
    "stage",
    "evaluation_scope",
    "client_id",
    "branch",
    "temperature",
    "samples",
    "batches",
    "branch_teacher_js",
    "branch_teacher_js_normalized",
    "teacher_branch_kl",
    "teacher_true_label_probability",
    "branch_true_label_probability",
    "teacher_label_advantage_signed",
    "teacher_label_advantage_positive",
    "teacher_label_advantage_rate",
    "need_js_x_label_advantage",
    "teacher_entropy_normalized",
    "branch_entropy_normalized",
    "teacher_accuracy",
    "branch_accuracy",
    "teacher_branch_top1_agreement",
    "teacher_correct_branch_wrong_rate",
    "branch_correct_teacher_wrong_rate",
    "existing_effective_lambda_mean",
    "existing_effective_lambda_min",
    "existing_effective_lambda_max",
    "existing_teacher_reliability_raw",
    "existing_teacher_reliability",
)


def parse_communication_rounds(spec: str, total_rounds: int) -> set[int]:
    """Parse one-based communication rounds used for local updates."""
    rounds: set[int] = set()
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(
                "KD-necessity rounds are one-based communication rounds and "
                f"must be >= 1; received {value}."
            )
        if value <= int(total_rounds):
            rounds.add(value)
    if not rounds:
        raise ValueError(
            "No KD-necessity diagnostic round falls within the configured "
            f"training horizon of {total_rounds} rounds."
        )
    return rounds


def _model_device(model) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _loader_generators(loaders: Iterable) -> list[torch.Generator]:
    generators: list[torch.Generator] = []
    seen: set[int] = set()
    for loader in loaders:
        if loader is None:
            continue
        candidates = (
            getattr(loader, "generator", None),
            getattr(getattr(loader, "sampler", None), "generator", None),
        )
        for generator in candidates:
            if isinstance(generator, torch.Generator) and id(generator) not in seen:
                seen.add(id(generator))
                generators.append(generator)
    return generators


@contextmanager
def preserve_evaluation_rng(loaders: Iterable, device: torch.device):
    """Restore process and explicit DataLoader RNGs after an evaluation pass."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state(device)
    generators = _loader_generators(loaders)
    generator_states = [generator.get_state() for generator in generators]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        for generator, state in zip(generators, generator_states):
            generator.set_state(state)


def _extract_teacher_and_branches(model_output):
    if not isinstance(model_output, tuple) or len(model_output) != 8:
        raise ValueError(
            "KD-necessity diagnostics require a BYOT model returning "
            "(final, B1, B2, B3, final_feature, B1_feature, B2_feature, "
            "B3_feature)."
        )
    teacher_logits, branch1, branch2, branch3 = model_output[:4]
    return teacher_logits, (branch1, branch2, branch3)


@dataclass
class _BranchAccumulator:
    samples: int = 0
    batches: int = 0
    sums: Dict[str, float] = field(default_factory=dict)

    def add(self, values: Mapping[str, torch.Tensor]) -> None:
        batch_samples = None
        for name, value in values.items():
            flat = value.detach().float().reshape(-1)
            if batch_samples is None:
                batch_samples = int(flat.numel())
            elif int(flat.numel()) != batch_samples:
                raise ValueError(f"Metric {name!r} has an inconsistent batch size.")
            self.sums[name] = self.sums.get(name, 0.0) + float(flat.sum().item())
        self.samples += int(batch_samples or 0)
        self.batches += 1

    def means(self) -> Dict[str, float]:
        if self.samples <= 0:
            raise ValueError("KD-necessity diagnostic loader produced no samples.")
        return {name: value / self.samples for name, value in self.sums.items()}


def evaluate_branch_teacher_metrics(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    temperature: float = 1.0,
    max_batches: int = 0,
) -> Dict[str, dict]:
    """Return sample-weighted B1/B2/B3 metrics for one model/data pair."""
    if temperature <= 0.0:
        raise ValueError("KD-necessity temperature must be positive.")
    if num_classes < 2:
        raise ValueError("KD-necessity diagnostics require at least two classes.")

    original_device = _model_device(model)
    was_training = bool(model.training)
    moved = original_device != device
    if moved:
        model.to(device)
    model.eval()
    accumulators = {name: _BranchAccumulator() for name in BRANCH_NAMES}
    log_num_classes = math.log(float(num_classes))
    log_two = math.log(2.0)

    try:
        with torch.inference_mode():
            for batch_index, (inputs, targets) in enumerate(loader):
                if int(max_batches) > 0 and batch_index >= int(max_batches):
                    break
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()
                teacher_logits, branch_logits = _extract_teacher_and_branches(
                    model(inputs)
                )
                teacher_log_prob = F.log_softmax(
                    teacher_logits.float() / temperature, dim=1
                )
                teacher_prob = teacher_log_prob.exp()
                teacher_y = teacher_prob.gather(1, targets[:, None]).squeeze(1)
                teacher_prediction = teacher_prob.argmax(dim=1)
                teacher_correct = teacher_prediction.eq(targets)
                teacher_entropy = -(
                    teacher_prob * teacher_log_prob
                ).sum(dim=1) / log_num_classes

                for name, logits in zip(BRANCH_NAMES, branch_logits):
                    branch_log_prob = F.log_softmax(
                        logits.float() / temperature, dim=1
                    )
                    branch_prob = branch_log_prob.exp()
                    branch_y = branch_prob.gather(1, targets[:, None]).squeeze(1)
                    branch_prediction = branch_prob.argmax(dim=1)
                    branch_correct = branch_prediction.eq(targets)
                    midpoint = 0.5 * (teacher_prob + branch_prob)
                    midpoint_log = midpoint.clamp_min(1e-12).log()
                    js = 0.5 * (
                        (teacher_prob * (teacher_log_prob - midpoint_log)).sum(dim=1)
                        + (branch_prob * (branch_log_prob - midpoint_log)).sum(dim=1)
                    )
                    teacher_branch_kl = (
                        teacher_prob * (teacher_log_prob - branch_log_prob)
                    ).sum(dim=1)
                    signed_advantage = teacher_y - branch_y
                    positive_advantage = signed_advantage.clamp_min(0.0)
                    js_normalized = (js / log_two).clamp(min=0.0, max=1.0)
                    branch_entropy = -(
                        branch_prob * branch_log_prob
                    ).sum(dim=1) / log_num_classes

                    accumulators[name].add(
                        {
                            "branch_teacher_js": js,
                            "branch_teacher_js_normalized": js_normalized,
                            "teacher_branch_kl": teacher_branch_kl,
                            "teacher_true_label_probability": teacher_y,
                            "branch_true_label_probability": branch_y,
                            "teacher_label_advantage_signed": signed_advantage,
                            "teacher_label_advantage_positive": positive_advantage,
                            "teacher_label_advantage_rate": signed_advantage.gt(0),
                            "need_js_x_label_advantage": (
                                js_normalized * positive_advantage
                            ),
                            "teacher_entropy_normalized": teacher_entropy,
                            "branch_entropy_normalized": branch_entropy,
                            "teacher_accuracy": teacher_correct,
                            "branch_accuracy": branch_correct,
                            "teacher_branch_top1_agreement": (
                                teacher_prediction.eq(branch_prediction)
                            ),
                            "teacher_correct_branch_wrong_rate": (
                                teacher_correct & ~branch_correct
                            ),
                            "branch_correct_teacher_wrong_rate": (
                                branch_correct & ~teacher_correct
                            ),
                        }
                    )
    finally:
        model.train(was_training)
        if moved:
            model.to(original_device)

    return {
        branch: {
            **accumulator.means(),
            "samples": int(accumulator.samples),
            "batches": int(accumulator.batches),
        }
        for branch, accumulator in accumulators.items()
    }


class KDNecessityDiagnostic:
    """Run-local CSV writer for stage-1 KD-necessity diagnostics."""

    def __init__(self, args, log_file_name: str, device, logger=None):
        if getattr(args, "model", "") not in ("resnet18_byot", "mobilenet_byot"):
            raise ValueError(
                "--analyze_kd_necessity requires a BYOT model with B1/B2/B3 logits."
            )
        self.args = args
        self.device = torch.device(device)
        self.logger = logger
        self.rounds = parse_communication_rounds(
            getattr(args, "kd_necessity_analysis_rounds", "50,100,250,500"),
            int(args.round),
        )
        self.temperature = float(
            getattr(args, "kd_necessity_temperature", 1.0)
        )
        self.max_batches = max(
            0, int(getattr(args, "kd_necessity_max_batches", 0))
        )
        self.client_count = max(
            0, int(getattr(args, "kd_necessity_client_count", 0))
        )
        output_dir = str(
            getattr(args, "kd_necessity_output_dir", "") or ""
        ).strip()
        if not output_dir:
            output_dir = os.path.join(
                str(args.logdir), f"{log_file_name}_kd_necessity"
            )
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "client_branch_metrics.csv")
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        os.makedirs(output_dir, exist_ok=True)
        overwrite = bool(getattr(args, "kd_necessity_overwrite", False))
        if overwrite and os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        self._write_manifest(log_file_name)

    def _write_manifest(self, log_file_name: str) -> None:
        manifest = {
            "diagnostic": "branch-wise KD necessity stage 1",
            "training_rule_modified": False,
            "log_file_name": log_file_name,
            "dataset": str(self.args.dataset),
            "partition": str(self.args.partition),
            "beta": float(getattr(self.args, "beta", float("nan"))),
            "seed": int(self.args.seed),
            "algorithm": str(self.args.alg),
            "model": str(self.args.model),
            "communication_rounds": sorted(self.rounds),
            "round_semantics": {
                "communication_round": "one-based local-update round",
                "client_pre_local_completed_aggregations": "communication_round - 1",
                "client_post_local_pre_aggregation_completed_aggregations": (
                    "communication_round - 1; current local update is not yet aggregated"
                ),
            },
            "evaluation_scope": "each selected client's local train loader",
            "all_selected_clients": self.client_count == 0,
            "client_count_cap": int(self.client_count),
            "all_local_batches": self.max_batches == 0,
            "max_batches": int(self.max_batches),
            "temperature": float(self.temperature),
            "rng_restored": True,
            "definitions": {
                "branch_teacher_js_normalized": "JS(p_b,q_T)/log(2)",
                "teacher_label_advantage_positive": "max(q_T(y)-p_b(y),0)",
                "need_js_x_label_advantage": (
                    "JS(p_b,q_T)/log(2) * max(q_T(y)-p_b(y),0)"
                ),
            },
            "existing_adaptive_rule": {
                "changed_by_this_diagnostic": False,
                "form": "lambda_client = lambda_round * R_client * B_client",
                "shared_across_B1_B2_B3": True,
                "lambda_max": float(getattr(self.args, "byot_alpha", 0.0)),
                "round_schedule": str(
                    getattr(self.args, "byot_round_lambda_schedule", "none")
                ),
                "round_warmup": int(
                    getattr(self.args, "byot_round_lambda_warmup", 0)
                ),
                "R_proxy": str(
                    getattr(self.args, "byot_client_proxy", "none")
                ),
                "B_proxy": str(
                    getattr(self.args, "byot_client_skew_proxy", "none")
                ),
                "B_correction": str(
                    getattr(
                        self.args, "byot_client_skew_correction_mode", "multiply"
                    )
                ),
            },
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

    def should_analyze(self, communication_round: int) -> bool:
        return int(communication_round) in self.rounds

    def _selected_ids(self, nets: Mapping[int, torch.nn.Module]) -> list[int]:
        ids = sorted(int(client_id) for client_id in nets)
        if self.client_count > 0:
            ids = ids[: self.client_count]
        return ids

    def _append_rows(self, rows: list[dict]) -> None:
        write_header = not os.path.exists(self.csv_path) or os.path.getsize(
            self.csv_path
        ) == 0
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def measure_clients(
        self,
        communication_round: int,
        stage: str,
        nets: Mapping[int, torch.nn.Module],
        dataloaders: Mapping[int, object],
        client_alpha_stats: Optional[Mapping[int, Mapping[str, float]]] = None,
        client_reliability_stats: Optional[Mapping[int, Mapping[str, float]]] = None,
    ) -> list[dict]:
        if not self.should_analyze(communication_round):
            return []
        if stage not in ("client_pre_local", "client_post_local_pre_aggregation"):
            raise ValueError(f"Unknown KD-necessity stage: {stage}")
        client_ids = self._selected_ids(nets)
        loaders = [dataloaders.get(client_id) for client_id in client_ids]
        rows: list[dict] = []
        completed_aggregations = int(communication_round) - 1

        with preserve_evaluation_rng(loaders, self.device):
            for client_id in client_ids:
                loader = dataloaders.get(client_id)
                if loader is None:
                    continue
                metrics = evaluate_branch_teacher_metrics(
                    nets[client_id],
                    loader,
                    self.device,
                    num_classes=int(self.args.num_classes),
                    temperature=self.temperature,
                    max_batches=self.max_batches,
                )
                for branch in BRANCH_NAMES:
                    alpha_stats = (client_alpha_stats or {}).get(client_id, {})
                    reliability_stats = (client_reliability_stats or {}).get(
                        client_id, {}
                    )
                    rows.append(
                        {
                            "dataset": str(self.args.dataset),
                            "partition": str(self.args.partition),
                            "beta": float(getattr(self.args, "beta", float("nan"))),
                            "seed": int(self.args.seed),
                            "algorithm": str(self.args.alg),
                            "model": str(self.args.model),
                            "communication_round": int(communication_round),
                            "completed_aggregations": completed_aggregations,
                            "stage": stage,
                            "evaluation_scope": "client_local_train",
                            "client_id": int(client_id),
                            "branch": branch,
                            "temperature": self.temperature,
                            **metrics[branch],
                            "existing_effective_lambda_mean": float(
                                alpha_stats.get("mean", float("nan"))
                            ),
                            "existing_effective_lambda_min": float(
                                alpha_stats.get("min", float("nan"))
                            ),
                            "existing_effective_lambda_max": float(
                                alpha_stats.get("max", float("nan"))
                            ),
                            "existing_teacher_reliability_raw": float(
                                reliability_stats.get(
                                    "raw_reliability", float("nan")
                                )
                            ),
                            "existing_teacher_reliability": float(
                                reliability_stats.get("reliability", float("nan"))
                            ),
                        }
                    )

        self._append_rows(rows)
        if self.logger is not None and rows:
            by_branch = {}
            for branch in BRANCH_NAMES:
                branch_rows = [row for row in rows if row["branch"] == branch]
                by_branch[branch] = float(
                    np.mean(
                        [row["need_js_x_label_advantage"] for row in branch_rows]
                    )
                )
            self.logger.info(
                "KD-necessity diagnostic "
                f"round={communication_round} stage={stage} clients={len(client_ids)} "
                + ", ".join(
                    f"{branch}_N={by_branch[branch]:.6f}"
                    for branch in BRANCH_NAMES
                )
            )
        return rows
