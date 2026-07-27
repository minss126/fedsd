#!/usr/bin/env python3
"""Post-process saved post-local full logits without rerunning FL.

The saved tensors contain every selected client's B1/B2/B3/teacher logits on
the same reference samples.  This script separates three notions that must not
be conflated in the BYOT analysis:

* branch--teacher class-relation agreement (including non-target JS);
* client prediction disagreement in probability space; and
* raw-logit scale/variance.

It also emits error-conditioned confidence and ECE, so an "overconfidence"
claim is evaluated rather than inferred from entropy alone.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_csv(raw, cast=str):
    return [cast(value.strip()) for value in str(raw).split(",") if value.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        default="logs/analysis/logs_postlocal_branch_distribution_full_logits_r500",
    )
    parser.add_argument("--datasets", default="cifar10,cifar100")
    parser.add_argument("--partitions", default="iid,beta_0.5,beta_0.1")
    parser.add_argument("--alphas", default="0p00,1p00")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--rounds", default="470,480,490")
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=-1.0,
        help="Temperature for KD-space metrics. A non-positive value reads each run's saved config.",
    )
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--output-prefix",
        default="logs/analysis/full_branch_logit_relation",
        help="Writes <prefix>_rows.{json,csv} and <prefix>_summary.csv.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def js_divergence(p, q):
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=-1)
        + (q * (q.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=-1)
    )


def expected_calibration_error(probs, labels, bins):
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels).float()
    ece = torch.zeros((), device=probs.device)
    for bin_id in range(max(int(bins), 1)):
        low = bin_id / max(int(bins), 1)
        high = (bin_id + 1) / max(int(bins), 1)
        in_bin = (confidence >= low) & (
            confidence <= high if bin_id == bins - 1 else confidence < high
        )
        if in_bin.any():
            ece += in_bin.float().mean() * (correct[in_bin].mean() - confidence[in_bin].mean()).abs()
    return float(ece.item())


def head_metrics(logits, labels, ece_bins):
    probs = F.softmax(logits, dim=1)
    confidence, prediction = probs.max(dim=1)
    true_prob = probs.gather(1, labels[:, None]).squeeze(1)
    entropy_norm = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / math.log(logits.shape[1])
    wrong = prediction.ne(labels)
    return {
        "acc": float(prediction.eq(labels).float().mean().item()),
        "nll": float(F.cross_entropy(logits, labels, reduction="mean").item()),
        "true_label_prob": float(true_prob.mean().item()),
        "confidence": float(confidence.mean().item()),
        "entropy_norm": float(entropy_norm.mean().item()),
        "wrong_confidence": float(confidence[wrong].mean().item()) if wrong.any() else float("nan"),
        "ece": expected_calibration_error(probs, labels, ece_bins),
    }


def branch_teacher_metrics(branch_logits, teacher_logits, labels, kd_temperature):
    num_classes = branch_logits.shape[1]
    branch_prob = F.softmax(branch_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits, dim=1)
    branch_kd = F.softmax(branch_logits / kd_temperature, dim=1)
    teacher_kd = F.softmax(teacher_logits / kd_temperature, dim=1)
    label_mask = F.one_hot(labels, num_classes=num_classes).bool()
    branch_nt = branch_kd.masked_fill(label_mask, 0.0)
    teacher_nt = teacher_kd.masked_fill(label_mask, 0.0)
    branch_nt = branch_nt / branch_nt.sum(dim=1, keepdim=True).clamp_min(1e-12)
    teacher_nt = teacher_nt / teacher_nt.sum(dim=1, keepdim=True).clamp_min(1e-12)
    branch_centered = branch_logits - branch_logits.mean(dim=1, keepdim=True)
    teacher_centered = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)
    cosine = F.cosine_similarity(branch_centered, teacher_centered, dim=1)
    normalized_l2 = (branch_centered - teacher_centered).norm(dim=1) / math.sqrt(num_classes)
    py_gap = (
        branch_prob.gather(1, labels[:, None]).squeeze(1)
        - teacher_prob.gather(1, labels[:, None]).squeeze(1)
    ).abs()
    return {
        "bt_js_t1": float(js_divergence(branch_prob, teacher_prob).mean().item()),
        "bt_js_tkd": float(js_divergence(branch_kd, teacher_kd).mean().item()),
        "bt_non_target_js_tkd": float(js_divergence(branch_nt, teacher_nt).mean().item()),
        "bt_centered_logit_cosine": float(cosine.mean().item()),
        "bt_normalized_logit_l2": float(normalized_l2.mean().item()),
        "bt_abs_true_label_prob_gap": float(py_gap.mean().item()),
    }


def client_dispersion(head_logits, membership):
    """Common-reference client disagreement for one head and frequency group."""
    js_values, l2_values, raw_var_values = [], [], []
    for sample_idx in range(head_logits.shape[1]):
        selected = membership[:, sample_idx]
        if int(selected.sum().item()) < 2:
            continue
        logits = head_logits[selected, sample_idx]
        probs = F.softmax(logits, dim=1)
        mean_prob = probs.mean(dim=0, keepdim=True)
        js_values.append(js_divergence(probs, mean_prob.expand_as(probs)).mean())
        l2_values.append(((probs - mean_prob) ** 2).sum(dim=1).mean())
        raw_var_values.append(logits.var(dim=0, unbiased=False).mean())
    if not js_values:
        return {
            "client_js_to_mean": float("nan"),
            "client_prob_l2_dispersion": float("nan"),
            "client_raw_logit_variance": float("nan"),
            "client_divergence_samples": 0,
        }
    return {
        "client_js_to_mean": float(torch.stack(js_values).mean().item()),
        "client_prob_l2_dispersion": float(torch.stack(l2_values).mean().item()),
        "client_raw_logit_variance": float(torch.stack(raw_var_values).mean().item()),
        "client_divergence_samples": len(js_values),
    }


def saved_temperature(config_path, override):
    if override > 0:
        return override
    if not config_path.exists():
        return 0.5
    with config_path.open("r", encoding="utf-8") as file:
        return float(json.load(file).get("temperature", 0.5))


def as_csv_value(value):
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_csv_value(row.get(key, "")) for key in fields})


def make_summary(rows):
    group_keys = ("dataset", "partition", "alpha", "frequency_group", "head", "metric_scope")
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in group_keys)].append(row)
    summary = []
    identifier_keys = set(group_keys) | {"seed", "round", "kd_temperature"}
    for key, bucket in sorted(buckets.items()):
        result = dict(zip(group_keys, key))
        result["n_checkpoints"] = len(bucket)
        numeric_keys = set().union(*(row.keys() for row in bucket)) - identifier_keys
        for metric in numeric_keys:
            values = [row[metric] for row in bucket if isinstance(row.get(metric), (int, float)) and math.isfinite(float(row[metric]))]
            if values:
                result[metric] = sum(values) / len(values)
        summary.append(result)
    return summary


def main():
    args = parse_args()
    root = Path(args.log_root)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    datasets = parse_csv(args.datasets)
    partitions = parse_csv(args.partitions)
    alphas = parse_csv(args.alphas)
    seeds = parse_csv(args.seeds, int)
    rounds = parse_csv(args.rounds, int)
    rows = []
    missing = []

    for dataset in datasets:
        setting_dataset = f"{dataset}_resnet18"
        for partition in partitions:
            for alpha in alphas:
                for seed in seeds:
                    stem = f"alpha{alpha}_seed{seed}_client_pretrain_branch_freq"
                    run_dir = root / setting_dataset / partition / "fedavg"
                    logits_dir = run_dir / f"{stem}_full_logits"
                    temperature = saved_temperature(run_dir / f"{stem}.json", args.kd_temperature)
                    for round_idx in rounds:
                        path = logits_dir / f"round_{round_idx:04d}.pt"
                        if not path.exists():
                            missing.append(str(path))
                            continue
                        payload = torch.load(path, map_location=device, weights_only=False)
                        logits = payload["logits"].to(device=device, dtype=torch.float32)
                        labels = payload["reference_labels"].to(device=device, dtype=torch.long)
                        groups = payload["local_frequency_groups"].to(device=device, dtype=torch.long)
                        head_order = list(payload["head_order"])
                        group_order = list(payload["group_order"])
                        if "teacher" not in head_order:
                            raise ValueError(f"Teacher head missing from {path}")
                        teacher_idx = head_order.index("teacher")
                        group_for_sample = groups[:, labels]

                        for group_idx, group_name in enumerate(group_order):
                            membership = group_for_sample.eq(group_idx)
                            flat_labels = labels.unsqueeze(0).expand_as(membership)[membership]
                            if flat_labels.numel() == 0:
                                continue
                            base = {
                                "dataset": dataset,
                                "partition": partition,
                                "alpha": alpha,
                                "seed": seed,
                                "round": round_idx,
                                "frequency_group": group_name,
                                "kd_temperature": temperature,
                            }
                            teacher_logits = logits[:, teacher_idx][membership]
                            teacher_row = dict(base, head="Teacher", metric_scope="head")
                            teacher_row.update(head_metrics(teacher_logits, flat_labels, args.ece_bins))
                            teacher_row.update(client_dispersion(logits[:, teacher_idx], membership))
                            rows.append(teacher_row)

                            for head_idx, head_name in enumerate(head_order):
                                if head_name == "teacher":
                                    continue
                                branch_logits = logits[:, head_idx][membership]
                                row = dict(base, head=head_name.upper(), metric_scope="branch_teacher")
                                row.update(head_metrics(branch_logits, flat_labels, args.ece_bins))
                                row.update(branch_teacher_metrics(branch_logits, teacher_logits, flat_labels, temperature))
                                row.update(client_dispersion(logits[:, head_idx], membership))
                                rows.append(row)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{output_prefix}_rows.json")
    csv_path = Path(f"{output_prefix}_rows.csv")
    summary_path = Path(f"{output_prefix}_summary.csv")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"rows": rows, "missing": missing}, file, indent=2)
    write_csv(csv_path, rows)
    write_csv(summary_path, make_summary(rows))
    print(f"rows={len(rows)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")
    if missing:
        print(f"missing_checkpoints={len(missing)}")


if __name__ == "__main__":
    main()
