#!/usr/bin/env python3
"""Recover geometry JSON when an end-to-end run finished before JSON export."""

import argparse
import json
import os
import pickle
from pathlib import Path


METHOD = "teacher_only_end_to_end_local_n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def recover(result_pickle):
    result_pickle = Path(result_pickle)
    prefix = result_pickle.with_suffix("")
    result_json = Path(f"{prefix}_postlocal_internal_geometry.json")
    final_checkpoint = Path(f"{prefix}_final.pt")
    if result_json.is_file() and result_json.stat().st_size > 0:
        return "existing", result_json
    if not final_checkpoint.is_file() or final_checkpoint.stat().st_size == 0:
        raise ValueError(f"missing final checkpoint: {final_checkpoint}")

    with open(result_pickle, "rb") as handle:
        result = pickle.load(handle)
    config = result.get("args")
    records = result.get("postlocal_feature_geometry")
    if not isinstance(config, dict) or not isinstance(records, list) or not records:
        raise ValueError(f"incomplete result pickle: {result_pickle}")

    completed_rounds = int(config["round"])
    final_record_round = max(int(record["round"]) for record in records)
    if final_record_round != completed_rounds - 1:
        raise ValueError(
            f"geometry ends at round {final_record_round}, expected "
            f"{completed_rounds - 1}: {result_pickle}"
        )
    final_record = max(records, key=lambda record: int(record["round"]))
    required = (
        "clients",
        "postlocal_cka_client_macro",
        "postlocal_logit_client_macro",
    )
    missing = [key for key in required if key not in final_record]
    if missing:
        raise ValueError(f"final geometry record misses {missing}: {result_pickle}")

    dataset = str(config["dataset"])
    global_train_samples = {
        "cifar10": 50_000,
        "cifar100": 50_000,
    }.get(dataset)
    if global_train_samples is None:
        raise ValueError(f"unsupported recovery dataset: {dataset}")
    client_count = int(config["n_clients"])
    client_samples = int(config["client_samples_per_client"])
    probe_per_class = int(config.get("postlocal_logit_probe_samples_per_class", 0))
    probe_samples = (
        global_train_samples
        if probe_per_class <= 0
        else probe_per_class * int(config["num_classes"])
    )

    payload = {
        "experiment": "multi_round_within_client_postlocal_internal_geometry",
        "args": config,
        "global_train_samples": global_train_samples,
        "client_train_sizes": [client_samples] * client_count,
        "active_client_train_samples": client_samples * client_count,
        "client_data_overlap": "disjoint_across_clients",
        "client_data_nesting": (
            "deterministic_nested_prefix_across_sample_size_conditions"
            if client_samples > 0
            else None
        ),
        "reference_set": "official_test_set",
        "feature_definition": "raw_trunk_gap_then_samplewise_l2_normalization",
        "cka_definition": (
            "centered linear CKA on raw-trunk GAP features over the same "
            "official-test samples"
        ),
        "logit_definition": (
            "per-round frozen global linear probes; per-sample class-centered "
            "and L2-normalized logit directions"
        ),
        "probe_train_split": "augmentation-free official training split",
        "probe_train_samples": probe_samples,
        "client_aggregation": "macro_mean_over_measured_postlocal_clients",
        "records": records,
        "recovered_after_export_failure": True,
    }

    result_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{result_json}.tmp.{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, result_json)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "recovered", result_json


def main():
    args = parse_args()
    root = Path(args.output_root)
    paths = sorted(root.glob(f"**/{METHOD}.pkl"))
    recovered = existing = failed = 0
    for path in paths:
        try:
            status, output = recover(path)
            if status == "recovered":
                recovered += 1
                print(f"recovered {output}")
            else:
                existing += 1
        except (KeyError, TypeError, ValueError, OSError, pickle.UnpicklingError) as error:
            failed += 1
            print(f"cannot recover {path}: {error}")
    print(
        f"recovery scan: recovered={recovered}, existing={existing}, "
        f"unrecoverable={failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
