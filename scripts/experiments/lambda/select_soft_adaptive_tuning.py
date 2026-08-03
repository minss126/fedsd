#!/usr/bin/env python3
"""Select a soft-b tau from completed stage-1/tau-tuning pickle logs."""

import argparse
import json
import pickle
import sys
from decimal import Decimal
from pathlib import Path


def tag(value):
    return format(Decimal(str(value)), ".2f").replace(".", "p")


def result_path(root, env_name, kd_temp, lambda_max, warmup, tau):
    name = (
        f"soft_b_tkd{tag(kd_temp)}_lmax{tag(lambda_max)}_"
        f"warm{warmup}_tau{tag(tau)}.pkl"
    )
    return root / env_name / "fedavg" / name


def last_window_accuracy(path, rounds, window):
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        values = payload["acc_global"]
    except (OSError, EOFError, KeyError, pickle.UnpicklingError) as error:
        raise ValueError(str(error)) from error
    if len(values) < rounds:
        raise ValueError(f"only {len(values)}/{rounds} rounds recorded")
    return sum(values[-window:]) / window


def mean_for_setting(root, envs, kd_temp, lambda_max, warmup, tau, rounds, window):
    scores = {}
    missing = []
    for env_name in envs:
        path = result_path(root, env_name, kd_temp, lambda_max, warmup, tau)
        if not path.exists():
            missing.append(f"{env_name}: {path.name} is missing")
            continue
        try:
            scores[env_name] = last_window_accuracy(path, rounds, window)
        except ValueError as error:
            missing.append(f"{env_name}: {path.name} ({error})")
    if missing:
        raise RuntimeError("; ".join(missing))
    return scores, sum(scores.values()) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--kd-temperature", required=True)
    parser.add_argument("--lambda-max", required=True)
    parser.add_argument("--taus", nargs="+", required=True)
    parser.add_argument("--stage1-kd-values", nargs="+", required=True)
    parser.add_argument("--stage1-lambda-values", nargs="+", required=True)
    parser.add_argument("--stage1-tau", default="0.85")
    parser.add_argument("--envs", nargs="+", default=["beta_0.5", "beta_0.1"])
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--require-candidate-best", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.log_root)
    try:
        stage1 = []
        for kd_temp in args.stage1_kd_values:
            for lambda_max in args.stage1_lambda_values:
                scores, mean = mean_for_setting(
                    root, args.envs, kd_temp, lambda_max, args.warmup,
                    args.stage1_tau, args.rounds, args.window,
                )
                stage1.append({
                    "kd_temperature": float(kd_temp),
                    "lambda_max": float(lambda_max),
                    "tau": float(args.stage1_tau),
                    "scores": scores,
                    "mean_last_window_accuracy": mean,
                })

        best_stage1 = max(stage1, key=lambda row: row["mean_last_window_accuracy"])
        candidate = (float(args.kd_temperature), float(args.lambda_max))
        if args.require_candidate_best and (
            best_stage1["kd_temperature"], best_stage1["lambda_max"]
        ) != candidate:
            print(
                "Stage-1 winner changed: "
                f"T_KD={best_stage1['kd_temperature']}, "
                f"lambda_max={best_stage1['lambda_max']}. "
                "Final validation was intentionally not launched.",
                file=sys.stderr,
            )
            return 2

        tau_rows = []
        for tau in args.taus:
            scores, mean = mean_for_setting(
                root, args.envs, args.kd_temperature, args.lambda_max,
                args.warmup, tau, args.rounds, args.window,
            )
            tau_rows.append({
                "tau": float(tau),
                "scores": scores,
                "mean_last_window_accuracy": mean,
            })
    except RuntimeError as error:
        print(f"Incomplete tuning logs: {error}", file=sys.stderr)
        return 3

    selected = max(tau_rows, key=lambda row: row["mean_last_window_accuracy"])
    report = {
        "selection_metric": f"mean last-{args.window} accuracy over {', '.join(args.envs)}",
        "selected_kd_temperature": float(args.kd_temperature),
        "selected_lambda_max": float(args.lambda_max),
        "selected_tau": selected["tau"],
        "tau_candidates": tau_rows,
        "stage1_candidates_at_tau": stage1,
        "stage1_winner": best_stage1,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("Stage-1 winner: "
          f"T_KD={best_stage1['kd_temperature']:.2f}, "
          f"lambda_max={best_stage1['lambda_max']:.2f}, "
          f"mean={best_stage1['mean_last_window_accuracy']:.3f}")
    for row in tau_rows:
        print(f"tau={row['tau']:.2f}: mean={row['mean_last_window_accuracy']:.3f}")
    print(f"Selected tau={selected['tau']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
