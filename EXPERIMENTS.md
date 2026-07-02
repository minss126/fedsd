# Experiment Index

This file maps experiment purposes to their run scripts, analysis scripts, and output directories.
Run commands from the repository root unless noted otherwise. The shell scripts also `cd` to the
repository root automatically.

## Main Layout

| Type | Location |
|---|---|
| Run scripts | `scripts/experiments/<topic>/` |
| Analysis scripts | `analysis/<topic>/` |
| Analysis outputs | `results_analysis/` |
| Main training logs | `logs/`, `logs_tuning/`, `logs_drift/`, `logs_probe/` |

## Alpha Experiments

| Purpose | Run Script | Analysis | Output |
|---|---|---|---|
| Alpha adaptation ablation | `scripts/experiments/alpha/run_fedsd_alpha_ablation.sh` | `analysis/alpha/analyze_fedsd_alpha_ablation.py` | `results_analysis/fedsd_alpha_ablation/` |
| Fixed alpha sweep | `scripts/experiments/alpha/run_fedsd_fixed_alpha_sweep.sh` | `analysis/alpha/analyze_fedsd_fixed_alpha_sweep.py` | `results_analysis/fedsd_fixed_alpha_sweep/` |
| Fixed alpha partition sweep | `scripts/experiments/alpha/run_fedsd_fixed_alpha_partition_sweep.sh` | `analysis/alpha/analyze_fedsd_fixed_alpha_partition_sweep.py` | `results_analysis/fedsd_fixed_alpha_partition_sweep/` |
| Proxy alpha ablation | `scripts/experiments/alpha/run_fedsd_proxy_ablation.sh` | `analysis/alpha/analyze_fedsd_proxy_ablation.py` | `results_analysis/fedsd_proxy_ablation/` |

## Drift Experiments

| Purpose | Run Script | Analysis | Output |
|---|---|---|---|
| Update-space client drift sweep | `scripts/experiments/drift/run_fedsd_drift_sweep.sh` | `analysis/drift/analyze_fedsd_drift_sweep.py` | `results_analysis/fedsd_drift_sweep/` |
| Gradient dissimilarity probe | `scripts/experiments/drift/run_fedsd_gradient_probe_sweep.sh` | `analysis/drift/analyze_fedsd_gradient_probe_sweep.py` | `results_analysis/fedsd_gradient_probe_sweep/` |
| FedAvg update-space client drift sweep | `scripts/experiments/drift/run_fedsd_drift_fedavg_sweep.sh` | `analysis/drift/analyze_fedsd_drift_sweep.py --root-dir logs_drift_fedavg --base-algo fedavg --out-dir results_analysis/fedsd_drift_fedavg_sweep` | `results_analysis/fedsd_drift_fedavg_sweep/` |
| FedAvg gradient dissimilarity probe | `scripts/experiments/drift/run_fedsd_gradient_probe_fedavg_sweep.sh` | `analysis/drift/analyze_fedsd_gradient_probe_sweep.py --root-dir logs_probe_fedavg --base-algo fedavg --out-dir results_analysis/fedsd_gradient_probe_fedavg_sweep` | `results_analysis/fedsd_gradient_probe_fedavg_sweep/` |
| FedAvg combined gradient probe + update drift | `scripts/experiments/drift/run_fedsd_probe_drift_fedavg_sweep.sh` | `analysis/drift/analyze_fedsd_gradient_probe_sweep.py --root-dir logs_probe_drift_fedavg --base-algo fedavg --method-suffix probe_drift --include-alpha0p01 --out-dir results_analysis/fedsd_gradient_probe_drift_fedavg_sweep` and `analysis/drift/analyze_fedsd_drift_sweep.py --root-dir logs_probe_drift_fedavg --base-algo fedavg --method-suffix probe_drift --out-dir results_analysis/fedsd_update_probe_drift_fedavg_sweep` | `results_analysis/fedsd_gradient_probe_drift_fedavg_sweep/`, `results_analysis/fedsd_update_probe_drift_fedavg_sweep/` |

## Branch Agreement Experiments

| Purpose | Run Script | Analysis | Output |
|---|---|---|---|
| Branch agreement min-scale sweep | `scripts/experiments/branch/run_branch_agreement_min_scale_sweep.sh` | `analysis/branch/analyze_branch_agreement_min_scale.py` | `results_analysis/branch_agreement_min_scale/` |
| Branch agreement seed check | `scripts/experiments/branch/run_branch_agreement_seed_check.sh` | `analysis/branch/analyze_branch_agreement_seed_check.py` | `results_analysis/branch_agreement_seed_check/` |
| Branch agreement partition pilot | `scripts/experiments/branch/run_branch_agreement_partition_pilot.sh` | `analysis/branch/analyze_branch_agreement_partition_pilot.py` | `results_analysis/branch_agreement_partition_pilot/` |
| Soft branch proxy pilot | `scripts/experiments/branch/run_soft_branch_proxy_pilot.sh` | `analysis/branch/analyze_soft_branch_proxy_pilot.py` | `results_analysis/soft_branch_proxy_pilot/` |

## General Analysis

| Purpose | Analysis | Output |
|---|---|---|
| Convergence metrics from existing curves | `analysis/general/analyze_convergence_metrics.py` | `results_analysis/convergence_metrics/` |

## Baseline And Selective Scripts

| Purpose | Run Script |
|---|---|
| Full original experiment grid | `scripts/experiments/baseline/run_experiment.sh` |
| IID FedAvg BYOT rerun | `scripts/experiments/baseline/run_iid_fedavg_byot.sh` |
| Selective runs | `scripts/experiments/selective/run_selective.sh` |

## Examples

```bash
./scripts/experiments/drift/run_fedsd_gradient_probe_sweep.sh
./venv/bin/python analysis/drift/analyze_fedsd_gradient_probe_sweep.py
```

```bash
./scripts/experiments/alpha/run_fedsd_fixed_alpha_partition_sweep.sh
./venv/bin/python analysis/alpha/analyze_fedsd_fixed_alpha_partition_sweep.py
```
