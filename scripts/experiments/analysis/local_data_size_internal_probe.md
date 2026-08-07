# Within-client local-data-size representation diagnostic

## Question

Starting from one fixed global checkpoint, does training on a smaller finite
local dataset increase the mismatch among B1, B2, B3, and final representations
inside the same post-local model before aggregation?

This experiment does **not** measure differences between clients. Multiple
local-model forks are repetitions used to estimate the typical within-client
effect and its uncertainty.

## Controlled setup

- Dataset: CIFAR-100
- Local objective: final teacher cross-entropy only
- BYOT/private branch forward and branch loss: disabled
- Initialization: the same fixed global checkpoint for every local fork
- Local data: deterministic IID subsets; subsets are nested across sample-size
  conditions for the same seed and fork id
- Primary optimization control: the same number of SGD steps in every condition
- Measurement reference: the full official CIFAR-100 test set (100 images/class)
- Semantic readout: four depth-specific linear probes fitted once on frozen raw
  features from the fixed global checkpoint, then frozen for every local fork

The default launcher uses the existing teacher-only K=20 final global model as
the fixed initialization for a controlled additional local-training fork. Set
`BASE_CHECKPOINT` to use another common initialization.

## Measurements and intended claims

| Measurement | Definition / direction | What it supports |
|---|---|---|
| Within-class feature variance, `W` | Mean squared distance from each L2-normalized feature to its class centroid | Larger `W` means examples of the same class are less compact at that depth |
| Between-class feature variance, `B` | Mean squared distance among class centroids | Larger `B` means class centroids are more separated |
| Feature separability, `B/W` | Between-class variance divided by within-class variance | Smaller `B/W` means weaker class geometry; it prevents interpreting a simultaneous collapse of both `W` and `B` as improvement |
| Pairwise centered-logit cosine | Six pairs: B1-B2, B1-B3, B1-final, B2-B3, B2-final, B3-final | Smaller cosine means different depths prefer different classes inside the same local model |
| Pairwise softmax JSD | Symmetric divergence between the two class-probability vectors | Larger JSD means different depths produce different predictive distributions, including confidence |
| Depth directional variance | Variance of four centered, unit-normalized logit directions around their within-model mean | Larger variance is a single-number summary of greater internal depth inconsistency |
| Probe accuracy / entropy / centered-logit norm | Per-depth readout diagnostics | Verifies that low JSD or high cosine is not a trivial consequence of uniform or near-zero logits |
| Delta from fixed global | Post-local metric minus the same metric at initialization | Attributes the observed internal change to the controlled local-training intervention rather than the architecture's baseline depth gap |

The main paper claim is supported if decreasing local sample size systematically
reduces `B/W`, reduces pairwise centered-logit cosine, and/or increases pairwise
JSD and depth directional variance. A branch-specific curve supports the more
specific statement that finite local data affect network depths unevenly.

These measurements do not by themselves support a claim that clients differ
from one another. They characterize the geometry and semantic consistency
inside a typical post-local model.

## Default design

- Local sample sizes: 100, 250, 500, 1,000, 2,500
- Sampling seeds: 0, 1, 2
- Local forks per sample-size/seed condition: 10
- Training budget: 100 SGD steps, batch size 50
- Local optimizer: SGD, LR 0.01, momentum 0.9, weight decay 1e-5
- Probe: 30 epochs using all 500 CIFAR-100 train images per class

The fixed-step design is primary because it holds optimization budget constant.
For a practical-FL sensitivity analysis with fixed local epochs, run with
`TRAIN_BUDGET=epochs LOCAL_EPOCHS=5` and use a separate `OUTPUT_ROOT`.

## Running

Four GPUs:

```bash
scripts/experiments/analysis/run_local_data_size_internal_probe_4gpu.sh
```

Two GPUs:

```bash
scripts/experiments/analysis/run_local_data_size_internal_probe_2gpu.sh
```

Different GPU ids:

```bash
GPUS_OVERRIDE="2 3" scripts/experiments/analysis/run_local_data_size_internal_probe_2gpu.sh
```

Example fixed-epoch sensitivity run:

```bash
TRAIN_BUDGET=epochs \
LOCAL_EPOCHS=5 \
OUTPUT_ROOT=logs/analysis/logs_local_data_size_internal_probe_epochs \
scripts/experiments/analysis/run_local_data_size_internal_probe_4gpu.sh
```

Useful overrides include `BASE_CHECKPOINT`, `SAMPLE_SIZES_OVERRIDE`,
`SEEDS_OVERRIDE`, `CLIENTS_PER_CONDITION`, `LOCAL_STEPS`, `LOCAL_LR`,
`NUM_WORKERS`, and `SKIP_EXISTING`.

## Outputs

- `shared_round_start_probe.pt`: one frozen probe and fixed-global baseline
- `sample_<n>/seed_<s>/metrics.json`: every local fork's raw measurements
- `sample_<n>/seed_<s>/terminal.log`: job progress and timing
- `summary.json`: client-pooled and seed-macro summaries
- `summary.csv`: plotting-friendly long-form summary

Use the seed-macro mean and Student-t 95% interval for paper plots. With only
three seeds, individual seed means should also be shown or retained in the
supplement.

## Expected time

The estimate excludes a first-time CIFAR-100 download and assumes GPU throughput
similar to the existing teacher-only logs (roughly 90-125 seconds per 250,000
training examples, before multi-process contention):

- 4 GPUs: approximately 8-15 minutes
- 2 GPUs: approximately 15-25 minutes

Fixed-epoch runs have similar cost under the default sample-size grid. Runtime
scales approximately linearly with the number of seeds, local forks, evaluated
test images, and fixed steps.
