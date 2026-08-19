# Centralized Epoch-100 Checkpoint: Local-n Motivation Results

## 1. Completion and protocol

- Status: complete; no active experiment process and no error/OOM traceback in terminal logs.
- Datasets: CIFAR-10 and CIFAR-100.
- Source model: teacher-only centralized training for 100 epochs.
- Local budget: fixed 100 optimizer steps, batch size 50 (5,000 processed examples per local model).
- Local unique sample counts: \(n\in\{100,500,2500\}\).
- Replication: 3 sampling seeds and 10 local models per condition.
- Evaluation: full official test set.
- Values below: first average the 10 local models within each seed, then average the three seed means.

## 2. Aggregate representation and performance metrics

### CIFAR-10

| Metric | Global | n=100 | n=500 | n=2500 | Delta (2500-100) |
|---|---:|---:|---:|---:|---:|
| Centered logit cosine | 0.6033 | 0.6020 | 0.5965 | 0.5621 | -0.0398 |
| Directional logit variance | 0.2975 | 0.2985 | 0.3027 | 0.3284 | +0.0299 |
| Softmax JSD | 0.2254 | 0.2228 | 0.2026 | 0.1826 | -0.0402 |
| Linear CKA | 0.4494 | 0.4665 | 0.5433 | 0.5599 | +0.0934 |
| Native Final accuracy (%) | 93.99 | 92.98 | 81.27 | 72.64 | -20.34 pp |
| Native Final test CE | 0.2116 | 0.2498 | 0.7127 | 0.9661 | +0.7164 |
| Mean local training loss | - | 0.0025 | 0.0661 | 0.3088 | +0.3063 |

### CIFAR-100

| Metric | Global | n=100 | n=500 | n=2500 | Delta (2500-100) |
|---|---:|---:|---:|---:|---:|
| Centered logit cosine | 0.4955 | 0.4912 | 0.4886 | 0.4661 | -0.0251 |
| Directional logit variance | 0.3784 | 0.3816 | 0.3836 | 0.4005 | +0.0188 |
| Softmax JSD | 0.2578 | 0.2330 | 0.2368 | 0.2284 | -0.0046 |
| Linear CKA | 0.4744 | 0.4803 | 0.4853 | 0.5141 | +0.0339 |
| Native Final accuracy (%) | 75.08 | 73.14 | 70.53 | 49.34 | -23.80 pp |
| Native Final test CE | 0.9572 | 1.0173 | 1.1486 | 2.3177 | +1.3005 |
| Mean local training loss | - | 0.0072 | 0.0374 | 0.4310 | +0.4238 |

## 3. Final-CE prefix-gradient agreement

\(U=\|\sum_b w_b g_b\|_2\) is the norm of the net test gradient, \(A=\sum_b w_b\|g_b\|_2\) is the mean batch-gradient norm, and \(\rho=U/A\). A high \(\rho\) means the test-batch gradients point in a common direction. It does not, by itself, mean that the current model is better.

### CIFAR-10

| Prefix | Metric | Global | n=100 | n=500 | n=2500 | Delta (2500-100) |
|---|---|---:|---:|---:|---:|---:|
| B1 | rho | 0.2496 | 0.4589 | 0.8914 | 0.9657 | +0.5068 |
| B2 | rho | 0.2512 | 0.4612 | 0.8828 | 0.9635 | +0.5023 |
| B3 | rho | 0.2664 | 0.4645 | 0.8682 | 0.9582 | +0.4937 |
| Final | rho | 0.2734 | 0.4777 | 0.8703 | 0.9581 | +0.4804 |
| B1 | U | 0.3657 | 0.8078 | 5.5156 | 10.2411 | +9.4333 |
| B2 | U | 0.6068 | 1.3102 | 8.0569 | 14.3412 | +13.0310 |
| B3 | U | 1.0036 | 2.0155 | 10.6788 | 17.7480 | +15.7326 |
| Final | U | 1.0787 | 2.1810 | 11.2414 | 18.5801 | +16.3991 |
| B1 | A | 1.4655 | 1.7237 | 6.0305 | 10.4945 | +8.7708 |
| B2 | A | 2.4158 | 2.7943 | 8.8962 | 14.7365 | +11.9422 |
| B3 | A | 3.7670 | 4.2735 | 11.9996 | 18.3515 | +14.0780 |
| Final | A | 3.9448 | 4.4974 | 12.6126 | 19.2207 | +14.7232 |

### CIFAR-100

| Prefix | Metric | Global | n=100 | n=500 | n=2500 | Delta (2500-100) |
|---|---|---:|---:|---:|---:|---:|
| B1 | rho | 0.3482 | 0.4958 | 0.5265 | 0.9185 | +0.4227 |
| B2 | rho | 0.3541 | 0.4784 | 0.5350 | 0.9291 | +0.4507 |
| B3 | rho | 0.3208 | 0.4391 | 0.4948 | 0.9120 | +0.4730 |
| Final | rho | 0.3041 | 0.4028 | 0.4600 | 0.8871 | +0.4843 |
| B1 | U | 0.4948 | 0.6948 | 0.8479 | 4.9694 | +4.2746 |
| B2 | U | 1.0744 | 1.4036 | 1.8307 | 10.9275 | +9.5239 |
| B3 | U | 1.4248 | 1.8487 | 2.4154 | 13.8064 | +11.9578 |
| Final | U | 1.9223 | 2.3697 | 3.0949 | 16.3919 | +14.0222 |
| B1 | A | 1.4211 | 1.3921 | 1.5893 | 5.3417 | +3.9497 |
| B2 | A | 3.0339 | 2.9190 | 3.3893 | 11.6600 | +8.7410 |
| B3 | A | 4.4411 | 4.1984 | 4.8445 | 15.0146 | +10.8162 |
| Final | A | 6.3216 | 5.8739 | 6.6916 | 18.3377 | +12.4638 |

All \(n=2500-n=100\) rho differences above exclude zero under the seed-macro Student-t 95% confidence interval. The number of seeds is only three, so this is evidence of repeatability within this setup rather than a broad statistical guarantee.

## 4. Branch-pair results

Pair order follows B1-B2, B1-B3, B1-Final, B2-B3, B2-Final, B3-Final.

### CIFAR-10

| Metric / pair | n=100 | n=500 | n=2500 | Delta (2500-100) |
|---|---:|---:|---:|---:|
| Logit cosine B1-B2 | 0.8551 | 0.8193 | 0.7273 | -0.1278 |
| Logit cosine B1-B3 | 0.4574 | 0.4740 | 0.3984 | -0.0590 |
| Logit cosine B1-Final | 0.3663 | 0.3771 | 0.3486 | -0.0177 |
| Logit cosine B2-B3 | 0.6258 | 0.6201 | 0.6004 | -0.0254 |
| Logit cosine B2-Final | 0.4838 | 0.4877 | 0.5063 | +0.0225 |
| Logit cosine B3-Final | 0.8234 | 0.8005 | 0.7918 | -0.0316 |
| CKA B1-B2 | 0.8163 | 0.8351 | 0.8346 | +0.0183 |
| CKA B1-B3 | 0.3497 | 0.5027 | 0.5364 | +0.1866 |
| CKA B1-Final | 0.2180 | 0.2664 | 0.2657 | +0.0477 |
| CKA B2-B3 | 0.4995 | 0.6666 | 0.7173 | +0.2179 |
| CKA B2-Final | 0.3033 | 0.3525 | 0.3695 | +0.0663 |
| CKA B3-Final | 0.6122 | 0.6365 | 0.6357 | +0.0235 |

### CIFAR-100

| Metric / pair | n=100 | n=500 | n=2500 | Delta (2500-100) |
|---|---:|---:|---:|---:|
| Logit cosine B1-B2 | 0.6945 | 0.6877 | 0.6338 | -0.0607 |
| Logit cosine B1-B3 | 0.4777 | 0.4725 | 0.4362 | -0.0415 |
| Logit cosine B1-Final | 0.1805 | 0.1819 | 0.1782 | -0.0023 |
| Logit cosine B2-B3 | 0.8048 | 0.7968 | 0.7532 | -0.0516 |
| Logit cosine B2-Final | 0.3201 | 0.3219 | 0.3259 | +0.0058 |
| Logit cosine B3-Final | 0.4695 | 0.4708 | 0.4690 | -0.0004 |
| CKA B1-B2 | 0.7830 | 0.7805 | 0.7758 | -0.0071 |
| CKA B1-B3 | 0.5145 | 0.5155 | 0.5312 | +0.0168 |
| CKA B1-Final | 0.1490 | 0.1564 | 0.1872 | +0.0382 |
| CKA B2-B3 | 0.8207 | 0.8185 | 0.8174 | -0.0033 |
| CKA B2-Final | 0.2502 | 0.2621 | 0.3222 | +0.0720 |
| CKA B3-Final | 0.3643 | 0.3787 | 0.4509 | +0.0866 |

## 5. Interpretation

1. Increasing \(n\) increased linear CKA but generally decreased centered-logit cosine and increased directional logit variance. Therefore raw representation geometry became more similar across depths while the class directions decoded by the frozen global probes did not become more aligned.
2. Both \(U\) and \(A\) rose strongly with \(n\), and \(U\) rose faster relative to \(A\); consequently rho approached one. However, Final accuracy simultaneously fell and test CE rose sharply. Here high rho identifies a large systematic correction direction shared by test batches, not a better representation.
3. The global checkpoint is close to a test optimum, so its test-batch gradients substantially cancel. A degraded local model can have higher rho because many test batches agree on how it should be corrected. Thus rho must be reported with at least \(U\), \(A\), test CE, and accuracy.
4. Under the fixed-step budget, every condition processes 5,000 examples, but the smaller datasets are repeated more often. The n=100 local loss rapidly becomes almost zero, so later effective updates are small; n=2500 continues to produce nontrivial gradients and moves much farther from the checkpoint. The restarted local LR of 0.01 amplifies this endpoint displacement, although it is common to all n conditions.
5. These results do not directly support the statement “limited local data produces lower gradient agreement.” They support a narrower claim that local data size changes both cross-depth representation organization and the structure/magnitude of the Final-CE optimization signal. To claim that self-distillation fixes harmful gradient disagreement, a method comparison should measure CE-only versus CE+distillation gradients or downstream performance under matched endpoint quality/displacement.

## 6. Dataset-specific caveat

CIFAR-10 clients observe all 10 classes for every n. CIFAR-100 clients observe, on average, 63.23, 99.30, and 100 classes at n=100, 500, and 2500, respectively. The CIFAR-100 local-n effect therefore includes semantic/class coverage as a real component of task difficulty.
