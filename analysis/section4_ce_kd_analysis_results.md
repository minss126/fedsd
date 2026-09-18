# Section 4 — CE/KD branch analysis 결과 정리

> 작성 기준: 저장된 로그만 사용했으며 새 실험은 수행하지 않았다.  
> 공통 핵심 설정: CIFAR-100, ResNet18-BYOT, FedAvg, 100 clients, participation 0.1, local epoch 5, 500 rounds, `min_require_size=64`, feature imitation loss 미사용.  
> CE trajectory는 branch objective의 `alpha=0`, KD trajectory는 `alpha=1`을 뜻한다.

## 0. 결과 유무와 추가 실험 필요성

| Section | 요구 결과 | 상태 | 현재 범위 | 비고 |
|---|---|---|---|---|
| 4.1-1 | Branch CE/KD gradient와 Final CE gradient cosine | 완료 | T=1, IID/β=0.3/β=0.1, B1/B2/B3/Combined, seed 0 | round 50부터 500까지 50-round 간격으로 저장됨 |
| 4.1-2 | Client-local gradient와 full-test gradient cosine | 완료 | CE/KD trajectory × CE/KD gradient, 세 partition, 네 depth, seed 0 | client gradient는 실제 FedAvg weight로 결합 |
| 4.2 | Depth-specific strict linear probe | 완료 | CIFAR-100, B1/B2/B3/Final, seeds 0–1 | final CE-only, β=0.5 checkpoint 사용 |
| 4.3-1 | CE(α=0) vs KD(α=1) accuracy | 완료 | T=0.5/1, IID/β=0.3/β=0.1, 모든 branch 및 Final, seed 0 | 마지막 30 rounds 평균 |
| 4.3-2 | Rare–Frequent client JS gap | 완료 | T=0.5/1, 세 partition, B1/B2/B3/Final, seed 0 | post-local rounds 470/480/490 평균 |
| 추가 검증 | 최종 JS-client adaptive의 seed 확장 | 완료 | IID/β=0.3/β=0.1, seeds 0–2 | accuracy 및 effective λ의 last-30 평균 |

원래 요청한 Section 4의 필수 항목은 모두 확보됐다. 따라서 **본문용 seed-0 pilot 결과를 작성하는 데 필수적인 추가 실험은 없다.** 다만 gradient/CE–KD/rare–frequent 결과의 seed 1–2 반복은 아직 `MISSING`이며, 통계적 유의성을 주장하려면 후속 반복이 필요하다.

## 1. 결과 집계 규칙

- Gradient 표: 500번째 completed round의 checkpoint에서 측정했다.
  - 원본 로그에는 rounds 50, 100, ..., 500의 값이 모두 있다.
  - `Combined`는 B1/B2/B3 branch loss를 합친 gradient route이다.
- Global gradient: augmentation 없는 CIFAR-100 full test set에서 계산했다.
- Client-local gradient: 해당 round의 selected clients가 가진 전체 local data에서 계산한 뒤 실제 FedAvg aggregation weight로 합쳤다.
- Accuracy: 마지막 30 communication rounds의 평균이다.
- Rare–Frequent JS: post-local checkpoints 470/480/490에서 계산한 값의 평균이다.
  - Rare: client 내 해당 class 수가 expected count의 0.5배 미만
  - Frequent: expected count의 1.5배 초과
  - 표의 gap은 `Rare client JS − Frequent client JS`이다.
- 모든 cosine은 클수록 두 gradient 방향이 유사함을 뜻한다.

---

## 4.1 Gradient Analysis

### 4.1.1 Branch CE/KD gradient vs Final CE gradient

각 셀은 `CE branch gradient / KD branch gradient` 순서이며, 두 값 모두 full-test Final CE gradient와의 cosine similarity이다.

| Partition | Training trajectory | B1 | B2 | B3 | Combined |
|---|---|---:|---:|---:|---:|
| IID | CE-trained | 0.351 / -0.033 | 0.659 / 0.159 | 0.915 / 0.021 | 0.657 / 0.052 |
| IID | KD-trained | 0.307 / -0.144 | 0.720 / 0.007 | 0.922 / 0.045 | 0.755 / -0.028 |
| β=0.3 | CE-trained | 0.389 / 0.091 | 0.730 / 0.101 | 0.925 / 0.298 | 0.681 / 0.137 |
| β=0.3 | KD-trained | 0.016 / -0.323 | 0.596 / 0.070 | 0.870 / -0.010 | 0.689 / -0.043 |
| β=0.1 | CE-trained | 0.061 / -0.497 | 0.760 / -0.373 | 0.965 / 0.562 | 0.791 / -0.116 |
| β=0.1 | KD-trained | 0.545 / 0.243 | 0.821 / 0.429 | 0.862 / 0.138 | 0.692 / 0.221 |

해석:

- CE branch gradient는 특히 B3에서 Final CE와 높은 정렬을 보인다(0.862–0.965).
- KD gradient는 Final CE와 훨씬 덜 정렬되며, 일부 shallow/combined 조건에서는 음의 cosine도 나타난다.
- 따라서 KD는 Final CE를 단순히 반복하는 auxiliary objective가 아니라 다른 update direction을 제공한다.
- 단, 낮은 cosine 자체가 성능상 이점이라는 뜻은 아니다. 이 방향이 단순한 local bias인지 아닌지는 다음 local-to-global 분석과 함께 판단해야 한다.

### 4.1.2 Client-local gradient vs full-test gradient

각 셀은 동일 objective에 대해 `CE local↔test cosine / KD local↔test cosine` 순서이다. Local gradient는 selected clients의 gradient를 실제 FedAvg weight로 합친 값이다.

> 아래 본문 표는 `min_require_size=64`인 canonical run의 **round 500** 결과이다. 이는
> `cos(Σ_k w_k g_k^local, g^test)`이며, client별 cosine을 평균한 값과는 다르다.
> `min_require_size=10`으로 수행된 `logs_gradient_route_probe_no_feature_t1_r500`의
> β=0.1 결과는 이 표에 사용하지 않는다.

| Partition | Training trajectory | B1 | B2 | B3 | Combined |
|---|---|---:|---:|---:|---:|
| IID | CE-trained | 0.568 / 0.701 | 0.379 / 0.459 | 0.536 / 0.509 | 0.461 / 0.534 |
| IID | KD-trained | 0.783 / 0.876 | 0.456 / 0.528 | 0.645 / 0.643 | 0.578 / 0.724 |
| β=0.3 | CE-trained | 0.369 / 0.666 | 0.172 / 0.311 | 0.508 / 0.800 | 0.267 / 0.510 |
| β=0.3 | KD-trained | 0.699 / 0.784 | 0.444 / 0.616 | 0.628 / 0.831 | 0.493 / 0.636 |
| β=0.1 | CE-trained | 0.729 / 0.786 | 0.634 / 0.612 | 0.740 / 0.866 | 0.671 / 0.733 |
| β=0.1 | KD-trained | 0.880 / 0.929 | 0.738 / 0.803 | 0.799 / 0.888 | 0.823 / 0.901 |

#### Round 500 client-wise cosine 평균 (canonical min-64)

각 셀은 각 client에서 cosine을 먼저 계산한 뒤 실제 FedAvg weight로 평균한
`CE local↔test cosine / KD local↔test cosine`이다. 따라서 바로 위 표보다 낮을 수 있으며,
이는 client gradient들이 서로 상쇄된 후의 aggregate cosine이 높아지는 효과를 구분해서 보여준다.

| Partition | Training trajectory | B1 | B2 | B3 | Combined |
|---|---|---:|---:|---:|---:|
| IID | CE-trained | 0.496 / 0.611 | 0.285 / 0.343 | 0.367 / 0.332 | 0.375 / 0.443 |
| IID | KD-trained | 0.707 / 0.805 | 0.356 / 0.428 | 0.430 / 0.425 | 0.467 / 0.633 |
| β=0.1 | CE-trained | 0.447 / 0.600 | 0.344 / 0.424 | 0.396 / 0.649 | 0.360 / 0.536 |
| β=0.1 | KD-trained | 0.642 / 0.795 | 0.485 / 0.629 | 0.471 / 0.683 | 0.554 / 0.743 |

#### Rounds 50–500 checkpoint 평균 (canonical min-64)

아래는 저장된 10개 checkpoint(`50, 100, ..., 500`)에서 얻은 값을 다시 평균한 결과이다.
각 셀은 `aggregate-gradient cosine / client-wise cosine 평균` 순서이며, CE와 KD objective는
별도 열로 분리했다.

| Partition | Training trajectory | Route | CE | KD |
|---|---|---|---:|---:|
| IID | CE-trained | B1 | 0.697 / 0.609 | 0.822 / 0.732 |
| IID | CE-trained | B2 | 0.529 / 0.434 | 0.573 / 0.464 |
| IID | CE-trained | B3 | 0.667 / 0.513 | 0.668 / 0.488 |
| IID | CE-trained | Combined | 0.618 / 0.523 | 0.685 / 0.587 |
| IID | KD-trained | B1 | 0.839 / 0.774 | 0.895 / 0.841 |
| IID | KD-trained | B2 | 0.676 / 0.570 | 0.699 / 0.607 |
| IID | KD-trained | B3 | 0.762 / 0.580 | 0.806 / 0.629 |
| IID | KD-trained | Combined | 0.755 / 0.651 | 0.814 / 0.738 |
| β=0.1 | CE-trained | B1 | 0.770 / 0.510 | 0.887 / 0.692 |
| β=0.1 | CE-trained | B2 | 0.721 / 0.463 | 0.813 / 0.647 |
| β=0.1 | CE-trained | B3 | 0.738 / 0.450 | 0.842 / 0.653 |
| β=0.1 | CE-trained | Combined | 0.741 / 0.471 | 0.838 / 0.652 |
| β=0.1 | KD-trained | B1 | 0.814 / 0.550 | 0.916 / 0.765 |
| β=0.1 | KD-trained | B2 | 0.761 / 0.500 | 0.893 / 0.726 |
| β=0.1 | KD-trained | B3 | 0.811 / 0.503 | 0.921 / 0.735 |
| β=0.1 | KD-trained | Combined | 0.797 / 0.515 | 0.912 / 0.739 |

해석:

- KD local gradient는 대부분의 조건에서 같은 KD objective의 full-test gradient와 CE보다 더 잘 정렬된다.
- Combined 기준 대표값은 β=0.3 CE-trained trajectory에서 `CE 0.267 → KD 0.510`, β=0.1 KD-trained trajectory에서 `CE 0.823 → KD 0.901`이다.
- 따라서 앞 표에서 관찰된 KD의 낮은 Final-CE cosine을 곧바로 local bias로 해석하기는 어렵다. KD는 Final CE와는 다른 방향을 주면서도, 그 KD 방향 자체는 local data와 global test data 사이에서 비교적 일관될 수 있다.

### 4.1 Source 및 protocol

- Seed: 0
- Temperature: T=1
- Feature loss: 0
- Checkpoint used in tables: completed round 500
- Stored checkpoints: rounds 50, 100, ..., 500
- Run root: `logs/analysis/logs_section4_unified_no_feature_t1p0_min64`
- Example source: `logs/analysis/logs_section4_unified_no_feature_t1p0_min64/cifar100_resnet18/iid/fedavg/alpha0p00_seed0_client_pretrain_branch_freq_gradient_routes/round_0500.json`
- Training script: `scripts/experiments/analysis/run_section4_unified_ce_kd_analysis.sh`

---

## 4.2 Intermediate Branch Analysis

### 4.2.1 CIFAR-100 strict linear probe accuracy

Final CE만으로 학습한 frozen raw ResNet trunk feature에 `GAP + fresh Linear`만 학습했다. Branch bottleneck, activation, BN, backbone update는 probe에 포함하지 않았다.

| B1 | B2 | B3 | Final |
|---:|---:|---:|---:|
| 5.870 ± 0.028 | 12.230 ± 0.325 | 25.035 ± 0.516 | 71.165 ± 0.460 |

해석:

- 얕은 representation의 strict linear class separability는 Final보다 매우 낮다.
- 따라서 shallow branch에 one-hot hard-label CE를 직접 요구하면 해당 depth가 아직 제공하기 어려운 class discrimination을 강제할 수 있다는 motivation을 제공한다.
- 이 결과는 KD가 무조건 더 낫다는 직접 증거가 아니라, shallow branch에서 CE와 KD가 서로 다른 역할을 할 수 있는 구조적 근거이다.

### 4.2 Source 및 protocol

- Seeds: 0, 1; 표는 seed mean ± sample std
- Source checkpoint: β=0.5, final CE-only, branch loss off, feature loss off
- Checkpoint: global round index 499, 500 completed rounds
- Probe train: official CIFAR-100 train 50,000 samples, augmentation 없음
- Probe test: official CIFAR-100 test 10,000 samples
- Probe fitting: 30 epochs, LR 0.1, weight decay 5e-4
- Summary: `logs/analysis/logs_strict_linear_representation_probe_r500/summary.md`
- Per-run values: `logs/analysis/logs_strict_linear_representation_probe_r500/per_run.csv`

주의: 이 probe는 β=0.5 checkpoint 기반이다. 현재 Section 4의 depth motivation에는 사용할 수 있지만, IID 전용 probe라고 표기하면 안 된다.

---

## 4.3 Empirical Comparison

### 4.3.1 CE(α=0) vs KD(α=1), T=1

마지막 30 rounds의 평균 accuracy(%)이다.

| Partition | Objective | B1 | B2 | B3 | Final |
|---|---|---:|---:|---:|---:|
| IID | CE | 54.386 | 64.534 | 70.588 | 71.539 |
| IID | KD | 55.752 | 65.646 | 70.846 | 71.356 |
| β=0.3 | CE | 47.974 | 57.837 | 63.676 | 65.208 |
| β=0.3 | KD | 48.133 | 58.613 | 64.769 | 65.340 |
| β=0.1 | CE | 39.440 | 46.114 | 48.907 | 52.167 |
| β=0.1 | KD | 36.585 | 44.521 | 50.400 | 52.497 |

Final accuracy의 KD−CE 차이는 IID `-0.183`, β=0.3 `+0.132`, β=0.1 `+0.330` percentage points이다. T=1에서는 두 objective의 최종 성능 차이가 작고 혼재한다.

### 4.3.2 CE(α=0) vs KD(α=1), T=0.5

CE objective는 KD temperature를 사용하지 않으므로 CE 행은 T=1과 동일하다.

| Partition | Objective | B1 | B2 | B3 | Final |
|---|---|---:|---:|---:|---:|
| IID | CE | 54.386 | 64.534 | 70.588 | 71.539 |
| IID | KD | 51.912 | 62.880 | 70.216 | 72.560 |
| β=0.3 | CE | 47.974 | 57.837 | 63.676 | 65.208 |
| β=0.3 | KD | 44.696 | 55.042 | 63.852 | 65.651 |
| β=0.1 | CE | 39.440 | 46.114 | 48.907 | 52.167 |
| β=0.1 | KD | 34.615 | 42.358 | 51.249 | 53.282 |

Final accuracy의 KD−CE 차이는 IID `+1.021`, β=0.3 `+0.443`, β=0.1 `+1.115` percentage points이다. 흥미롭게도 T=0.5 KD는 B1/B2의 독립 분류 정확도를 낮추면서 Final accuracy를 높인다. 이는 높은 auxiliary branch accuracy 자체가 좋은 final representation의 필요조건은 아님을 보여준다.

### 4.3.3 Rare–Frequent client JS gap

각 셀은 `CE / KD` 순서이며, 값이 클수록 rare-class sample에 대한 client prediction dispersion이 frequent-class sample보다 크다는 뜻이다.

| T | Partition | B1 | B2 | B3 | Final |
|---:|---|---:|---:|---:|---:|
| 1.0 | IID | 0.012 / 0.013 | 0.020 / 0.012 | 0.015 / 0.011 | 0.017 / 0.017 |
| 1.0 | β=0.3 | 0.158 / 0.150 | 0.168 / 0.144 | 0.169 / 0.140 | 0.178 / 0.173 |
| 1.0 | β=0.1 | 0.250 / 0.232 | 0.265 / 0.238 | 0.259 / 0.225 | 0.262 / 0.259 |
| 0.5 | IID | 0.012 / 0.001 | 0.020 / 0.003 | 0.015 / -0.003 | 0.017 / 0.013 |
| 0.5 | β=0.3 | 0.158 / 0.045 | 0.168 / 0.043 | 0.169 / 0.032 | 0.178 / 0.170 |
| 0.5 | β=0.1 | 0.250 / 0.084 | 0.265 / 0.087 | 0.259 / 0.073 | 0.262 / 0.265 |

해석:

- T=1 KD는 non-IID에서 branch-level rare–frequent gap을 일관되게 조금 줄인다.
- T=0.5 KD는 branch-level gap을 크게 줄인다. 예를 들어 β=0.1 B3는 `0.259 → 0.073`이다.
- 반면 Final gap은 거의 유지된다. β=0.1에서는 T=0.5 KD가 `0.262 → 0.265`로 오히려 아주 조금 증가한다.
- 따라서 주장은 “KD가 final classifier의 모든 client bias를 제거한다”가 아니라, “KD branch supervision이 rare/frequent 조건에 따른 intermediate branch prediction dispersion을 완화한다”로 제한하는 것이 정확하다.
- IID gap은 거의 0에 가깝다. IID에서도 유한 표본의 client별 class-count 변동으로 low/high group이 형성되지만, non-IID의 rare/frequent 구조와 동일한 강도로 해석하면 안 된다.

### 4.3 Source 및 protocol

- Seed: 0
- Accuracy averaging: last 30 rounds
- JS averaging: post-local rounds 470/480/490
- Reference set: official test set에서 class당 8개로 고정된 balanced subset
- T=1 run root: `logs/analysis/logs_section4_unified_no_feature_t1p0_min64`
- T=0.5 run root: `logs/analysis/logs_section4_unified_no_feature_t0p5_min64`
- T=1 JS summary: `logs/analysis/section4_unified_t1p0_rare_frequent_rare_frequent_gap.csv`
- T=0.5 JS summary: `logs/analysis/section4_unified_t0p5_rare_frequent_rare_frequent_gap.csv`

---

## 4.4 최종 JS-client adaptive의 seed 확장 결과

이 표는 위의 CE/KD mechanism 분석과 별개로, 최종 adaptive method의 반복 안정성을 확인하기 위한 결과이다. Accuracy와 effective λ 모두 last-30 평균이다.

| Partition | Seed 0 | Seed 1 | Seed 2 | Accuracy mean ± std | Effective λ mean ± std |
|---|---:|---:|---:|---:|---:|
| IID | 72.150 | 72.270 | 72.136 | 72.186 ± 0.073 | 0.2239 ± 0.0016 |
| β=0.3 | 65.022 | 65.382 | 65.868 | 65.424 ± 0.425 | 0.1704 ± 0.0021 |
| β=0.1 | 54.697 | 53.843 | 55.142 | 54.561 ± 0.660 | 0.0985 ± 0.0061 |

Protocol:

- JS-client adaptive, feature loss 0
- KD temperature 1, proxy temperature 1
- λ max 1, soft threshold τ=0.85, warm-up 250/500 rounds
- canonical execution: `paired_resnet_init`, `paired_execution_rng`, `preserve_byot_proxy_rng` 미사용
- Seed 1–2 root: `logs/lambda/final/logs_js_client_seed12_cifar100_core`
- Seed-0 sources:
  - IID/β=0.1: `logs/lambda/adaptive/logs_js_granularity_temperature_canonical_no_feature`
  - β=0.3: `logs/lambda/adaptive/logs_js_client_lmax_crosscheck_canonical_no_feature`

해석:

- 평균 effective λ는 IID `0.224`, β=0.3 `0.170`, β=0.1 `0.099`로 이질성이 심해질수록 감소한다.
- IID와 β=0.3에서는 seed별 effective λ가 매우 안정적이다.
- β=0.1은 accuracy 표준편차가 0.660 pp로 더 크므로 severe non-IID 결과는 단일 seed보다 3-seed 평균을 사용하는 것이 적절하다.
- 이 표만으로 plain/fixed 대비 통계적 우월성을 주장할 수는 없다. 동일 seed의 baseline 분산과 paired difference가 필요하다.

---

## 5. 논문 본문에 사용할 핵심 숫자

1. **KD는 Final CE와 다른 gradient를 제공한다.**  
   IID CE-trained checkpoint의 Combined cosine은 CE `0.657`, KD `0.052`이며, β=0.1에서는 CE `0.791`, KD `-0.116`이다.

2. **이 차이가 단순한 local-only 방향 불일치로만 나타나지는 않는다.**  
   Local-to-full-test Combined cosine은 β=0.3 CE-trained trajectory에서 CE `0.267`, KD `0.510`; β=0.1 KD-trained trajectory에서 CE `0.823`, KD `0.901`이다.

3. **Shallow feature의 strict linear separability는 Final보다 현저히 낮다.**  
   CIFAR-100 probe accuracy는 B1/B2/B3/Final 각각 `5.87 / 12.23 / 25.04 / 71.17%`이다.

4. **Temperature는 KD의 empirical effect를 크게 바꾼다.**  
   Final KD−CE 차이는 T=1에서 `-0.183 / +0.132 / +0.330 pp`, T=0.5에서 `+1.021 / +0.443 / +1.115 pp`이며 순서는 IID/β=0.3/β=0.1이다.

5. **KD는 intermediate rare–frequent dispersion을 줄일 수 있다.**  
   β=0.1 B3 JS gap은 CE `0.259`에서 T=1 KD `0.225`, T=0.5 KD `0.073`으로 감소한다. Final gap은 각각 `0.262 / 0.259 / 0.265`로 거의 변하지 않는다.

6. **최종 adaptive method는 partition에 따라 λ를 실제로 다르게 배정한다.**  
   3-seed last-30 effective λ는 IID `0.224`, β=0.3 `0.170`, β=0.1 `0.099`이다.

## 6. 남은 선택적 실험

필수 추가 실험은 없지만, 논문의 주장 강도에 따라 다음 순서로 확장할 수 있다.

1. Section 4.1과 4.3의 seeds 1–2 반복: 현재 seed 0 결과의 통계적 안정성 확인
2. 동일 seed의 plain/fixed baseline까지 seed 0–2를 맞춰 adaptive paired difference 계산
3. Gradient geometry의 temperature dependence를 본문 주장으로 삼을 경우에만 T=0.5 gradient route 추가
4. Linear probe를 IID-specific 결과로 제시해야 할 경우에만 IID final-CE checkpoint로 재측정

현재 논문 구성에서는 1–4를 모두 추가하지 않아도 Section 4의 mechanism narrative는 작성 가능하다. 다만 “통계적으로 유의하게 우수하다”는 표현은 multi-seed paired baseline이 확보되기 전에는 피하는 것이 안전하다.
