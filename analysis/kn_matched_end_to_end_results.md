# $K$--$n$ matched end-to-end FedAvg 결과 분석

> 분석 대상: `logs/analysis/logs_end_to_end_kn_matched_fl/`  
> 완료 상태: CIFAR-10/100 × 3개 $n$ 조건 × 3 seeds = **18/18 완료**  
> 측정 지점: 100번째 local update 직후, aggregation 직전 local models의 client-macro 평균

## 1. 실험 설정

기존의 고정 $K=20$ 실험은 $n$이 작아질수록 global model이 접하는 전체 고유 train data도 함께 감소했다. 수정 실험은 모든 조건에서

$$
K\times n=50{,}000
$$

을 유지해 이 문제를 제거했다.

| Local samples $n$ | Clients $K$ | Steps/client/round | Total client steps/round | Processed samples/round |
|---:|---:|---:|---:|---:|
| 100 | 500 | 10 | 5,000 | 250,000 |
| 500 | 100 | 50 | 5,000 | 250,000 |
| 2,500 | 20 | 250 | 5,000 | 250,000 |

- IID, client 간 disjoint partition, full participation
- 각 client는 5 local epochs, batch size 50
- 모든 조건에서 매 round 전체 50,000개 고유 sample을 각각 5회 처리
- 100 independent FL rounds, cosine LR ($0.1\rightarrow0$)
- Branch objective 없이 Final CE만 사용
- CIFAR-10/100, seeds 0/1/2
- Linear probe는 각 trajectory의 round-start global model에서 전체 official train set으로 학습
- Logit과 CKA는 전체 official test set에서 측정

따라서 이번 결과는 **전체 data coverage나 총 처리량 차이**가 아니라, $n$과 $K$가 함께 바뀌는 실제 FL의 client granularity를 비교한다. 단, $n$만의 독립적인 인과 효과는 아니다. $n$이 작아질수록 $K$가 커지고 client당 연속 local step 수가 짧아지는 효과가 함께 포함된다.

## 2. 핵심 aggregate 결과

아래 값은 먼저 한 seed 안에서 모든 client를 평균하고, 그 seed-level mean 세 개를 다시 평균한 값이다. $Delta$는

$$
\Delta_{2500-100}=M(n=2500)-M(n=100)
$$

이며, `±`는 seed-matched difference의 Student-$t$ 95% CI half-width이다.

### 2.1 CIFAR-10

| Metric | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---:|---:|---:|---:|
| Mean centered-logit cosine ↑ | 0.8134 | 0.7130 | 0.6477 | **-0.1657 ± 0.0259** |
| Directional logit variance ↓ | 0.1399 | 0.2152 | 0.2642 | **+0.1243 ± 0.0194** |
| Mean linear CKA ↑ | 0.5579 | 0.5708 | 0.5051 | -0.0528 ± 0.1394 |

### 2.2 CIFAR-100

| Metric | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---:|---:|---:|---:|
| Mean centered-logit cosine ↑ | 0.7146 | 0.5950 | 0.5143 | **-0.2003 ± 0.0067** |
| Directional logit variance ↓ | 0.2141 | 0.3038 | 0.3643 | **+0.1502 ± 0.0051** |
| Mean linear CKA ↑ | 0.5873 | 0.5364 | 0.4845 | **-0.1028 ± 0.0297** |

Directional variance는 네 depth의 중심화·정규화된 logit direction이 흩어진 정도다. 네 depth를 사용할 때 mean pairwise cosine과

$$
V_{\mathrm{dir}}=\frac{3}{4}(1-\overline{\cos})
$$

관계이므로 두 지표는 독립적인 증거가 아니라 같은 현상의 반대 표현이다.

### 2.3 1차 해석

- 두 데이터셋 모두 $n$이 증가하고 $K$가 감소할수록 logit cosine은 일관되게 감소하고 directional variance는 증가했다.
- 즉 작은 $n$/큰 $K$ 조건에서 depth들의 class-semantic output이 오히려 더 비슷했다.
- CKA는 CIFAR-100에서 같은 감소가 명확했지만, CIFAR-10에서는 $n=500$이 가장 높고 endpoint CI도 0을 포함했다. 따라서 CIFAR-10의 aggregate CKA를 단조 경향으로 주장할 수 없다.

## 3. 정확도와 alignment의 관계

### 3.1 100-round final aggregated global model의 native accuracy

| Dataset | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ |
|---|---:|---:|---:|
| CIFAR-10 | 79.75 ± 2.31 | 89.66 ± 0.84 | 94.02 ± 0.19 |
| CIFAR-100 | 41.42 ± 1.19 | 60.30 ± 2.03 | 73.02 ± 0.59 |

각 값은 세 seed 평균 ± 95% CI이다. $n=100$에서 $n=2,500$으로 갈 때 native accuracy는 CIFAR-10에서 약 **14.26%p**, CIFAR-100에서 약 **31.60%p** 증가한다. 반면 같은 구간에서 logit alignment는 감소한다.

따라서 이번 실험에서도 높은 cross-depth alignment를 좋은 representation 또는 좋은 task performance로 해석할 수 없다. 작은 $n$/큰 $K$에서는 Final이 충분히 분화·전문화되지 않아 depth output이 서로 비슷하게 남으면서도 정확도는 낮을 수 있다.

### 3.2 Frozen-probe accuracy

| Dataset | Depth | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | B1 | 47.91 | 38.93 | 34.37 | **-13.54 ± 1.76** |
| CIFAR-10 | B2 | 60.74 | 52.39 | 47.06 | **-13.67 ± 1.99** |
| CIFAR-10 | B3 | 72.10 | 72.83 | 79.00 | **+6.90 ± 5.15** |
| CIFAR-10 | Final | 79.78 | 89.65 | 93.97 | **+14.20 ± 1.95** |
| CIFAR-100 | B1 | 12.76 | 6.00 | 4.33 | **-8.43 ± 0.37** |
| CIFAR-100 | B2 | 17.99 | 12.13 | 10.65 | **-7.34 ± 0.93** |
| CIFAR-100 | B3 | 27.93 | 20.99 | 22.89 | **-5.04 ± 2.73** |
| CIFAR-100 | Final | 42.38 | 60.25 | 72.71 | **+30.33 ± 1.64** |

Final-only CE로 학습했기 때문에 Final의 성능은 $n$과 함께 크게 향상되지만 B1/B2의 선형 분류 가능성은 오히려 감소한다. 즉 더 잘 학습된 trajectory에서는 class information이 모든 depth에 균일하게 존재하는 것이 아니라 network 후반부에 더 강하게 조직된다. 작은 $n$/큰 $K$의 높은 logit alignment는 shallow branch가 좋아져서라기보다 depth hierarchy가 덜 분화된 결과라는 해석과 일치한다.

## 4. Branch-pair별 centered-logit cosine

### 4.1 CIFAR-10

| Pair | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---:|---:|---:|---:|
| B1--B2 | 0.8549 | 0.8981 | 0.9165 | +0.0616 ± 0.0646 |
| B1--B3 | 0.7419 | 0.6565 | 0.5980 | **-0.1439 ± 0.0826** |
| B2--B3 | 0.9201 | 0.8197 | 0.7343 | **-0.1858 ± 0.1362** |
| B1--Final | 0.6630 | 0.4895 | 0.3800 | **-0.2830 ± 0.0624** |
| B2--Final | 0.8028 | 0.6066 | 0.4757 | **-0.3271 ± 0.0151** |
| B3--Final | 0.8977 | 0.8078 | 0.7817 | **-0.1159 ± 0.0608** |

CIFAR-10에서는 B1--B2만 반대 방향이며 CI도 0을 포함한다. Aggregate cosine 감소는 주로 B1/B2--Final과 B2--B3 감소가 만든다.

### 4.2 CIFAR-100

| Pair | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---:|---:|---:|---:|
| B1--B2 | 0.9129 | 0.8662 | 0.7088 | **-0.2040 ± 0.0286** |
| B1--B3 | 0.7137 | 0.6654 | 0.5469 | **-0.1668 ± 0.0339** |
| B2--B3 | 0.8651 | 0.8577 | 0.8326 | **-0.0324 ± 0.0129** |
| B1--Final | 0.4813 | 0.2876 | 0.2096 | **-0.2717 ± 0.0060** |
| B2--Final | 0.5814 | 0.3805 | 0.3318 | **-0.2497 ± 0.0053** |
| B3--Final | 0.7331 | 0.5125 | 0.4559 | **-0.2771 ± 0.0236** |

CIFAR-100에서는 여섯 pair가 모두 단조 감소하고 endpoint CI도 모두 0을 포함하지 않는다. 특히 모든 branch--Final pair의 감소가 크다.

## 5. Branch-pair별 linear CKA

### 5.1 CIFAR-10

| Pair | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---:|---:|---:|---:|
| B1--B2 | 0.8909 | 0.8695 | 0.8958 | +0.0049 ± 0.1736 |
| B1--B3 | 0.6373 | 0.5621 | 0.4822 | -0.1550 ± 0.5041 |
| B2--B3 | 0.7805 | 0.7372 | 0.6056 | -0.1749 ± 0.2068 |
| B1--Final | 0.3332 | 0.2847 | 0.2232 | **-0.1101 ± 0.0814** |
| B2--Final | 0.4079 | 0.3721 | 0.2820 | **-0.1259 ± 0.0751** |
| B3--Final | 0.2977 | 0.5993 | 0.5416 | **+0.2439 ± 0.0689** |

CIFAR-10 CKA는 pair마다 방향이 다르다. 특히 B3--Final은 $n=500$에서 급증하며, B1/B2--Final과 반대 방향이다. Aggregate endpoint CI도 0을 포함하므로 “$n$이 커질수록 CKA가 감소한다”는 단일 문장으로 요약하면 안 된다. $n=100$의 aggregate CKA는 seed 2의 shallow-pair 값이 다른 seeds보다 낮아 seed 변동도 크다.

### 5.2 CIFAR-100

| Pair | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---:|---:|---:|---:|
| B1--B2 | 0.8897 | 0.8812 | 0.7918 | **-0.0979 ± 0.0854** |
| B1--B3 | 0.5656 | 0.6192 | 0.5918 | +0.0262 ± 0.0477 |
| B2--B3 | 0.7740 | 0.8265 | 0.8390 | +0.0650 ± 0.0910 |
| B1--Final | 0.2772 | 0.2164 | 0.1531 | **-0.1241 ± 0.0384** |
| B2--Final | 0.4093 | 0.2763 | 0.2200 | **-0.1893 ± 0.0641** |
| B3--Final | 0.6080 | 0.3985 | 0.3111 | **-0.2969 ± 0.0174** |

CIFAR-100의 aggregate CKA 감소는 주로 세 branch--Final pair에서 발생한다. Shallow--shallow CKA는 일부 증가하거나 비단조이며 endpoint CI가 0을 포함한다. 따라서 feature geometry에서도 핵심은 모든 layer가 일괄적으로 멀어지는 것이 아니라 **Final과 intermediate depth 사이의 hierarchy가 커지는 현상**이다.

## 6. 마지막 local update와 누적 global trajectory의 분리

100번째 round의 round-start global model과 post-local client 평균을 비교하면 차이가 매우 작다.

| Dataset | Metric | $n=100$ | $n=500$ | $n=2,500$ |
|---|---|---:|---:|---:|
| CIFAR-10 | Post-local − global cosine | -0.00069 | -0.00035 | -0.00020 |
| CIFAR-10 | Post-local − global variance | +0.00052 | +0.00027 | +0.00015 |
| CIFAR-10 | Post-local − global CKA | -0.00003 | -0.00041 | -0.00073 |
| CIFAR-100 | Post-local − global cosine | -0.00108 | -0.00038 | -0.00036 |
| CIFAR-100 | Post-local − global variance | +0.00081 | +0.00029 | +0.00027 |
| CIFAR-100 | Post-local − global CKA | +0.00112 | +0.00013 | -0.00002 |

이는 마지막 round LR가 약 $2.47\times10^{-5}$이기 때문이다. 따라서 final pre-aggregation local 결과의 절대 차이는 마지막 local update가 즉시 만든 차이라기보다, 서로 다른 $K/n$ 조건에서 99회의 aggregation을 거쳐 누적된 **trajectory-specific global state**를 거의 그대로 반영한다.

이 점은 설계 오류가 아니다. 이번 실험의 질문은 “같은 checkpoint에서 한 번의 local adaptation만 바꾸면 무엇이 달라지는가?”가 아니라 “처음부터 client granularity가 다른 실제 FL system이 최종적으로 어떤 pre-aggregation local representation을 만드는가?”이기 때문이다. 전자의 순수 local mechanism은 동일-checkpoint 실험이 담당한다.

## 7. 기존 고정 $K=20$ 결과와 비교

$n=100$에서 전체 data coverage를 2,000개에서 50,000개로 수정한 효과는 다음과 같다. $n=2,500$은 두 설계 모두 $K=20$이고 전체 50,000개를 사용하므로 동일한 기준점이다.

| Dataset | Design at $n=100$ | Final global acc. | Logit cosine | CKA |
|---|---|---:|---:|---:|
| CIFAR-10 | 기존: $K=20$, total data 2,000 | 55.86 | 0.8499 | 0.5351 |
| CIFAR-10 | 수정: $K=500$, total data 50,000 | 79.75 | 0.8134 | 0.5579 |
| CIFAR-100 | 기존: $K=20$, total data 2,000 | 16.40 | 0.7665 | 0.5908 |
| CIFAR-100 | 수정: $K=500$, total data 50,000 | 41.42 | 0.7146 | 0.5873 |

Coverage를 맞추자 작은-$n$ 조건의 정확도는 크게 개선되고 logit cosine 격차도 줄었다. 그러나 $n=100$의 cosine은 여전히 $n=2,500$보다 뚜렷하게 높았고, CIFAR-100 CKA도 같은 방향을 유지했다. 따라서 이전의 반대 경향 전체를 data coverage confound로 설명할 수는 없다. **많은 client의 짧은 local trajectory를 평균하는 FL dynamics 자체**가 depth specialization을 억제하거나 지연시키는 효과가 남아 있다는 해석이 더 타당하다.

## 8. 논문에서 사용할 수 있는 결론

### 직접 지지되는 결론

1. 전체 50,000개 IID data와 라운드당 총 sample exposure를 통제해도 client granularity에 따라 최종 pre-aggregation local model의 depth 관계가 크게 달라진다.
2. 작은 $n$/큰 $K$에서는 logit alignment가 높지만 Final accuracy가 낮다. 따라서 높은 alignment는 좋은 representation의 충분조건이 아니다.
3. 큰 $n$/작은 $K$에서는 Final accuracy가 향상되는 동시에 branch--Final logit cosine이 감소한다. 이는 학습이 진행될수록 task information이 depth별로 분화되는 현상과 일치한다.
4. CIFAR-100에서는 branch--Final CKA도 모두 감소해 feature geometry 수준의 hierarchy 확대가 확인된다.
5. CIFAR-10 CKA는 pair와 seed에 따라 혼합되므로 logit 결과와 동일한 보편적 결론으로 제시해서는 안 된다.

### 직접 지지되지 않는 결론

- “Local sample 수만의 순수 효과”라고 부를 수 없다. 이 실험에서는 $n$과 $K$가 구조적으로 결합돼 있다.
- “Local data가 작으면 depth discrepancy가 커진다”는 단순 motivation은 결과와 반대다.
- “Cross-depth alignment를 높이는 것 자체가 self-distillation의 목적”이라고 주장하면 작은-$n$ 조건의 낮은 성능·높은 alignment와 충돌한다.

### Self-distillation motivation에 맞춘 권장 해석

> Fine-grained FL with many small clients can yield highly aligned yet under-specialized depth-wise predictions, whereas longer local trajectories improve the final predictor while increasing its semantic separation from intermediate representations. This shows that cross-depth similarity alone does not guarantee task-relevant knowledge at shallow depths. Within-model self-distillation should therefore be motivated as transferring the final predictor's task-relevant semantics to intermediate branches while preserving useful depth specialization, rather than merely maximizing representation alignment.

즉 method 결과에서는 alignment만 제시하지 말고 Final accuracy, branch accuracy, 그리고 branch--Final semantic transfer를 함께 보여줘야 한다.

## 9. 결과 파일

- CIFAR-10 summary: `logs/analysis/logs_end_to_end_kn_matched_fl/cifar10/fixed_epoch/summary.json`
- CIFAR-100 summary: `logs/analysis/logs_end_to_end_kn_matched_fl/cifar100/fixed_epoch/summary.json`
- Seed별 geometry: `logs/analysis/logs_end_to_end_kn_matched_fl/{dataset}/fixed_epoch/sample_{n}/seed_{seed}/teacher_only_end_to_end_local_n_postlocal_internal_geometry.json`
- Seed별 terminal log 및 final checkpoint: 동일한 seed directory 아래 `*_terminal.log`, `*_final.pt`

