# FL의 local data 규모와 network 내부 representation 분석

> 작성 목적: 논문의 **Analysis / Motivation** 절에 사용할 결과와 해석을 한 문서에 정리한다.  
> 작성 기준일: 2026-08-17  
> 상태: $K\times n=50{,}000$으로 보정한 end-to-end 재실험까지 18/18 trajectory가 완료되어 포함했다.

## 1. 분석 질문

본 분석은 non-IID에 의한 client 간 차이보다, **aggregation 직전 하나의 local model 내부에서 depth별 representation이 어떻게 달라지는가**에 초점을 둔다. 구체적인 질문은 다음과 같다.

1. Client가 이용하는 local sample 수 $n$이 달라지면 B1, B2, B3, Final 사이의 representation 관계가 달라지는가?
2. 이러한 변화는 feature geometry를 비교하는 CKA와 class-semantic output을 비교하는 logit cosine에서 동일하게 나타나는가?
3. 관찰된 경향이 CIFAR-10과 CIFAR-100, fixed-step과 fixed-epoch, 그리고 서로 다른 학습 단계에서 재현되는가?
4. 높은 cross-depth alignment를 곧바로 좋은 representation으로 해석할 수 있는가?

핵심 결론은 다음과 같다.

> 성숙한 동일 checkpoint에서 local data 규모가 증가할수록 cross-depth CKA는 대체로 증가했다. 반면 실제 $K$--$n$ matched FL에서는 작은 $n$/큰 $K$ 조건이 더 높은 logit alignment와 더 낮은 Final accuracy를 만들었고, CIFAR-100 CKA도 같은 방향을 보였다. 이는 한 번의 local adaptation 효과와 누적 FL trajectory 효과를 구분해야 하며, alignment의 절대값을 representation quality와 동일시할 수 없음을 보여준다.

## 2. 실험 설정과 표기

### 2.1 Network와 depth

Backbone은 CIFAR용 ResNet-18 BYOT 구조이며, 분석 depth는 다음 네 개이다.

- B1: 가장 shallow한 auxiliary branch 위치
- B2: 두 번째 auxiliary branch 위치
- B3: 가장 deep한 auxiliary branch 위치
- Final: 최종 classifier 위치

Motivation 분석에서는 branch objective를 비활성화하고 Final CE만으로 backbone을 학습했다. 따라서 측정된 branch feature는 self-distillation의 결과가 아니라, 일반적인 Final-only 학습에서 자연스럽게 형성된 intermediate representation이다.

### 2.2 공통 reference와 probe

- Feature와 logit 측정: 전체 official test set
- Linear probe 학습: augmentation을 제거한 전체 official train set
- Probe를 학습할 때 backbone parameter는 고정
- 동일 checkpoint 비교에서는 global checkpoint에서 depth별 probe를 학습한 뒤 모든 local model에 고정하여 적용

따라서 probe가 전체 train set을 사용한다는 것은 **representation을 읽는 진단 classifier**가 전체 데이터를 사용한다는 의미다. Local backbone 자체는 각 조건의 $n$개 local sample만 이용해 업데이트된다.

### 2.3 Fixed-step과 fixed-epoch

Batch size는 50이다.

- Fixed-step: 모든 $n$에서 100 optimizer steps, 즉 5,000 processed examples
- Fixed-epoch: 모든 $n$에서 5 local epochs

Fixed-epoch의 optimizer step 수는 다음과 같다.

| Local samples $n$ | Steps per local update |
|---:|---:|
| 100 | 10 |
| 250 | 25 |
| 500 | 50 |
| 1,000 | 100 |
| 2,500 | 250 |

Fixed-step은 optimizer budget을 통제하면서 고유 sample diversity의 영향을 본다. Fixed-epoch은 각 sample을 같은 횟수만큼 사용하는 실제 FedAvg local training에 더 가깝지만, $n$과 optimizer step 수가 함께 변한다.

### 2.4 통계 집계

동일-checkpoint local diagnostic은 sampling seed 3개와 seed당 local fork 10개를 사용한다. 먼저 seed 내부에서 client/fork 평균을 계산하고, 그다음 seed-level mean 세 개를 macro-average한다. 표의 $95\%$ CI는 seed-level paired difference에 대한 양측 Student-$t$ interval이다.

Paired CI가 0을 포함하면 관찰된 세 seed만으로 변화 방향이 안정적으로 양수 또는 음수라고 결론 내리기 어렵다는 뜻이다. 다만 seed가 3개뿐이므로 CI가 넓을 수 있으며, `0 포함`을 효과가 정확히 없다는 증거로 해석하지 않는다.

표에서

$$
\Delta_{2500-100}=M(n=2500)-M(n=100)
$$

로 정의한다. 따라서 CKA와 cosine의 양의 Δ는 $n$이 증가할수록 alignment가 증가했다는 뜻이고, variance의 양의 Δ는 depth 간 차이가 증가했다는 뜻이다.

## 3. 측정 지표

### 3.1 Pairwise centered-logit cosine

Depth $b$의 sample $i$에 대한 logit을 $z_{i,b}\in\mathbb{R}^{C}$라고 하자. Class-independent offset을 제거하고 방향만 비교하기 위해 다음과 같이 중심화하고 정규화한다.

$$
\widetilde z_{i,b}
=z_{i,b}-\frac{1}{C}\mathbf{1}\mathbf{1}^{\top}z_{i,b},
\qquad
u_{i,b}=\frac{\widetilde z_{i,b}}{\lVert\widetilde z_{i,b}\rVert_2}.
$$

Depth pair $(b,b')$의 centered-logit cosine은

$$
S_{b,b'}
=\frac{1}{N}\sum_{i=1}^{N}u_{i,b}^{\top}u_{i,b'}
$$

로 계산한다. 값이 클수록 두 depth가 같은 sample에 대해 비슷한 class 방향을 출력한다. Aggregate cosine은 여섯 pair

$$
\mathcal P=\{\text{B1--B2},\text{B1--B3},\text{B2--B3},
\text{B1--Final},\text{B2--Final},\text{B3--Final}\}
$$

의 평균이다.

### 3.2 Directional logit variance

Sample $i$에서 네 depth의 단위 logit direction 평균을

$$
\bar u_i=\frac{1}{4}\sum_{b=1}^{4}u_{i,b}
$$

라고 하면 directional variance는

$$
V_{\mathrm{dir}}
=\frac{1}{N}\sum_{i=1}^{N}
\frac{1}{4}\sum_{b=1}^{4}\lVert u_{i,b}-\bar u_i\rVert_2^2
$$

이다. 값이 클수록 네 depth의 class direction이 서로 다르다.

현재처럼 네 개의 중심화된 단위 logit vector를 사용하면 aggregate cosine $\bar S$와 directional variance는 다음 관계를 정확히 만족한다.

$$
V_{\mathrm{dir}}=\frac{3}{4}(1-\bar S).
$$

따라서 directional variance는 cosine과 독립적인 증거가 아니라, 같은 현상을 반대 방향으로 나타낸 보조 지표다.

### 3.3 Linear CKA

같은 $N$개 sample에 대한 두 depth의 centered feature matrix를 $X\in\mathbb{R}^{N\times d_x}$, $Y\in\mathbb{R}^{N\times d_y}$라고 하자. Linear CKA는

$$
\operatorname{CKA}(X,Y)
=\frac{\lVert X^{\top}Y\rVert_F^2}
{\lVert X^{\top}X\rVert_F\,\lVert Y^{\top}Y\rVert_F}
$$

로 계산한다. CKA는 feature axis의 회전이나 차원 차이에 비교적 불변이며, 두 depth가 sample 사이의 관계를 얼마나 유사하게 구성하는지를 측정한다.

Logit cosine과 CKA의 차이는 다음과 같다.

| Metric | 비교 대상 | 직접 반영하는 것 |
|---|---|---|
| Centered-logit cosine | Depth별 probe가 출력한 class direction | 선형적으로 읽히는 class-semantic output |
| Linear CKA | Raw feature가 형성한 sample 관계 | 전체 representation geometry |

초기에 측정했던 within-class/between-class feature variance는 이후 CIFAR-10/100 및 독립 round protocol에서 동일하게 반복되지 않았고, 본 분석의 최종 세 지표에도 포함되지 않았다. 따라서 protocol이 일치하는 logit cosine, directional variance, CKA를 본문의 정량 근거로 사용하고 W/B 결과는 이 통합 문서에서 제외한다.

따라서 CKA가 높더라도 probe가 읽는 class direction은 다를 수 있고, 반대로 일부 class direction이 비슷하더라도 전체 feature geometry는 다를 수 있다.

## 4. Global checkpoint의 학습 단계

서로 독립적으로 학습한 $R\in\{0,10,50,100\}$ checkpoint를 비교했다. 각 endpoint는 자신의 round budget을 horizon으로 하는 cosine LR를 사용했다.

| Dataset | Rounds | Mean logit cosine | Directional variance | Mean CKA | Probe accuracy B1/B2/B3/Final (%) |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 0 | 0.8662 | 0.1003 | 0.8609 | 29.2 / 27.1 / 24.1 / 21.6 |
| CIFAR-10 | 10 | 0.7986 | 0.1511 | 0.6934 | 41.7 / 53.5 / 66.1 / 75.3 |
| CIFAR-10 | 50 | 0.6982 | 0.2264 | 0.5574 | 37.6 / 50.6 / 73.6 / 92.4 |
| CIFAR-10 | 100 | 0.6644 | 0.2517 | 0.5245 | 35.7 / 48.0 / 76.7 / 94.1 |
| CIFAR-100 | 0 | 0.7958 | 0.1532 | 0.8667 | 6.4 / 5.0 / 3.9 / 3.4 |
| CIFAR-100 | 10 | 0.7172 | 0.2121 | 0.6799 | 7.8 / 12.0 / 19.3 / 39.7 |
| CIFAR-100 | 50 | 0.5785 | 0.3162 | 0.5231 | 6.4 / 13.2 / 23.8 / 69.9 |
| CIFAR-100 | 100 | 0.5445 | 0.3416 | 0.4920 | 4.8 / 12.0 / 24.8 / 72.9 |

두 데이터셋 모두 학습이 진행될수록 Final accuracy는 크게 증가하지만, global checkpoint의 cross-depth cosine과 CKA는 감소했다. 특히 무작위 초기화에 해당하는 $R=0$에서 두 alignment 지표가 가장 높다. 이는 높은 alignment가 항상 좋은 representation을 의미하지 않음을 직접 보여준다. 학습이 진행되면 Final depth가 class-discriminative한 geometry를 형성하면서 shallow depth와 역할이 분화되기 때문이다.

따라서 이후 결과에서 alignment 증가는 반드시 accuracy 또는 probe quality와 함께 해석해야 한다.

## 5. 동일 FL checkpoint에서 local sample 수를 바꾼 통제 실험

### 5.1 Aggregate absolute values at mature checkpoints

#### Mean pairwise centered-logit cosine

| Round | Dataset / budget | $n=100$ | 250 | 500 | 1,000 | 2,500 |
|---:|---|---:|---:|---:|---:|---:|
| 50 | CIFAR-10 / fixed-step | 0.6989 | 0.6975 | 0.6971 | 0.6951 | 0.6896 |
| 50 | CIFAR-100 / fixed-step | 0.5777 | 0.5785 | 0.5807 | 0.5843 | 0.5839 |
| 50 | CIFAR-10 / fixed-epoch | 0.6961 | 0.6933 | 0.6974 | 0.6941 | 0.6952 |
| 50 | CIFAR-100 / fixed-epoch | 0.5756 | 0.5746 | 0.5772 | 0.5846 | 0.5883 |
| 100 | CIFAR-10 / fixed-step | 0.6656 | 0.6634 | 0.6573 | 0.6585 | 0.6619 |
| 100 | CIFAR-100 / fixed-step | 0.5445 | 0.5453 | 0.5477 | 0.5481 | 0.5424 |
| 100 | CIFAR-10 / fixed-epoch | 0.6641 | 0.6603 | 0.6553 | 0.6493 | 0.6590 |
| 100 | CIFAR-100 / fixed-epoch | 0.5425 | 0.5419 | 0.5445 | 0.5475 | 0.5384 |

#### Directional logit variance

| Round | Dataset / budget | $n=100$ | 250 | 500 | 1,000 | 2,500 |
|---:|---|---:|---:|---:|---:|---:|
| 50 | CIFAR-10 / fixed-step | 0.2258 | 0.2269 | 0.2272 | 0.2287 | 0.2328 |
| 50 | CIFAR-100 / fixed-step | 0.3167 | 0.3161 | 0.3145 | 0.3118 | 0.3121 |
| 50 | CIFAR-10 / fixed-epoch | 0.2279 | 0.2301 | 0.2269 | 0.2294 | 0.2286 |
| 50 | CIFAR-100 / fixed-epoch | 0.3183 | 0.3191 | 0.3171 | 0.3116 | 0.3088 |
| 100 | CIFAR-10 / fixed-step | 0.2508 | 0.2524 | 0.2571 | 0.2562 | 0.2536 |
| 100 | CIFAR-100 / fixed-step | 0.3417 | 0.3410 | 0.3392 | 0.3389 | 0.3432 |
| 100 | CIFAR-10 / fixed-epoch | 0.2519 | 0.2548 | 0.2585 | 0.2630 | 0.2557 |
| 100 | CIFAR-100 / fixed-epoch | 0.3431 | 0.3436 | 0.3416 | 0.3393 | 0.3462 |

#### Mean pairwise linear CKA

| Round | Dataset / budget | $n=100$ | 250 | 500 | 1,000 | 2,500 |
|---:|---|---:|---:|---:|---:|---:|
| 50 | CIFAR-10 / fixed-step | 0.5662 | 0.5682 | 0.5783 | 0.5920 | 0.6028 |
| 50 | CIFAR-100 / fixed-step | 0.5362 | 0.5391 | 0.5452 | 0.5555 | 0.5711 |
| 50 | CIFAR-10 / fixed-epoch | 0.5581 | 0.5575 | 0.5714 | 0.5933 | 0.5971 |
| 50 | CIFAR-100 / fixed-epoch | 0.5268 | 0.5341 | 0.5428 | 0.5556 | 0.5617 |
| 100 | CIFAR-10 / fixed-step | 0.5313 | 0.5340 | 0.5523 | 0.5622 | 0.5671 |
| 100 | CIFAR-100 / fixed-step | 0.5015 | 0.5046 | 0.5135 | 0.5338 | 0.5613 |
| 100 | CIFAR-10 / fixed-epoch | 0.5252 | 0.5231 | 0.5282 | 0.5610 | 0.5743 |
| 100 | CIFAR-100 / fixed-epoch | 0.4944 | 0.4992 | 0.5106 | 0.5344 | 0.5493 |

성숙한 $R=50,100$ checkpoint에서는 CKA가 $n$과 함께 증가하는 경향이 두 데이터셋과 두 budget에서 가장 일관되었다. 반면 aggregate logit cosine은 변화 폭이 작고 방향도 데이터셋 및 checkpoint에 따라 달랐다.

### 5.2 Endpoint change across independently trained rounds

아래 표는 각 독립 endpoint 내부에서 계산한 paired $\Delta_{2500-100}$이다. `±` 뒤의 값은 $95\%$ CI half-width이다.

| Round | Dataset / budget | Δ logit cosine | Δ directional variance | Δ CKA |
|---:|---|---:|---:|---:|
| 0 | CIFAR-10 / fixed-step | -0.0171 ± 0.0074 | +0.0128 ± 0.0056 | +0.0491 ± 0.0594 |
| 0 | CIFAR-100 / fixed-step | -0.0039 ± 0.0143 | +0.0029 ± 0.0107 | -0.0268 ± 0.0116 |
| 0 | CIFAR-10 / fixed-epoch | +0.4692 ± 0.0149 | -0.3519 ± 0.0111 | -0.1900 ± 0.0452 |
| 0 | CIFAR-100 / fixed-epoch | +0.4190 ± 0.0136 | -0.3142 ± 0.0102 | -0.2008 ± 0.0216 |
| 10 | CIFAR-10 / fixed-step | +0.0180 ± 0.0112 | -0.0135 ± 0.0084 | +0.0164 ± 0.0298 |
| 10 | CIFAR-100 / fixed-step | +0.0127 ± 0.0087 | -0.0096 ± 0.0065 | +0.0052 ± 0.0100 |
| 10 | CIFAR-10 / fixed-epoch | -0.0168 ± 0.0301 | +0.0126 ± 0.0226 | -0.0131 ± 0.0236 |
| 10 | CIFAR-100 / fixed-epoch | -0.0099 ± 0.0056 | +0.0074 ± 0.0042 | -0.0204 ± 0.0111 |
| 50 | CIFAR-10 / fixed-step | -0.0093 ± 0.0062 | +0.0070 ± 0.0046 | +0.0366 ± 0.0115 |
| 50 | CIFAR-100 / fixed-step | +0.0062 ± 0.0060 | -0.0047 ± 0.0045 | +0.0349 ± 0.0135 |
| 50 | CIFAR-10 / fixed-epoch | -0.0009 ± 0.0035 | +0.0007 ± 0.0027 | +0.0390 ± 0.0057 |
| 50 | CIFAR-100 / fixed-epoch | +0.0127 ± 0.0028 | -0.0095 ± 0.0021 | +0.0350 ± 0.0019 |
| 100 | CIFAR-10 / fixed-step | -0.0037 ± 0.0193 | +0.0028 ± 0.0145 | +0.0358 ± 0.0349 |
| 100 | CIFAR-100 / fixed-step | -0.0020 ± 0.0132 | +0.0015 ± 0.0099 | +0.0598 ± 0.0021 |
| 100 | CIFAR-10 / fixed-epoch | -0.0051 ± 0.0145 | +0.0039 ± 0.0109 | +0.0491 ± 0.0145 |
| 100 | CIFAR-100 / fixed-epoch | -0.0041 ± 0.0123 | +0.0031 ± 0.0092 | +0.0549 ± 0.0062 |

해석은 다음과 같다.

1. $R=0$은 initialization control이다. Random feature와 $n$에 따라 크게 다른 fixed-epoch step 수가 결합되므로 trained endpoint의 근거로 사용하지 않는다.
2. $R=10$에서는 CKA 경향도 아직 안정적이지 않다.
3. $R=50,100$에서는 모든 dataset/budget 조합에서 CKA endpoint Δ가 양수다. 특히 $R=50$의 네 조건과 $R=100$의 CIFAR-10 fixed-epoch 및 CIFAR-100 두 조건은 CI가 0을 포함하지 않는다.
4. Aggregate logit cosine은 같은 수준의 재현성을 보이지 않는다. 이는 다음 branch-pair 결과처럼 서로 반대 방향의 pair가 평균에서 상쇄되기 때문이다.

### 5.3 Branch-pair endpoint change

아래 값은 $n=2500$ minus $n=100$이다. 표의 pair 순서는 B1--B2, B1--B3, B2--B3, B1--Final, B2--Final, B3--Final이다. 표의 폭을 줄이기 위해 mean만 표시했으며, pair별 CI는 원본 결과 파일에 보존되어 있다.

#### Centered-logit cosine Δ

| R | Dataset / budget | B1--B2 | B1--B3 | B2--B3 | B1--Final | B2--Final | B3--Final |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | CIFAR-10 / fixed-step | -0.0351 | -0.0277 | -0.0185 | +0.0113 | +0.0197 | -0.0056 |
| 50 | CIFAR-100 / fixed-step | -0.0069 | +0.0030 | -0.0064 | +0.0150 | +0.0168 | +0.0158 |
| 50 | CIFAR-10 / fixed-epoch | -0.0354 | -0.0041 | +0.0014 | +0.0137 | +0.0227 | -0.0040 |
| 50 | CIFAR-100 / fixed-epoch | -0.0030 | +0.0084 | +0.0011 | +0.0206 | +0.0251 | +0.0238 |
| 100 | CIFAR-10 / fixed-step | -0.0558 | -0.0146 | -0.0224 | +0.0398 | +0.0462 | -0.0153 |
| 100 | CIFAR-100 / fixed-step | -0.0238 | -0.0268 | -0.0346 | +0.0213 | +0.0275 | +0.0243 |
| 100 | CIFAR-10 / fixed-epoch | -0.0507 | -0.0271 | -0.0103 | +0.0238 | +0.0381 | -0.0046 |
| 100 | CIFAR-100 / fixed-epoch | -0.0376 | -0.0491 | -0.0419 | +0.0247 | +0.0406 | +0.0385 |

$R=100$에서 B1--B2, B1--B3, B2--B3는 모든 조건에서 음수다. 반면 B1--Final과 B2--Final은 모든 조건에서 양수이며, B3--Final은 CIFAR-100에서 양수이고 CIFAR-10에서 소폭 음수다. 따라서 local sample 수가 증가하면 shallow depth끼리의 class direction은 오히려 분화될 수 있지만, B1/B2의 output은 Final의 class direction과 더 가까워질 수 있다. Aggregate cosine은 이 두 효과를 평균하기 때문에 변화가 작게 나타난다.

#### Linear CKA Δ

| R | Dataset / budget | B1--B2 | B1--B3 | B2--B3 | B1--Final | B2--Final | B3--Final |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | CIFAR-10 / fixed-step | +0.0116 | +0.0586 | +0.0832 | +0.0228 | +0.0397 | +0.0038 |
| 50 | CIFAR-100 / fixed-step | +0.0068 | +0.0297 | +0.0208 | +0.0400 | +0.0502 | +0.0619 |
| 50 | CIFAR-10 / fixed-epoch | +0.0174 | +0.0551 | +0.0769 | +0.0238 | +0.0406 | +0.0202 |
| 50 | CIFAR-100 / fixed-epoch | +0.0089 | +0.0352 | +0.0246 | +0.0375 | +0.0461 | +0.0575 |
| 100 | CIFAR-10 / fixed-step | -0.0490 | +0.0525 | +0.1079 | +0.0312 | +0.0660 | +0.0061 |
| 100 | CIFAR-100 / fixed-step | +0.0120 | +0.0258 | +0.0153 | +0.0719 | +0.1077 | +0.1261 |
| 100 | CIFAR-10 / fixed-epoch | -0.0227 | +0.0549 | +0.0951 | +0.0487 | +0.0799 | +0.0385 |
| 100 | CIFAR-100 / fixed-epoch | +0.0030 | +0.0230 | +0.0236 | +0.0657 | +0.0970 | +0.1174 |

CKA는 $R=50$에서 24개 pair가 모두 양수이고, $R=100$에서도 CIFAR-10의 B1--B2 두 조건을 제외한 모든 pair가 양수다. 특히 CIFAR-100의 branch--Final CKA 증가가 크다. 이는 local sample 수 증가가 depth별 raw feature geometry의 공통 sample structure를 안정화한다는 해석과 부합한다.

### 5.4 Fixed-step과 fixed-epoch의 관계

Fixed-step과 fixed-epoch는 서로 대체 관계가 아니라 다른 confound를 통제한다.

- Fixed-step 결과가 재현되면, 단순히 optimizer step이 많아져서 생긴 효과가 아니라 고유 local sample diversity와 관련된 근거가 된다.
- Fixed-epoch 결과가 재현되면, 실제 FedAvg처럼 모든 local sample을 같은 epoch 수만큼 사용하는 조건에서도 현상이 유지된다는 근거가 된다.

$n=1000$에서는 fixed-step 100 steps와 fixed-epoch $5\times1000/50=100$ steps가 일치한다. 두 결과의 CKA가 매우 가까웠지만 logit 값에는 작은 차이가 남았으며, 이는 augmentation 및 optimization stochasticity 때문에 작은 budget 간 차이를 과도하게 해석하면 안 된다는 점을 보여준다.

## 6. Centralized checkpoint 통제 실험

FL checkpoint 특유의 현상인지 확인하기 위해, 전체 train set으로 500 epochs 학습한 centralized teacher-only model을 공통 checkpoint로 사용하고 같은 local diagnostic을 수행했다.

### 6.1 Global checkpoint

| Dataset | Mean logit cosine | Directional variance | Mean CKA | Probe accuracy B1/B2/B3/Final (%) |
|---|---:|---:|---:|---:|
| CIFAR-10 | 0.6096 | 0.2928 | 0.4612 | 32.11 / 45.93 / 84.56 / 94.02 |
| CIFAR-100 | 0.5280 | 0.3540 | 0.4754 | 5.42 / 13.49 / 35.68 / 76.27 |

CIFAR-100은 CIFAR-10보다 global logit cosine이 낮지만, mean CKA는 오히려 0.0142 높다. 따라서 두 데이터셋만으로 task complexity가 CKA의 절대 수준을 결정한다고 주장할 수 없다. 반면 class-semantic readout의 차이는 CIFAR-100에서 더 크게 나타난다.

### 6.2 Post-local results

| Dataset / budget | Metric | $n=100$ | 250 | 500 | 1,000 | 2,500 | Δ(2,500−100) ± 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 / fixed-step | Logit cosine | 0.6047 | 0.5190 | 0.5243 | 0.5194 | 0.4818 | -0.1229 ± 0.0748 |
| CIFAR-10 / fixed-step | Directional variance | 0.2964 | 0.3608 | 0.3568 | 0.3604 | 0.3887 | +0.0922 ± 0.0561 |
| CIFAR-10 / fixed-step | CKA | 0.4843 | 0.5389 | 0.5774 | 0.5786 | 0.5660 | +0.0817 ± 0.0217 |
| CIFAR-100 / fixed-step | Logit cosine | 0.5317 | 0.5248 | 0.4976 | 0.4802 | 0.4905 | -0.0411 ± 0.0130 |
| CIFAR-100 / fixed-step | Directional variance | 0.3513 | 0.3564 | 0.3768 | 0.3899 | 0.3821 | +0.0308 ± 0.0098 |
| CIFAR-100 / fixed-step | CKA | 0.4884 | 0.5147 | 0.5794 | 0.5851 | 0.6045 | +0.1161 ± 0.0237 |
| CIFAR-10 / fixed-epoch | Logit cosine | 0.6070 | 0.5912 | 0.5458 | 0.4864 | 0.5389 | -0.0680 ± 0.0822 |
| CIFAR-10 / fixed-epoch | Directional variance | 0.2948 | 0.3066 | 0.3407 | 0.3852 | 0.3458 | +0.0510 ± 0.0616 |
| CIFAR-10 / fixed-epoch | CKA | 0.4656 | 0.4735 | 0.5264 | 0.5756 | 0.5735 | +0.1079 ± 0.0281 |
| CIFAR-100 / fixed-epoch | Logit cosine | 0.5268 | 0.5245 | 0.4761 | 0.4971 | 0.5072 | -0.0196 ± 0.0265 |
| CIFAR-100 / fixed-epoch | Directional variance | 0.3549 | 0.3566 | 0.3929 | 0.3772 | 0.3696 | +0.0147 ± 0.0199 |
| CIFAR-100 / fixed-epoch | CKA | 0.4769 | 0.4875 | 0.5333 | 0.5893 | 0.5785 | +0.1016 ± 0.0112 |

Centralized checkpoint에서도 네 조건 모두 CKA endpoint Δ가 양수였고 CI가 0을 포함하지 않았다. 반면 logit cosine은 $n$이 증가하면서 감소했다. 따라서 CKA와 logit cosine의 불일치는 FL aggregation만으로 발생한 현상이 아니라, **동일한 deep network 안에서 raw feature geometry와 선형 class readout이 서로 다른 방식으로 변할 수 있다는 구조적 현상**이다.

## 7. Frozen probe와 local-refit probe

CKA는 증가하지만 frozen-probe logit cosine은 다른 방향을 보이는 원인을 확인하기 위해, 각 local model의 feature를 고정한 뒤 local model별 probe를 전체 train set으로 다시 학습했다. 아래 표의 괄호는 `refit minus frozen` accuracy point다.

| Dataset | Budget | $n$ | B1 frozen→refit | B2 frozen→refit | B3 frozen→refit | Final frozen→refit |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | fixed-step | 100 | 35.44→35.59 (+0.16) | 47.75→48.02 (+0.27) | 75.68→75.63 (-0.05) | 93.19→93.46 (+0.28) |
| CIFAR-10 | fixed-step | 2,500 | 31.64→34.77 (+3.13) | 41.34→45.69 (+4.35) | 60.28→64.32 (+4.04) | 78.86→84.40 (+5.53) |
| CIFAR-10 | fixed-epoch | 100 | 35.60→35.64 (+0.04) | 47.89→48.02 (+0.13) | 76.25→76.10 (-0.14) | 93.57→93.77 (+0.19) |
| CIFAR-10 | fixed-epoch | 2,500 | 32.60→34.98 (+2.38) | 44.03→46.24 (+2.20) | 65.31→66.46 (+1.15) | 83.27→86.40 (+3.13) |
| CIFAR-100 | fixed-step | 100 | 4.80→4.87 (+0.07) | 11.88→11.78 (-0.10) | 24.13→24.22 (+0.09) | 70.75→71.83 (+1.09) |
| CIFAR-100 | fixed-step | 2,500 | 4.74→5.00 (+0.26) | 10.76→11.17 (+0.41) | 20.25→20.17 (-0.08) | 45.61→59.30 (+13.68) |
| CIFAR-100 | fixed-epoch | 100 | 4.79→4.88 (+0.09) | 11.96→11.86 (-0.10) | 24.29→24.41 (+0.12) | 71.37→72.25 (+0.88) |
| CIFAR-100 | fixed-epoch | 2,500 | 4.72→5.02 (+0.31) | 11.36→11.70 (+0.33) | 22.20→21.84 (-0.36) | 55.74→62.38 (+6.64) |

Refit은 특히 큰 $n$에서 Final accuracy를 크게 복구했다. 이는 local update 후 feature와 global frozen probe 사이의 방향 mismatch가 실제로 존재함을 보여준다. 그러나 B1--B3의 refit gain은 작거나 음수인 경우도 많다. 따라서 frozen-probe mismatch만으로 shallow pair의 logit/CKA 차이를 모두 설명할 수는 없다. 더 정확한 해석은 다음 두 요인이 함께 존재한다는 것이다.

1. Local training이 global probe와 feature 사이의 좌표 관계를 변화시킨다.
2. CKA가 보존하는 전체 sample geometry와 probe가 읽는 class-semantic direction은 원래부터 동일한 정보가 아니다.

## 8. 실제 $K$별 CIFAR-100 FL 실험

전체 train set 50,000개를 유지하면서 $K\in\{1,5,20,50,100\}$으로 IID 분할했다. 따라서 $n=50{,}000/K$이다. 모든 조건은 50 rounds, 5 local epochs, full participation으로 학습했다. 이 실험은 seed 0 하나이며, $K\ge20$에서는 저장된 10개 pre-aggregation local client를 분석했으므로 confirmatory evidence가 아니라 진단 결과로 취급한다.

| $K$ | Local $n$ | Local B1 acc. | B2 acc. | B3 acc. | Local Final acc. | Mean branch--Final cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50,000 | 5.06 | 12.46 | 31.60 | 75.94 | 0.3358 |
| 5 | 10,000 | 4.66 | 12.43 | 27.17 | 74.60 | 0.3398 |
| 20 | 2,500 | 6.25 | 12.97 | 23.62 | 68.94 | 0.3743 |
| 50 | 1,000 | 7.58 | 13.24 | 21.42 | 58.57 | 0.4157 |
| 100 | 500 | 9.96 | 14.57 | 23.54 | 48.72 | 0.5049 |

$K$가 증가하고 $n$이 감소할수록 Final accuracy는 크게 감소하지만 branch--Final cosine은 증가했다. 따라서 이 alignment 증가는 좋은 semantic sharing의 증거로 보기 어렵다. Shallow accuracy가 함께 향상된 것이 아니라 Final이 충분히 전문화되지 못해 depth hierarchy가 압축된 결과와 일치한다.

이 실험은 전체 데이터 수를 고정한 실제 FL 조건이라는 장점이 있지만, $K$, $n$, aggregation trajectory가 함께 변한다. 따라서 $n$의 순수한 local effect는 동일-checkpoint 실험으로 해석하고, 실제 $K$별 결과는 FL-level external validity로 분리해야 한다.

## 9. 고정 $K=20$ end-to-end local-$n$ 실험: 설계상 confound

이 실험은 $K=20$을 고정하고 $n\in\{100,500,2500\}$을 처음부터 적용하여 100-round FL trajectory를 각각 학습했다. 그러나 이 설정에서는 global model이 접근한 고유 train data가 각각 2,000, 10,000, 50,000개가 된다. 즉 local sample 수와 global data coverage가 동시에 바뀐다.

| Dataset / budget | $n$ | Total unique train data | Final global acc. | Logit cosine | Directional variance | CKA |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 / fixed-step | 100 | 2,000 | 54.34 | 0.7171 | 0.2122 | 0.5366 |
| CIFAR-10 / fixed-step | 500 | 10,000 | 82.37 | 0.7188 | 0.2109 | 0.5711 |
| CIFAR-10 / fixed-step | 2,500 | 50,000 | 92.95 | 0.6842 | 0.2369 | 0.5516 |
| CIFAR-100 / fixed-step | 100 | 2,000 | 18.89 | 0.5826 | 0.3130 | 0.5534 |
| CIFAR-100 / fixed-step | 500 | 10,000 | 39.88 | 0.5605 | 0.3296 | 0.5022 |
| CIFAR-100 / fixed-step | 2,500 | 50,000 | 72.07 | 0.5559 | 0.3331 | 0.5042 |
| CIFAR-10 / fixed-epoch | 100 | 2,000 | 55.86 | 0.8499 | 0.1126 | 0.5351 |
| CIFAR-10 / fixed-epoch | 500 | 10,000 | 80.82 | 0.7425 | 0.1931 | 0.5898 |
| CIFAR-10 / fixed-epoch | 2,500 | 50,000 | 94.02 | 0.6477 | 0.2642 | 0.5051 |
| CIFAR-100 / fixed-epoch | 100 | 2,000 | 16.40 | 0.7665 | 0.1751 | 0.5908 |
| CIFAR-100 / fixed-epoch | 500 | 10,000 | 38.57 | 0.6213 | 0.2840 | 0.5286 |
| CIFAR-100 / fixed-epoch | 2,500 | 50,000 | 73.02 | 0.5143 | 0.3643 | 0.4845 |

특히 fixed-epoch에서 $n$이 증가할수록 accuracy는 크게 증가하지만 logit cosine과 CKA는 감소했다. 이 결과는 동일-checkpoint 실험과 반대지만 모순은 아니다. 작은 data coverage로 학습한 model은 Final까지 충분히 전문화되지 않아 depth들이 비슷한 generic geometry를 유지할 수 있고, 그 결과 성능은 낮으면서 alignment는 높을 수 있다.

또한 round 99의 cosine LR는 약 $2.47\times10^{-5}$이므로 final post-local model과 round-start global model의 차이는 매우 작다. 이 실험의 최종 값은 마지막 local update의 효과보다 100-round global training trajectory의 차이를 주로 반영한다.

따라서 이 결과는 다음 주장에는 사용할 수 있다.

> Global data coverage와 model maturity가 cross-depth alignment의 절대 수준을 강하게 바꾼다.

하지만 다음 주장에는 사용할 수 없다.

> Client당 local sample 수만 감소했기 때문에 cross-depth CKA가 증가했다.

## 10. $K$--$n$ matched end-to-end 재실험

위 confound를 제거하기 위해 전체 고유 train data를 항상 50,000개로 유지했다.

| Local $n$ | Clients $K$ | $K\times n$ |
|---:|---:|---:|
| 100 | 500 | 50,000 |
| 500 | 100 | 50,000 |
| 2,500 | 20 | 50,000 |

- IID, disjoint partition, full participation
- 100 independent FL rounds
- 5 fixed local epochs
- Cosine LR, initial LR 0.1
- CIFAR-10/100, seeds 0/1/2
- 전체 selected clients에서 final pre-aggregation geometry 측정
- CIFAR-10/100 × 3개 $n/K$ 조건 × seeds 0/1/2의 18개 trajectory 모두 완료

모든 조건은 매 round 50,000개 고유 sample을 5회씩 처리하므로 processed samples는 250,000개, 총 client optimizer step은 5,000으로 같다. 대신 client당 step은 10, 50, 250으로 달라진다. 따라서 이 실험은 data coverage와 system-level sample budget을 맞춘 상태에서 실제 FL의 client granularity, 즉 $n$, $K$, local trajectory 길이가 결합된 효과를 측정한다.

| Dataset | Metric | $n=100$ / $K=500$ | $n=500$ / $K=100$ | $n=2,500$ / $K=20$ | $\Delta_{2500-100}$ |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | Logit cosine | 0.8134 | 0.7130 | 0.6477 | **-0.1657 ± 0.0259** |
| CIFAR-10 | Directional variance | 0.1399 | 0.2152 | 0.2642 | **+0.1243 ± 0.0194** |
| CIFAR-10 | Linear CKA | 0.5579 | 0.5708 | 0.5051 | -0.0528 ± 0.1394 |
| CIFAR-10 | Final frozen-probe acc. | 79.78 | 89.65 | 93.97 | **+14.20 ± 1.95** |
| CIFAR-100 | Logit cosine | 0.7146 | 0.5950 | 0.5143 | **-0.2003 ± 0.0067** |
| CIFAR-100 | Directional variance | 0.2141 | 0.3038 | 0.3643 | **+0.1502 ± 0.0051** |
| CIFAR-100 | Linear CKA | 0.5873 | 0.5364 | 0.4845 | **-0.1028 ± 0.0297** |
| CIFAR-100 | Final frozen-probe acc. | 42.38 | 60.25 | 72.71 | **+30.33 ± 1.64** |

두 데이터셋 모두 $n$이 증가할수록 Final accuracy가 크게 향상되지만 cross-depth logit cosine은 감소했다. 작은 $n$/큰 $K$에서 관찰된 높은 alignment는 좋은 성능이 아니라 depth hierarchy가 덜 분화된 상태와 일치한다. CIFAR-100 CKA도 같은 방향으로 유의하게 감소했다. CIFAR-10 CKA는 $n=500$에서 가장 높고 endpoint CI가 0을 포함하므로 단조 결론을 내릴 수 없다.

Branch-pair 수준에서는 branch--Final logit cosine이 두 데이터셋의 모든 pair에서 감소했다. CIFAR-100의 branch--Final CKA도 모두 감소했지만 CIFAR-10의 B3--Final CKA는 반대로 증가해, task와 depth에 따른 차이가 존재한다.

마지막 cosine LR가 약 $2.47\times10^{-5}$이므로 post-local 값과 round-start global 값의 차이는 대부분 $10^{-3}$ 이하이다. 따라서 이 결과는 마지막 local update의 즉시 효과보다 100 rounds 동안 누적된 $K/n$-specific FL trajectory의 최종 pre-aggregation 상태를 나타낸다. 상세 branch-pair 표와 정확도 분석은 `analysis/kn_matched_end_to_end_results.md`에 정리했다.

## 11. 종합 해석

### 11.1 가장 강하게 지지되는 결론

1. **Local sample 수는 하나의 local model 내부 depth 관계에 영향을 준다.** 성숙한 동일 checkpoint에서 CKA는 $n$이 증가할수록 대체로 증가했다.
2. **Feature geometry와 class-semantic readout은 동일한 축이 아니다.** CKA는 일관되게 증가하는 반면 aggregate logit cosine은 작거나 혼합된 변화를 보였다.
3. **Logit 변화는 branch pair에 따라 구조화되어 있다.** $R=100$에서 shallow--shallow cosine은 $n$과 함께 감소했지만 B1/B2--Final cosine은 증가했다.
4. **높은 alignment 자체는 quality 지표가 아니다.** Random initialization과 data coverage가 부족한 trajectory에서도 높은 CKA와 cosine이 나타났고, 동시에 accuracy는 낮았다.
5. **관찰된 현상은 FL checkpoint에만 한정되지 않는다.** Centralized checkpoint에서도 CKA 증가와 logit cosine 감소가 함께 관찰되었다.
6. **실제 FL에서 작은 $n$/큰 $K$는 높은 alignment와 낮은 성능을 동시에 만들 수 있다.** 전체 data coverage와 처리량을 맞춘 재실험에서도 이 경향이 유지되었다.

### 11.2 논문에서 피해야 할 주장

- `Local data가 작을수록 항상 branch--Final representation 차이가 커진다.`  
  현재 logit과 CKA 결과는 이 단순한 단조 주장을 지지하지 않는다.
- `Alignment가 높을수록 representation이 좋다.`  
  $R=0$ 및 under-trained trajectory가 직접적인 반례다.
- `Directional variance와 cosine이 서로 독립적으로 같은 결론을 확인했다.`  
  두 aggregate 지표는 대수적으로 중복된다.
- `CIFAR-10/100 차이는 순수하게 task complexity 때문에 발생했다.`  
  class 수, local class coverage, checkpoint source가 함께 다르다.
- `고정 K=20 end-to-end 결과가 local n의 인과 효과다.`  
  전체 global training data coverage가 함께 변했다.
- `$K$--$n$ matched 결과가 n만의 인과 효과다.`  
  수정 실험에서도 $n$, $K$, client당 local trajectory 길이는 함께 변한다.

### 11.3 Self-distillation motivation과의 연결

현재 결과가 직접 뒷받침하는 motivation은 “모든 depth를 무조건 동일하게 만들어야 한다”가 아니다. 더 정확한 연결은 다음과 같다.

> Final-only local training에서는 local data 규모에 따라 depth별 sample geometry와 class-semantic readout이 서로 다른 방식으로 변한다. 특히 shallow depth 사이의 semantic relation과 shallow--Final relation은 동일한 방향으로 움직이지 않으며, raw feature geometry가 유사해도 Final의 class direction이 intermediate depth에서 동일하게 읽히는 것은 아니다. 따라서 Final prediction을 intermediate branch에 직접 제공하는 self-distillation은, 단순한 feature 동일화가 아니라 Final에 형성된 task-relevant semantic signal을 local intermediate representation에 전달하는 명시적 supervision으로 해석할 수 있다.

Self-distillation의 유효성은 alignment 증가만으로 판단하지 않고 다음을 함께 확인해야 한다.

- Final 또는 global test accuracy 유지/향상
- Branch probe accuracy 유지/향상
- Logit/CKA 변화가 random-like collapse나 under-specialization으로 설명되지 않음
- Baseline 대비 Final-to-branch KD의 실제 성능 gain

## 12. 논문용 서술 초안

### 12.1 Metric paragraph

> We analyze cross-depth representations of pre-aggregation local models using two complementary metrics. Linear CKA measures the similarity of sample-wise feature geometry and is invariant to orthogonal changes of feature coordinates, whereas centered-logit cosine measures the agreement of class-semantic directions extracted by depth-specific linear probes. Since the directional variance of four centered unit logit vectors is algebraically determined by their mean pairwise cosine, we report it only as an equivalent dispersion view rather than independent evidence.

### 12.2 Main result paragraph

> At independently trained mature FL checkpoints, increasing the number of local samples consistently increased cross-depth CKA across both CIFAR-10 and CIFAR-100 and under both fixed-step and fixed-epoch local budgets. The corresponding logit-level trend was substantially weaker because shallow--shallow and shallow--Final pairs moved in opposite directions. At round 100, all shallow--shallow cosine differences between $n=2{,}500$ and $n=100$ were negative, whereas B1--Final and B2--Final differences were positive in every condition. These observations indicate that additional local data stabilizes shared sample geometry while redistributing linearly accessible class information across network depth.

### 12.3 Interpretation paragraph

> Cross-depth similarity should not be interpreted as representation quality in isolation. Randomly initialized checkpoints exhibited the highest CKA and logit agreement, while their classification accuracy was the lowest. Likewise, trajectories trained with severely reduced global data coverage showed high alignment but poor final accuracy. The relevant phenomenon is therefore not the absolute maximization of cross-depth alignment, but the manner in which task-relevant semantics are organized across depth under limited local supervision.

### 12.4 Motivation paragraph

> In standard local training, intermediate representations are optimized only indirectly through the final classification objective. Our analysis shows that feature geometry and class-semantic readouts respond differently to the amount of local data, and that the semantic relation among shallow depths can evolve differently from the relation between shallow and final depths. This motivates using the final prediction as an explicit within-model semantic target for intermediate branches. Such self-distillation is intended to transfer task-relevant information from the final predictor without requiring all depths to learn identical representations.

### 12.5 실제 $K$--$n$ matched FL 결과 paragraph

> When the total IID training coverage and per-round sample exposure were held constant by setting $Kn=50{,}000$, fine-grained FL with many small clients produced higher cross-depth logit agreement but substantially lower final accuracy. Increasing $n$ from 100 to 2,500 reduced the mean centered-logit cosine by 0.166 on CIFAR-10 and 0.200 on CIFAR-100, while improving the final frozen-probe accuracy by 14.2 and 30.3 percentage points, respectively. The effect was concentrated in branch--Final pairs. These results indicate that high cross-depth agreement can reflect insufficient depth-wise specialization rather than successful semantic transfer.

## 13. 권장 본문/부록 배치

### 본문에 권장

1. Section 4의 global checkpoint maturity 표
2. Section 5.2의 $R=50,100$ CKA endpoint 결과
3. Section 5.3의 $R=100$ branch-pair logit/CKA 결과
4. Section 7의 local-refit 결과 중 $n=100,2500$ endpoint
5. Section 10의 $K$--$n$ matched 실제 FL 결과

### 부록에 권장

1. $R=0,10$ control
2. 모든 $n$의 absolute value
3. Centralized checkpoint 전체 표
4. Fixed $K=20$ coverage-confounded end-to-end 결과와 제한점
5. 실제 $K$별 single-seed pilot

## 14. 원본 결과 파일

- 독립 round 결과 요약: `logs/analysis/independent_round_budget_logit_cka_analysis/result_summary.md`
- 독립 round 일관성 분석: `logs/analysis/independent_round_budget_logit_cka_analysis/independent_budget_consistency_report.md`
- Branch-pair raw CSV: `logs/analysis/independent_round_budget_logit_cka_analysis/branch_pair_postlocal_metrics.csv`
- CIFAR-10/100 fixed-checkpoint 비교: `logs/analysis/cifar10_cifar100_checkpoint_logit_comparison/comparison_report.md`
- Local-refit probe: `logs/analysis/local_probe_refit_analysis/local_probe_refit_accuracy.csv`
- Centralized checkpoint 결과: `logs/analysis/logs_centralized_checkpoint_local_n/{dataset}/{budget}/summary.json`
- 실제 $K$별 pilot: `logs/analysis/logs_iid_client_count_representation/cifar100_resnet18/iid/clients_*/fedavg/seed0/*_representation_probe.json`
- Coverage-confounded end-to-end 결과: `logs/analysis/logs_end_to_end_local_n_fl/{dataset}/{budget}/summary.json`
- 수정된 $K$--$n$ matched 실행 결과: `logs/analysis/logs_end_to_end_kn_matched_fl/`
- 수정된 $K$--$n$ matched 상세 분석: `analysis/kn_matched_end_to_end_results.md`
