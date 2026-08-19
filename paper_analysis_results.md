# FL 환경의 BYOT Branch Supervision 분석

> 이 문서는 논문의 **analysis section**을 작성하기 위한 실험 결과 및 해석 초안이다.  
> 요청에 따라 client 수 또는 client당 데이터 수를 바꾸어 중간 표현 저하를 보인 **motivation 실험은 모두 제외**하였다.  
> 별도 표기가 없으면 정확도는 `%`, 500-round 실험은 마지막 30개 round 평균(`Last-30`)이다. `±`가 있는 값은 seed 평균 ± sample standard deviation이다.

---

## 1. 핵심 결론

현재 결과가 지지하는 가장 안전한 결론은 다음과 같다.

1. **원래의 BYOT처럼 모든 branch에 hard-label CE를 부과하는 방식은 FL에서 항상 유리하지 않다.** 그 효과는 task complexity와 data heterogeneity에 모두 의존한다.
2. **단순 task의 non-IID 환경에서는 branch CE가 branch KD보다 낫지만, branch CE 자체도 feature-only보다 나쁘다.** 따라서 이 영역의 정확한 결론은 “CE가 유익하다”가 아니라 “KD가 더 불필요하며 CE가 상대적으로 덜 해롭다”이다.
3. **CIFAR-100, TinyImageNet, ImageNet100-64와 같은 복잡한 task에서는 `alpha`를 높여 branch CE를 제거하는 방향이 일관되게 유리하다.** 다만 극심한 non-IID에서는 `alpha=1`도 plain 또는 feature-only보다 낮을 수 있으므로, 이는 “CE 제거”와 “강한 KD 적용”을 분리해야 함을 뜻한다.
4. **`alpha=1`은 branch prediction을 더 soft하게 만들고, 특히 locally rare class에 대한 client 간 branch probability dispersion을 줄인다.** 같은 변화가 teacher에서는 거의 나타나지 않으므로 이는 전체 model이 동일한 teacher로 수렴한 결과가 아니라 branch supervision에 특이적인 변화이다.
5. **Branch loss가 final teacher 성능에 영향을 주는 인과 경로는 shared backbone을 통하는 gradient이다.** Branch gradient를 detach하면 CE와 KD 모두 final curve가 branch-off와 정확히 같아진다.
6. 그러나 **KD가 유리한 이유를 “세부 class-relation dark knowledge” 또는 “teacher CE gradient와 더 잘 정렬되기 때문”이라고 쓰면 현재 증거와 맞지 않는다.** 세부 non-target 분포와 sample별 teacher softness를 제거해도 C100 성능이 유지되거나 증가했고, KD gradient의 teacher-CE cosine도 CE gradient보다 작았다.
7. 현재 결과가 가장 잘 지지하는 기계적 설명은 다음과 같다. **복잡한 task에서 shallow exit에 one-hot 판별을 직접 강제하는 대신, temperature-scaled soft auxiliary target과 더 작은 KD gradient scale을 사용하면 shared representation에 가해지는 hard-label pressure를 완화할 수 있다.** 이 효과는 세부 class relation 자체보다 target softness, temperature, gradient scale의 결합에 더 가깝다.
8. 따라서 논문의 방법론적 결론은 **복잡한 task에서 branch CE를 제거하되, KD의 절대 세기 `lambda`는 non-IID 정도에 맞게 다시 조절해야 한다**는 것이다.

---

## 2. 설정, 표기, 지표

### 2.1 BYOT loss

기본 CE–KD blend는 다음과 같다.

$$
\mathcal{L}
=\mathcal{L}^{\mathrm{CE}}_{T}
+\sum_{b=1}^{3}
\left[(1-\alpha)\mathcal{L}^{\mathrm{CE}}_{b}
+\alpha\mathcal{L}^{\mathrm{KD}}_{b}\right]
+\beta_{\mathrm{feat}}\sum_{b=1}^{3}\mathcal{L}^{\mathrm{feat}}_{b}.
$$

- `Teacher` 또는 `Final`: 최종 classifier. 항상 hard-label CE를 받는다.
- `B1/B2/B3`: 서로 다른 depth에 붙은 intermediate exit이다.
- $\alpha=0$: branch prediction loss는 CE만 사용한다.
- $\alpha=1$: branch CE를 제거하고 branch KD만 사용한다.
- 따라서 **$\alpha$가 커진다는 것은 KD의 절대 세기가 커진다는 뜻이 아니다.** CE와 KD의 상대 구성이 바뀌는 것이다.

CE 제거 후 KD 세기를 별도로 조절한 실험은 다음 loss를 사용한다.

$$
\mathcal{L}
=\mathcal{L}^{\mathrm{CE}}_{T}
+\lambda\sum_{b=1}^{3}\mathcal{L}^{\mathrm{KD}}_{b}
+\beta_{\mathrm{feat}}\sum_{b=1}^{3}\mathcal{L}^{\mathrm{feat}}_{b}.
$$

### 2.2 CE와 KD는 모두 logit에 gradient를 전달한다

Branch logit을 $z_b$, 확률을 $p_b=\operatorname{softmax}(z_b)$라고 하면 CE의 logit gradient는 다음과 같다.

$$
\frac{\partial \mathcal{L}^{\mathrm{CE}}_b}{\partial z_b}
=p_b-e_y,
$$

여기서 $e_y$는 one-hot label이다. 즉 CE는 정답 class 확률을 1로 만드는 방향을 모든 sample에 동일하게 요구한다.

Teacher와 branch의 temperature-scaled 분포를

$$
q_T^{(T_t)}=\operatorname{softmax}(z_T/T_t),\qquad
p_b^{(T_s)}=\operatorname{softmax}(z_b/T_s)
$$

라고 하고

$$
\mathcal{L}^{\mathrm{KD}}_b
=T_s^2\,D_{\mathrm{KL}}\!\left(q_T^{(T_t)}\,\|\,p_b^{(T_s)}\right)
$$

를 사용하면, student logit에 대한 leading gradient는 대략 다음 형태이다.

$$
\frac{\partial \mathcal{L}^{\mathrm{KD}}_b}{\partial z_b}
\propto T_s\left(p_b^{(T_s)}-q_T^{(T_t)}\right).
$$

따라서 CE와 KD의 차이는 “CE는 probability를, KD는 logit을 학습한다”가 아니다. **둘 다 softmax를 거쳐 logit에 gradient를 전달하지만, CE는 one-hot target을 사용하고 KD는 soft target과 temperature-dependent scale을 사용한다.**

### 2.3 주요 비교 조건

| Condition | Branch CE | Branch KD | Feature imitation | 용도 |
|---|---:|---:|---:|---|
| Plain ResNet18 | 없음 | 없음 | 없음 | Branch 자체가 없는 architecture baseline |
| Off | 0 | 0 | 0 | BYOT-shaped model에서 모든 branch-side loss 제거 |
| Feature-only | 0 | 0 | 사용 | Feature imitation만의 효과 |
| CE + feature | 사용 | 0 | 사용 | $\alpha=0$에 대응 |
| KD + feature | 0 | 사용 | 사용 | $\alpha=1$에 대응 |
| CE/KD detached | Head에는 전달 | Head에는 전달 | 조건별 상이 | Branch gradient를 shared backbone에서 차단하는 causal control |

`Plain`은 branch가 없는 ResNet18이고, `Off`는 BYOT-shaped architecture를 유지하되 branch-side gradient를 제거한 control이다. 따라서 architecture 자체와 branch supervision의 효과를 분리할 때는 `Off`, 최종 방법의 실용 성능을 비교할 때는 `Plain`이 중요하다.

### 2.4 Logit/probability 지표

| 지표 | 정의 | 의미 |
|---|---|---|
| True-label probability | $p_y=p(y\mid x)$ | 정답 class에 배정한 확률 |
| Normalized entropy | $H_{\mathrm{norm}}(p)=-\sum_c p_c\log p_c/\log C$ | 0이면 매우 sharp, 1이면 uniform |
| JS-to-client-mean | $\frac{1}{K}\sum_k D_{\mathrm{JS}}(p_k,\bar p)$ | 같은 reference sample에 대한 client별 확률분포 불일치 |
| Raw-logit variance | class별 client logit 분산의 평균 | 방향 차이와 logit scale 차이를 모두 포함 |
| Centered-logit cosine | $\cos(z_i-\bar z_i,z_j-\bar z_j)$ | additive shift와 scale을 제거한 class-contrast 방향의 유사도 |
| Normalized logit L2 | $\lVert\tilde z_i-\tilde z_j\rVert_2/\sqrt C$ | centered logit의 scale-sensitive 거리 |
| JS@$T$ | $JS(\operatorname{softmax}(z_i/T),\operatorname{softmax}(z_j/T))$ | 해당 temperature에서의 probability 관계 차이 |

Entropy 증가와 client JS 감소는 서로 다른 현상이다. 전자는 **각 client prediction의 softness**, 후자는 **서로 다른 client prediction 중심의 일치도**를 나타낸다.

---

## 3. Task에 따라 `alpha` 경향이 반대로 나타난다

### 3.1 단순 task: CIFAR-10

아래는 500-round FedAvg의 Last-30 결과이다. 이 legacy $\alpha$ sweep은 branch-loss reduction=`sum`, KD temperature $T=0.5$를 사용한다.

| Partition | Plain | $\alpha=0$ | $\alpha=0.3$ | $\alpha=0.7$ | $\alpha=1$ | $\Delta(1-0)$ | $\Delta(1-\mathrm{Plain})$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 92.114 | 92.362 | 92.403 | 92.385 | 92.520 | +0.158 | +0.406 |
| $\beta=0.5$ | 78.833 | 75.629 | 74.616 | 74.159 | 72.847 | -2.782 | -5.986 |
| $\beta=0.3$ | 70.464 | 63.543 | 63.149 | 62.068 | 60.563 | -2.980 | -9.901 |
| $\beta=0.1$ | 29.304 | 23.787 | 22.985 | 19.945 | 18.911 | -4.876 | -10.393 |

**해석.** IID에서는 $\alpha$에 따른 차이가 작지만, non-IID에서는 $\alpha$가 커질수록 성능이 일관되게 감소한다. 그러나 $\alpha=0$도 plain보다 낮다. 따라서 CIFAR-10 non-IID의 관측은 “branch CE가 유익하다”가 아니라 다음 두 문장으로 나누어야 한다.

- Branch CE는 branch KD보다 상대적으로 낫다.
- 하지만 branch prediction supervision 자체는 plain FL 또는 feature-only보다 해로울 수 있다.

Fashion-MNIST 500-round 재실행에서도 강한 non-IID에서 같은 방향이 나타난다. Plain 로그는 100 rounds라 같은 표에서 직접 비교하지 않았다.

| Partition | $\alpha=0$ | $\alpha=0.3$ | $\alpha=0.7$ | $\alpha=1$ | $\Delta(1-0)$ |
|---|---:|---:|---:|---:|---:|
| IID | 94.253 | 94.231 | 94.278 | 94.329 | +0.076 |
| $\beta=0.5$ | 86.883 | 87.574 | 87.126 | 86.365 | -0.518 |
| $\beta=0.3$ | 78.971 | 78.801 | 77.885 | 77.498 | -1.473 |
| $\beta=0.1$ | 43.024 | 39.948 | 39.879 | 37.268 | -5.756 |

Fashion-MNIST의 $\beta=0.5$에서는 $\alpha=0.3$이 가장 좋으므로 단순 task에서도 반드시 $\alpha=0$만 최적인 것은 아니다. 더 안전한 결론은 **강한 KD dominance가 필요하지 않으며, heterogeneity가 커질수록 $\alpha=1$이 불리해진다**는 것이다.

### 3.2 복잡한 task: CIFAR-100과 TinyImageNet

| Dataset | Partition | Plain | $\alpha=0$ | $\alpha=0.3$ | $\alpha=0.7$ | $\alpha=1$ | $\Delta(1-0)$ | $\Delta(1-\mathrm{Plain})$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | IID | 70.336 | 71.713 | 71.845 | **72.471** | 72.370 | +0.657 | +2.034 |
| CIFAR-100 | $\beta=0.5$ | 67.495 | 67.754 | 67.791 | **68.052** | 68.049 | +0.295 | +0.554 |
| CIFAR-100 | $\beta=0.3$ | 65.153 | 65.100 | 65.260 | 65.080 | **66.030** | +0.930 | +0.877 |
| CIFAR-100 | $\beta=0.1$ | 55.359 | 52.436 | 52.226 | 52.410 | **53.601** | +1.165 | -1.758 |
| TinyImageNet | IID | 54.652 | 54.765 | 55.324 | 55.808 | **57.391** | +2.626 | +2.739 |
| TinyImageNet | $\beta=0.5$ | 53.347 | 53.511 | 53.289 | 54.272 | **55.073** | +1.562 | +1.726 |
| TinyImageNet | $\beta=0.3$ | 52.913 | 51.802 | 52.033 | 53.142 | **54.442** | +2.640 | +1.529 |
| TinyImageNet | $\beta=0.1$ | 48.502 | 46.670 | 46.832 | 48.005 | **49.084** | +2.414 | +0.582 |

**해석.** 모든 partition에서 $\alpha=1$은 $\alpha=0$보다 높다. CIFAR-100 IID와 $\beta=0.5$에서는 $\alpha=0.7$이 각각 0.101, 0.003 point 높으므로 “항상 정확히 $\alpha=1$이 최고”라고 쓸 수는 없다. 그러나 **branch CE를 줄이거나 제거하는 방향**은 일관된다.

Plain과 비교하면 결론이 더 제한된다. CIFAR-100 $\beta=0.1$에서는 $\alpha=1$도 plain보다 1.758 point 낮다. 즉 severe non-IID에서 CE 제거는 CE-containing BYOT를 회복시키지만, branch supervision 전체의 문제를 완전히 해결하지는 못한다.

### 3.3 ImageNet100-64에서도 같은 방향이 재현된다

ImageNet100-64는 300-round Last-30 결과이다.

| Partition | Plain | $\alpha=0$ | $\alpha=0.3$ | $\alpha=0.7$ | $\alpha=1$ | $\Delta(1-0)$ | $\Delta(1-\mathrm{Plain})$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 68.305 | 69.667 | 69.457 | 70.463 | **70.915** | +1.248 | +2.610 |
| $\beta=0.5$ | 66.075 | 67.133 | 67.076 | 67.336 | **68.042** | +0.909 | +1.967 |
| $\beta=0.3$ | 64.338 | 65.335 | 65.170 | 65.687 | **66.308** | +0.973 | +1.970 |
| $\beta=0.1$ | 57.258 | 55.141 | 55.544 | 56.092 | **57.041** | +1.900 | -0.217 |

이 결과는 복잡한 task에서 $\alpha$ 증가가 유리하다는 현상이 CIFAR-100에만 한정되지 않음을 보인다. 동시에 $\beta=0.1$에서는 여전히 plain을 안정적으로 넘지 못한다.

### 3.4 TinyImageNet 경향은 짧은 horizon에서도 나타난다

각 값은 해당 round를 끝점으로 하는 30-round 평균이다.

| End round | Partition | Plain | $\alpha=0$ | $\alpha=0.3$ | $\alpha=0.7$ | $\alpha=1$ |
|---:|---|---:|---:|---:|---:|---:|
| 100 | IID | 41.034 | 45.170 | 45.048 | 45.280 | **45.534** |
| 100 | $\beta=0.5$ | 39.446 | 42.546 | 42.438 | 42.634 | **42.951** |
| 100 | $\beta=0.3$ | 38.381 | 40.783 | 40.707 | 41.291 | **41.814** |
| 100 | $\beta=0.1$ | 34.029 | 33.739 | 33.829 | 34.224 | **34.805** |
| 300 | IID | 51.846 | 53.203 | 53.888 | 53.901 | **55.226** |
| 300 | $\beta=0.5$ | 50.134 | 51.629 | 51.480 | 52.278 | **52.887** |
| 300 | $\beta=0.3$ | 49.691 | 50.102 | 49.975 | 50.907 | **51.976** |
| 300 | $\beta=0.1$ | 45.435 | 44.827 | 44.699 | 45.664 | **46.659** |
| 500 | IID | 54.652 | 54.765 | 55.324 | 55.808 | **57.391** |
| 500 | $\beta=0.5$ | 53.347 | 53.511 | 53.289 | 54.272 | **55.073** |
| 500 | $\beta=0.3$ | 52.913 | 51.802 | 52.033 | 53.142 | **54.442** |
| 500 | $\beta=0.1$ | 48.502 | 46.670 | 46.832 | 48.005 | **49.084** |

$\alpha=1$의 상대적 우세는 100, 300, 500 rounds에서 모두 관찰된다. 따라서 TinyImageNet의 경향을 단순한 late-training artifact로 보기는 어렵다.

### 3.5 CIFAR-100 nested class-count 실험

CIFAR-100의 동일한 seeded class permutation에서 앞의 $K$개 class만 선택해 $K\in\{10,20,50,100\}$을 비교했다. $\beta=0.5$, 500 rounds, 3 seeds이다.

| Class count $K$ | $\alpha=0$ | $\alpha=1$ | $\Delta(1-0)$ |
|---:|---:|---:|---:|
| 10 | 64.657 ± 3.348 | 57.729 ± 3.084 | -6.928 |
| 20 | 72.121 ± 0.718 | 70.718 ± 2.480 | -1.403 |
| 50 | 67.247 ± 0.711 | 67.920 ± 0.677 | +0.673 |
| 100 | 64.443 ± 0.608 | 64.966 ± 0.517 | +0.523 |

**해석.** 같은 CIFAR-100 source 내에서도 class 수가 증가하면서 $\alpha=1-\alpha=0$ 차이가 음수에서 양수로 바뀐다. 이는 class cardinality가 supervision preference에 기여한다는 증거이다.

다만 이 구현은 선택하지 않은 class sample을 제거하므로 train set도 $500K$개로 바뀐다. 즉 class 수와 총 sample 수가 함께 변한다. 따라서 이 표만으로 **class 수가 유일한 원인**이라고 단정할 수는 없다. 논문에서는 “class cardinality와 함께 preference가 전환되었다”까지 주장하고, 완전한 인과 주장은 sample-count-matched control 뒤로 미루는 것이 안전하다.

---

## 4. 단순 task의 shallow representation은 얼마나 판별적인가?

### 4.1 Frozen shallow-representation probe의 정의

이 실험은 branch-side loss를 모두 끈 BYOT-shaped model을 FL로 학습한 뒤, backbone을 freeze한다. 그 다음 중앙 reference train set으로 B1/B2/B3 feature 위에 **새로운 linear classifier만 30 epochs 학습**한다.

따라서 이 수치는 native branch classifier 성능이 아니라 다음 질문에 답한다.

> “해당 depth의 frozen feature에 class를 선형 분리할 정보가 얼마나 들어 있는가?”

Teacher 열은 fresh probe가 아니라 기존 final classifier의 test accuracy이므로 branch probe와 완전히 대등한 classifier comparison은 아니다. 이는 representational diagnostic으로만 사용해야 한다.

| Dataset | Teacher native acc | Frozen B1 probe | Frozen B2 probe | Frozen B3 probe |
|---|---:|---:|---:|---:|
| CIFAR-10, $\beta=0.5$ | 75.220 | 58.075 | 73.305 | **89.095** |
| CIFAR-100, $\beta=0.5$ | 67.970 | 36.745 | 48.030 | 60.895 |

각 값은 2-seed 평균이다.

**해석.** CIFAR-10에서는 B2/B3 feature만으로도 높은 선형 분리 성능을 얻을 수 있다. 특히 B3 probe는 native final classifier보다 높다. 반면 CIFAR-100에서는 B1–B3 probe가 모두 teacher native accuracy보다 낮고, depth가 깊어질수록 격차가 점차 줄어든다.

이 결과는 “단순 task는 intermediate feature에 이미 충분한 판별 정보가 있고, 복잡한 task의 shallow exit에는 더 큰 표현 한계가 있다”는 전제와 일치한다. 그러나 이것만으로 **C100 branch CE가 final model에 해롭다는 인과 관계**가 증명되는 것은 아니다. 그 경로는 다음 causal control에서 확인한다.

---

## 5. CE, KD, feature imitation을 분리하면 무엇이 남는가?

### 5.1 Partition extension: Feature-only vs. CE + feature vs. KD + feature

아래 표는 모든 branch를 사용하고 branch-loss reduction=`sum`으로 맞춘 비교이다. Partition extension은 seed 0의 경향성 확인 실험이다.

| Dataset | Partition | Feature-only | CE + feature | KD + feature | CE − Feature | KD − CE |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | IID | 91.824 | 92.173 | **92.354** | +0.349 | +0.181 |
| CIFAR-10 | $\beta=0.5$ | **79.069** | 74.982 | 71.287 | -4.087 | -3.695 |
| CIFAR-10 | $\beta=0.1$ | **30.474** | 24.864 | 18.315 | -5.610 | -6.549 |
| CIFAR-100 | IID | 69.367 | 71.651 | **72.370** | +2.284 | +0.719 |
| CIFAR-100 | $\beta=0.5$ | 66.841 | 67.875 | **68.470** | +1.035 | +0.594 |
| CIFAR-100 | $\beta=0.1$ | **54.999** | 52.253 | 53.601 | -2.746 | +1.348 |

**해석.**

- CIFAR-10 non-IID에서는 `Feature-only > CE + feature > KD + feature`이다. CE가 KD보다 낫지만 CE도 harmful하다.
- CIFAR-100 IID와 moderate non-IID에서는 `KD + feature > CE + feature > Feature-only`이다.
- CIFAR-100 $\beta=0.1$에서는 둘 다 feature-only보다 낮지만 KD가 CE보다 덜 해롭다.

따라서 논문의 범위는 자연스럽게 나뉜다. 단순 task에서는 branch prediction loss 자체가 불필요할 수 있으므로 CE 제거 후 KD를 쓰는 방법의 적용 대상으로 삼기 어렵다. 복잡한 task에서는 branch prediction supervision이 유용할 수 있지만, 그 target은 hard CE보다 KD가 적절하다.

### 5.2 CE 효과는 heterogeneity와 task에 따라 달라진다

2 seeds를 사용한 CE mechanism ablation이다.

| Dataset | Partition | Feature-only | CE + feature | CE effect |
|---|---|---:|---:|---:|
| CIFAR-10 | IID | 91.718 | 92.435 | +0.718 |
| CIFAR-10 | $\beta=0.5$ | 77.507 | 73.987 | -3.521 |
| CIFAR-10 | $\beta=0.3$ | 71.255 | 67.414 | -3.841 |
| CIFAR-10 | $\beta=0.1$ | 29.094 | 23.333 | -5.761 |
| CIFAR-100 | IID | 69.736 | 72.351 | +2.614 |
| CIFAR-100 | $\beta=0.5$ | 66.590 | 68.310 | +1.720 |
| CIFAR-100 | $\beta=0.3$ | 64.250 | 65.288 | +1.039 |
| CIFAR-100 | $\beta=0.1$ | 53.888 | 52.196 | -1.692 |

CE는 IID에서는 두 dataset 모두 도움이 되지만, CIFAR-10에서는 moderate non-IID부터 빠르게 해로워지고 CIFAR-100에서는 $\beta=0.3$까지 이득이 남는다. 그러므로 “branch CE는 FL에서 항상 해롭다”가 아니라 **hard branch supervision의 유효 영역이 task와 heterogeneity에 의존한다**고 쓰는 것이 맞다.

### 5.3 CE weight를 줄이면 단순/복잡 task가 다르게 반응한다

$\beta=0.5$, 2 seeds 결과이다.

| Dataset | Feature-only | CE weight 0.10 | CE weight 0.25 | CE weight 0.50 | CE weight 1.00 |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | **77.507** | 76.107 | 75.439 | 74.203 | 73.987 |
| CIFAR-100 | 66.590 | 67.030 | 68.501 | **68.796** | 68.310 |

CIFAR-10에서는 작은 CE weight도 feature-only보다 낮고, CE weight가 커질수록 대체로 악화된다. CIFAR-100에서는 중간 CE weight까지 성능이 증가한 뒤 감소한다. 즉 C100에서도 one-hot signal이 완전히 무가치한 것은 아니지만, 과도한 hard-label pressure는 최적이 아니다.

### 5.4 어떤 branch에 CE를 거는가?

Feature imitation 없이 CE branch만 활성화한 $\beta=0.5$, 2-seed 결과이다.

| Dataset | Off | B1 CE | B2 CE | B3 CE | B1+B2+B3 CE |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | **77.347** | 74.693 | 73.092 | 76.272 | 74.264 |
| CIFAR-100 | 66.629 | 67.662 | **68.629** | 66.571 | 68.096 |

CIFAR-10에서는 어느 branch CE도 off를 넘지 못했고 B3가 가장 덜 해롭다. CIFAR-100에서는 B2 CE가 가장 좋고 B3 CE는 거의 변화가 없다. 단순히 “더 shallow해서 나쁘다” 또는 “더 deep해서 좋다”라는 단조로운 규칙은 성립하지 않는다. Branch 위치, 해당 prefix의 capacity, 그리고 gradient가 공유되는 범위를 함께 봐야 한다.

---

## 6. Branch loss가 final model에 영향을 주는 인과 경로

### 6.1 Shared backbone이란 무엇인가?

Shared backbone 또는 shared trunk는 input부터 각 branch가 갈라지는 지점까지 teacher와 branch가 공동으로 사용하는 convolutional blocks를 뜻한다. Branch CE/KD의 gradient가 이 parameter까지 전달되면 branch objective가 final teacher가 사용하는 representation도 바꾼다.

Branch $b$가 공유하는 parameter를 $\theta_{\le b}$라고 하면

$$
\nabla_{\theta_{\le b}}\mathcal{L}_b
=J_b(\theta_{\le b})^\top\left(p_b-t_b\right),
$$

여기서 $t_b=e_y$이면 CE, $t_b=q_T$이면 KD이다. Detach control은 $\nabla_{\theta_{\le b}}\mathcal{L}_b$를 0으로 만들고 branch-private head만 학습한다.

### 6.2 Attached vs. detached causal control

$\beta=0.5$, 500 rounds, 2 seeds의 동일 causal suite에서 먼저 네 기본 조건을 비교하면 다음과 같다.

| Dataset | Off | Feature-only | CE + feature | KD + feature |
|---|---:|---:|---:|---:|
| CIFAR-10 | 77.420 ± 2.158 | **77.756 ± 1.784** | 73.335 ± 2.613 | 71.599 ± 1.924 |
| CIFAR-100 | 66.650 ± 0.437 | 66.753 ± 0.333 | 67.672 ± 0.361 | **68.353 ± 0.429** |

CIFAR-10에서는 feature imitation만으로 final 성능이 거의 변하지 않지만 branch CE/KD를 연결하면 성능이 하락한다. CIFAR-100에서는 CE와 KD가 모두 off/feature-only보다 높고 KD가 가장 좋다. 이는 앞선 target ablation과 동일한 task-dependent ranking을 재현한다.

아래 표의 B1–B3는 같은 causal suite의 native branch accuracy이다.

| Dataset | Condition | B1 | B2 | B3 | Final Last-30 |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | Off | 9.019 | 8.614 | 9.168 | **77.771 ± 2.987** |
| CIFAR-10 | CE attached | 69.197 | 71.140 | 70.796 | 73.503 ± 2.609 |
| CIFAR-10 | CE detached | 63.224 | 70.921 | 73.393 | **77.771 ± 2.987** |
| CIFAR-10 | KD attached | 66.216 | 68.942 | 68.886 | 71.623 ± 1.693 |
| CIFAR-10 | KD detached | 61.950 | 70.015 | 73.805 | **77.771 ± 2.987** |
| CIFAR-100 | Off | 0.823 | 0.979 | 1.019 | 66.650 ± 0.437 |
| CIFAR-100 | CE attached | 50.717 | 60.678 | 66.353 | 67.570 ± 0.174 |
| CIFAR-100 | CE detached | 41.138 | 48.739 | 57.775 | 66.650 ± 0.437 |
| CIFAR-100 | KD attached | 47.766 | 58.481 | 67.019 | **68.487 ± 0.095** |
| CIFAR-100 | KD detached | 39.367 | 46.876 | 56.660 | 66.650 ± 0.437 |

Detached CE/KD의 final accuracy, loss, ECE curve는 같은 seed의 off와 bitwise-identical하다. Branch head는 학습되었지만 final은 전혀 변하지 않았다.

**직접 결론.** Branch CE/KD가 final model에 영향을 주는 경로는 branch classifier의 존재 자체가 아니라 shared backbone으로 전달되는 gradient이다.

**아직 설명하지 않는 것.** 이 control은 “왜 C100에서는 KD gradient가 CE gradient보다 좋은가”까지 설명하지 않는다. 그 차이는 target, temperature, gradient direction/scale을 추가로 분리해야 한다.

### 6.3 Branch accuracy와 final accuracy는 같은 목적함수가 아니다

독립 branch-logit probe rerun의 Last-30 결과이다.

| Dataset | Partition | $\alpha$ | B1 | B2 | B3 | Final |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | $\beta=0.5$ | 0 | 70.763 | 72.375 | 72.009 | 74.737 |
| CIFAR-10 | $\beta=0.5$ | 1 | 68.972 | 70.534 | 69.856 | 72.991 |
| CIFAR-100 | $\beta=0.5$ | 0 | 50.527 | 60.473 | 66.751 | 67.918 |
| CIFAR-100 | $\beta=0.5$ | 1 | 47.818 | 58.294 | **66.944** | **68.723** |
| CIFAR-100 | $\beta=0.1$ | 0 | 39.553 | 46.064 | 48.769 | 52.068 |
| CIFAR-100 | $\beta=0.1$ | 1 | 34.246 | 42.111 | **51.544** | **53.315** |

C100에서는 $\alpha=1$에서 B1/B2 accuracy가 감소해도 B3와 final accuracy는 증가한다. 이는 KD가 branch를 독립 local classifier로 최대화하기보다, shared representation에 다른 auxiliary constraint를 제공한다는 해석과 일치한다. 특히 **branch accuracy가 높다는 사실만으로 final teacher에 유익한 supervision이라고 판단할 수 없다.**

### 6.4 Global teacher의 NLL과 ECE

$\beta=0.5$, 2 seeds, Last-30 평균이다. NLL과 ECE는 낮을수록 좋다.

| Dataset | Condition | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|
| CIFAR-10 | Feature-only | **77.109** | **0.674** | **0.054** |
| CIFAR-10 | CE + feature | 72.989 | 0.788 | 0.066 |
| CIFAR-10 | KD + feature | 70.775 | 0.853 | 0.080 |
| CIFAR-100 | Feature-only | 66.871 | 1.221 | **0.038** |
| CIFAR-100 | CE + feature | 67.864 | 1.162 | 0.063 |
| CIFAR-100 | KD + feature | **68.551** | **1.141** | 0.066 |

CIFAR-10에서는 KD가 accuracy뿐 아니라 NLL과 ECE도 가장 나쁘다. 숨겨진 calibration 이득은 관찰되지 않는다.

CIFAR-100에서는 KD가 CE보다 accuracy와 NLL을 모두 개선하지만 ECE는 0.003 높다. 따라서 KD 이득은 단순 confidence inflation만으로 설명되지는 않지만, calibration이 모든 지표에서 개선된 것도 아니다.

---

## 7. Class frequency에 따른 post-local branch distribution

### 7.1 측정 프로토콜

- 각 client에서 local class count가 client 평균의 0.5배 미만이면 `Low`, 1.5배 초과이면 `High`, 나머지는 `Mid`로 나눈다.
- 동일한 common-reference sample을 round-start global model에서 갈라진 client model들에 입력한다.
- Local training 후, aggregation 전에 B1/B2/B3와 teacher의 full logits을 저장한다.
- 아래 endpoint 표는 rounds 470, 480, 490과 seeds 0, 1을 합친 6-checkpoint macro이다.
- 따라서 `Low/High`는 sample의 global rarity가 아니라 **해당 client에서 그 class가 locally rare/frequent한가**를 뜻한다.

### 7.2 CIFAR-10, $\beta=0.5$

`H`는 normalized entropy, `p_y`는 true-label probability, `JS`는 client JS-to-mean, `Var(z)`는 raw-logit variance이다.

| Group | Head | $H_{\alpha=0}$ | $H_{\alpha=1}$ | $p_{y,\alpha=0}$ | $p_{y,\alpha=1}$ | $JS_{\alpha=0}$ | $JS_{\alpha=1}$ | $Var(z)_{\alpha=0}$ | $Var(z)_{\alpha=1}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| High | B1 | 0.126 | 0.458 | 0.804 | 0.619 | 0.056 | 0.044 | 3.375 | 0.472 |
| High | B2 | 0.120 | 0.406 | 0.808 | 0.676 | 0.059 | 0.037 | 3.000 | 0.425 |
| High | B3 | 0.105 | 0.366 | 0.823 | 0.709 | 0.057 | 0.039 | 2.104 | 0.387 |
| High | Teacher | 0.098 | 0.107 | 0.828 | 0.814 | 0.055 | 0.059 | 2.093 | 1.876 |
| Low | B1 | 0.264 | 0.602 | 0.179 | 0.159 | 0.212 | 0.107 | 7.085 | 0.973 |
| Low | B2 | 0.259 | 0.584 | 0.216 | 0.201 | 0.213 | 0.111 | 6.886 | 0.920 |
| Low | B3 | 0.245 | 0.530 | 0.236 | 0.228 | 0.214 | 0.124 | 5.285 | 0.944 |
| Low | Teacher | 0.240 | 0.245 | 0.247 | 0.259 | 0.211 | 0.205 | 5.261 | 4.270 |

### 7.3 CIFAR-100, $\beta=0.5$

| Group | Head | $H_{\alpha=0}$ | $H_{\alpha=1}$ | $p_{y,\alpha=0}$ | $p_{y,\alpha=1}$ | $JS_{\alpha=0}$ | $JS_{\alpha=1}$ | $Var(z)_{\alpha=0}$ | $Var(z)_{\alpha=1}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| High | B1 | 0.251 | 0.677 | 0.422 | 0.213 | 0.143 | 0.070 | 2.678 | 0.431 |
| High | B2 | 0.222 | 0.647 | 0.506 | 0.269 | 0.125 | 0.066 | 1.957 | 0.341 |
| High | B3 | 0.231 | 0.692 | 0.513 | 0.282 | 0.127 | 0.055 | 1.231 | 0.200 |
| High | Teacher | 0.196 | 0.200 | 0.533 | 0.521 | 0.129 | 0.134 | 1.301 | 1.237 |
| Low | B1 | 0.319 | 0.739 | 0.083 | 0.050 | 0.264 | 0.102 | 3.638 | 0.593 |
| Low | B2 | 0.313 | 0.743 | 0.127 | 0.067 | 0.252 | 0.096 | 2.691 | 0.465 |
| Low | B3 | 0.342 | 0.811 | 0.135 | 0.067 | 0.250 | 0.074 | 1.753 | 0.286 |
| Low | Teacher | 0.304 | 0.308 | 0.153 | 0.150 | 0.258 | 0.262 | 1.826 | 1.709 |

### 7.4 Rare class의 client prediction은 실제로 더 다르다

아래 표는 B1–B3의 client JS-to-mean을 평균한 값이다. Teacher는 별도로 제시한다.

| Dataset | Partition | $\alpha$ | Branch Low JS | Branch High JS | Branch Low−High | Teacher Low JS | Teacher High JS | Teacher Low−High |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | $\beta=0.5$ | 0 | 0.213 | 0.057 | 0.156 | 0.211 | 0.055 | 0.157 |
| CIFAR-10 | $\beta=0.5$ | 1 | 0.114 | 0.040 | 0.074 | 0.205 | 0.059 | 0.146 |
| CIFAR-10 | $\beta=0.1$ | 0 | 0.339 | 0.057 | 0.283 | 0.342 | 0.056 | 0.286 |
| CIFAR-10 | $\beta=0.1$ | 1 | 0.217 | 0.058 | 0.158 | 0.334 | 0.065 | 0.268 |
| CIFAR-100 | $\beta=0.5$ | 0 | 0.255 | 0.131 | 0.124 | 0.258 | 0.129 | 0.130 |
| CIFAR-100 | $\beta=0.5$ | 1 | 0.091 | 0.064 | 0.027 | 0.262 | 0.134 | 0.128 |
| CIFAR-100 | $\beta=0.1$ | 0 | 0.388 | 0.128 | 0.260 | 0.389 | 0.120 | 0.269 |
| CIFAR-100 | $\beta=0.1$ | 1 | 0.158 | 0.076 | 0.083 | 0.392 | 0.124 | 0.269 |

이 표에서 다음은 직접 관측이다.

1. 모든 dataset과 $\alpha$에서 Low JS가 High JS보다 크다. 즉 locally rare class에 대한 client probability distribution은 실제로 더 다르다.
2. $\beta=0.1$에서 Low−High gap이 $\beta=0.5$보다 크다. Heterogeneity가 심할수록 rare-class disagreement가 커진다.
3. $\alpha=1$은 branch Low JS와 Low−High gap을 크게 줄인다.
4. Teacher Low−High gap은 거의 유지된다. 예를 들어 C100 $\beta=0.1$에서 0.269와 0.269이다.

따라서 $\alpha=1$의 효과는 “client들이 같은 teacher를 만들었다”가 아니다. 더 정확한 설명은 다음과 같다.

> 각 client teacher는 여전히 서로 다른 local 중심을 유지하지만, KD-supervised branch는 그 teacher를 그대로 복제하기보다 더 soft한 probability map을 형성하고, 그 결과 common reference에 대한 branch probability dispersion이 감소한다.

### 7.5 Softness와 dispersion은 구분해야 한다

C100 $\beta=0.5$ Low B1을 예로 들면, entropy는 0.319에서 0.739로 증가하고 JS는 0.264에서 0.102로 감소한다.

- Entropy 증가는 client A, B 각각의 distribution이 더 평평해졌음을 뜻한다.
- JS 감소는 A와 B의 probability distribution이 서로 더 가까워졌음을 뜻한다.
- Raw-logit variance가 3.638에서 0.593으로 감소한 것은 logit scale까지 함께 작아졌음을 뜻한다.

Probability JS는 logit scale compression만으로도 감소할 수 있다. 따라서 이것을 곧바로 “동일한 semantic direction을 학습했다”라고 부르면 과도하다. 그 주장은 client 간 centered-logit cosine까지 함께 봐야 한다. 현재 표가 직접 보이는 것은 **softening과 probability-level dispersion 감소**이다.

또한 CIFAR-10에서도 같은 softening과 dispersion 감소가 나타나지만 final accuracy는 감소한다. 따라서 이 현상은 KD의 작동 방식은 설명하지만, 그 자체로 성능 향상의 충분조건은 아니다.

### 7.6 Depth별 branch–teacher 관계

아래 값은 $\beta=0.5$에서 Low/Mid/High group 평균이다. `JS@KD`는 실제 KD temperature에서 branch와 teacher probability를 비교한다.

| Dataset | $\alpha$ | Head | Centered-logit cosine ↑ | Normalized logit L2 ↓ | JS@KD ↓ | Non-target JS@KD ↓ |
|---|---:|---|---:|---:|---:|---:|
| CIFAR-10 | 0 | B1 | 0.819 | 2.034 | 0.174 | 0.235 |
| CIFAR-10 | 0 | B2 | 0.947 | 1.061 | 0.061 | 0.094 |
| CIFAR-10 | 0 | B3 | **0.981** | **0.630** | **0.016** | **0.043** |
| CIFAR-10 | 1 | B1 | 0.769 | 2.217 | 0.195 | 0.249 |
| CIFAR-10 | 1 | B2 | 0.898 | 1.954 | 0.092 | 0.146 |
| CIFAR-10 | 1 | B3 | **0.928** | **1.899** | **0.039** | **0.117** |
| CIFAR-100 | 0 | B1 | 0.650 | 2.305 | 0.385 | 0.446 |
| CIFAR-100 | 0 | B2 | 0.770 | 1.732 | 0.286 | 0.351 |
| CIFAR-100 | 0 | B3 | **0.936** | **0.766** | **0.087** | **0.116** |
| CIFAR-100 | 1 | B1 | 0.633 | 1.587 | 0.397 | 0.438 |
| CIFAR-100 | 1 | B2 | 0.721 | 1.451 | 0.330 | 0.382 |
| CIFAR-100 | 1 | B3 | **0.847** | **1.387** | **0.208** | **0.265** |

B1→B3로 갈수록 teacher와의 centered-logit cosine은 커지고 JS는 작아진다. 더 깊은 feature가 teacher class contrast를 재현하기 쉽다는 해석과 일치한다.

그러나 이 depth ordering은 $\alpha=0$에서도 더 강하게 나타난다. 또한 $\alpha=1$이 $\alpha=0$보다 branch–teacher JS를 낮추지 않았다. 따라서 다음과 같이 구분해야 한다.

- **직접 관측:** B3가 B1보다 teacher output에 가깝다.
- **직접 관측:** $\alpha=1$은 branch끼리 또는 branch–teacher를 모두 같은 predictor로 collapse시키지 않는다.
- **추정:** KD는 각 depth의 capacity 안에서 soft auxiliary target을 제공하며, shallow/deep predictor의 역할 분화를 허용한다.

마지막 문장은 plausible mechanism이지만 output metric만으로 학습 원리를 인과적으로 증명한 것은 아니다.

---

## 8. Gradient 분석: 무엇이 원인이 아니었는가?

### 8.1 동일 checkpoint/client/batch에서 branch CE/KD와 teacher CE 비교

아래 cosine은 같은 model checkpoint와 같은 client batch에서 shared prefix parameter에 대해 계산했다. Rounds 50–450의 probe 평균을 seed별로 구한 뒤 2 seeds를 평균했다.

| Dataset | Trained condition | Branch | $\cos(g_{\mathrm{branchCE}},g_{\mathrm{teacherCE}})$ | $\cos(g_{\mathrm{branchKD}},g_{\mathrm{teacherCE}})$ |
|---|---|---|---:|---:|
| CIFAR-10 | Feature-only | B1 | 0.015 | 0.009 |
| CIFAR-10 | Feature-only | B2 | -0.031 | -0.001 |
| CIFAR-10 | Feature-only | B3 | 0.043 | 0.025 |
| CIFAR-10 | CE-trained | B1 | 0.723 | 0.014 |
| CIFAR-10 | CE-trained | B2 | 0.910 | 0.112 |
| CIFAR-10 | CE-trained | B3 | 0.956 | 0.411 |
| CIFAR-10 | KD-trained | B1 | 0.639 | 0.138 |
| CIFAR-10 | KD-trained | B2 | 0.739 | 0.133 |
| CIFAR-10 | KD-trained | B3 | 0.797 | 0.216 |
| CIFAR-100 | Feature-only | B1 | -0.007 | -0.021 |
| CIFAR-100 | Feature-only | B2 | -0.035 | -0.034 |
| CIFAR-100 | Feature-only | B3 | -0.004 | -0.003 |
| CIFAR-100 | CE-trained | B1 | 0.470 | 0.099 |
| CIFAR-100 | CE-trained | B2 | 0.676 | 0.091 |
| CIFAR-100 | CE-trained | B3 | 0.896 | 0.055 |
| CIFAR-100 | KD-trained | B1 | 0.291 | 0.148 |
| CIFAR-100 | KD-trained | B2 | 0.405 | 0.171 |
| CIFAR-100 | KD-trained | B3 | 0.597 | 0.159 |

CE gradient가 teacher CE와 훨씬 더 정렬되어 있다. 이는 두 loss가 같은 one-hot target을 사용하기 때문에 자연스럽다. C100에서 KD가 더 좋은 final accuracy를 만든다는 사실은 **KD가 teacher CE와 같은 방향으로 더 잘 정렬되기 때문이 아니다.** 오히려 KD는 teacher CE와 다른 auxiliary direction과 더 작은 scale을 제공한다.

### 8.2 Client 간 shared-gradient dispersion

아래는 single-seed decomposed probe의 rounds 50–450 평균이다. `Relative`가 낮고 `Cosine`이 높을수록 선택 client들의 gradient가 더 일치한다. `KD/CE norm`은 같은 checkpoint에서 KD gradient norm을 CE gradient norm으로 나눈 값이다.

| Dataset | Partition | Trained $\alpha$ | CE relative ↓ | CE cosine ↑ | KD relative ↓ | KD cosine ↑ | KD/CE norm |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | $\beta=0.5$ | 0 | 2.009 | 0.607 | **1.424** | **0.672** | 0.130 |
| CIFAR-10 | $\beta=0.5$ | 1 | 2.240 | 0.586 | **1.560** | **0.661** | 0.164 |
| CIFAR-100 | $\beta=0.5$ | 0 | **2.501** | **0.542** | 2.923 | 0.517 | 0.265 |
| CIFAR-100 | $\beta=0.5$ | 1 | **1.010** | **0.711** | 1.880 | 0.600 | 0.228 |
| CIFAR-100 | $\beta=0.1$ | 0 | 2.540 | 0.543 | **1.929** | **0.599** | 0.158 |
| CIFAR-100 | $\beta=0.1$ | 1 | 1.703 | 0.623 | **1.320** | **0.676** | 0.247 |

KD gradient가 client 간에 더 잘 정렬된다는 패턴은 CIFAR-10과 C100 $\beta=0.1$에서는 나타나지만 C100 $\beta=0.5$에서는 반대이다. 따라서 **cross-client gradient alignment가 KD 이득의 보편적 원인이라는 주장도 지지되지 않는다.**

더 일관된 차이는 KD gradient norm이 CE의 약 13–27%라는 점이다. 이는 C100에서 KD의 이득을 이해할 때 direction뿐 아니라 **auxiliary gradient strength**가 중요하다는 후속 factorial 결과와 연결된다.

### 8.3 한 FL round가 common-reference representation을 얼마나 바꾸는가?

CIFAR-10 $\beta=0.5$ causal run에서 round-start global model과 local training 및 FedAvg가 끝난 post-aggregation global model에 동일한 test reference를 입력했다. Cosine과 linear CKA는 높을수록 전후 representation이 비슷하고, relative delta는 낮을수록 변화량이 작다. 아래 값은 rounds 50–450의 2-seed 평균이다.

| Condition | Depth | Pre/post cosine ↑ | Relative delta ↓ | Linear CKA ↑ |
|---|---|---:|---:|---:|
| Off | B1 | 0.985 | 0.174 | 0.968 |
| Off | B2 | 0.978 | 0.213 | 0.937 |
| Off | B3 | 0.960 | 0.292 | 0.819 |
| Off | Teacher | 0.885 | 0.517 | 0.799 |
| CE attached | B1 | 0.934 | 0.391 | 0.882 |
| CE attached | B2 | 0.905 | 0.478 | 0.812 |
| CE attached | B3 | 0.864 | 0.564 | 0.739 |
| CE attached | Teacher | 0.858 | 0.568 | 0.761 |
| KD attached | B1 | 0.935 | 0.405 | 0.837 |
| KD attached | B2 | 0.905 | 0.506 | 0.762 |
| KD attached | B3 | 0.853 | 0.610 | 0.684 |
| KD attached | Teacher | 0.841 | 0.611 | 0.730 |

Branch supervision을 연결하면 off보다 한 round 동안 shared representation이 더 크게 바뀐다. 특히 KD attached는 B2/B3/teacher에서 가장 큰 relative delta와 가장 낮은 CKA를 보이고, final ranking도 `Off > CE > KD`이다.

이 표는 **simple task에서 불필요한 branch objective가 이미 유용한 representation을 더 크게 교란할 수 있다**는 설명과 일치한다. 다만 변화량과 성능 저하의 상관을 보인 것이지 “representation movement 자체가 유일한 원인”을 조작한 실험은 아니다. 또한 같은 metric의 C100 counterpart는 해당 run에서 기록되지 않았으므로, 이 표를 C100 KD 이득의 설명으로 역으로 확장하면 안 된다.

---

## 9. KD target의 어떤 정보가 필요한가?

### 9.1 Target ablation

$\beta=0.5$, 500 rounds, 2 seeds이다.

| Condition | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Feature-only | **77.109 ± 2.772** | 66.871 ± 0.043 |
| CE + feature | 72.989 ± 2.818 | 67.864 ± 0.016 |
| Label-smoothed CE + feature | 73.702 ± 2.102 | 67.768 ± 0.086 |
| Full KD + feature | 70.775 ± 0.724 | 68.551 ± 0.115 |
| Teacher-mass + uniform non-target KD | 72.695 ± 1.782 | 68.628 ± 0.397 |
| Teacher-correct sample KD | 72.387 ± 2.466 | **68.898 ± 0.000** |
| Teacher-correct and confidence $\ge 0.8$ KD | 74.152 ± 2.061 | 68.267 ± 0.026 |

각 target의 의미는 다음과 같다.

- `Full KD`: teacher의 전체 class probability vector를 사용한다.
- `Teacher-mass + uniform non-target`: sample별 teacher 정답 확률 $q_T(y\mid x)$는 보존하되, 나머지 확률은 모든 오답 class에 균등 분배한다.
- `Teacher-correct`: teacher가 정답을 맞힌 sample에만 KD를 적용한다.
- `Correct & confidence >= 0.8`: 위 조건에 confidence threshold를 추가한다.

**직접 관측.**

1. C100에서 full KD와 uniform-non-target KD가 거의 같다: 68.551 vs. 68.628.
2. 따라서 세부 non-target class relation은 이 설정에서 필수적이지 않다.
3. 고정 label smoothing은 C100 KD 이득을 재현하지 못한다: 67.768 vs. 68.551.
4. Teacher-correct filtering은 C100에서 가장 좋지만, confidence 0.8 threshold를 추가하면 0.631 point 감소한다.
5. C10에서는 confidence filtering이 full KD보다 낫지만 여전히 feature-only보다 2.957 point 낮다.

따라서 confidence는 universal reliability proxy가 아니다. 높은 confidence는 reliability일 수도 있지만 local bias 또는 overconfidence일 수도 있어 dataset과 heterogeneity에 따라 calibration이 필요하다.

### 9.2 Target adaptivity × student temperature × gradient scale factorial

C100 $\beta=0.5$, 500 rounds, 2 seeds이다. Teacher temperature는 0.5로 고정했다.

- `Adaptive mass`: sample별 $q_T(y\mid x)$를 유지하고 non-target은 균등화한다.
- `Batch-mean mass`: $q_T(y\mid x)$ 대신 batch 평균 $\bar q_y$를 모든 sample에 사용해 sample별 adaptivity를 제거한다.
- `Native`: KL multiplier $T_s^2$.
- `Unit-gradient`: KL multiplier $T_s$로 leading temperature gradient factor를 1로 맞춘다.

| Target mass | Student $T_s$ | Scale | Last-30 accuracy |
|---|---:|---|---:|
| Adaptive | 0.5 | Native $T_s^2$ | 68.481 ± 0.083 |
| Adaptive | 0.5 | Unit-gradient $T_s$ | 68.062 ± 0.012 |
| Adaptive | 1.0 | Native $T_s^2$ | 67.868 ± 0.084 |
| Adaptive | 1.0 | Unit-gradient $T_s$ | 67.868 ± 0.084 |
| Batch-mean | 0.5 | Native $T_s^2$ | **68.984 ± 0.014** |
| Batch-mean | 0.5 | Unit-gradient $T_s$ | 68.381 ± 0.342 |
| Batch-mean | 1.0 | Native $T_s^2$ | 68.003 ± 0.155 |
| Batch-mean | 1.0 | Unit-gradient $T_s$ | 68.003 ± 0.155 |

$T_s=1$에서는 두 scale 정의가 동일하므로 결과도 동일하다. $T_s=0.5$에서는 native scale이 unit-gradient보다 adaptive target에서 0.419, batch-mean target에서 0.603 point 높다. 또한 batch-mean target이 adaptive target보다 높다.

이 결과는 기존 mechanistic hypothesis를 다음과 같이 수정한다.

- **지지되지 않음:** C100 KD 이득의 핵심은 sample마다 다른 teacher 정답 확률이다.
- **지지되지 않음:** 세부 non-target class relation이 반드시 필요하다.
- **지지됨:** temperature와 KD gradient scale은 실제 성능을 크게 바꾼다.
- **가장 유력한 설명:** one-hot branch CE 대신 더 soft하고 더 약한 temperature-scaled auxiliary gradient를 주는 것이 중요하다.

Batch-mean target도 batch별 teacher 평균을 사용하므로 완전히 고정된 label smoothing과 같지는 않다. 따라서 “teacher 정보가 전혀 필요 없다”까지는 말할 수 없다. 다만 현재 결과는 논문의 설명 중심을 **dark knowledge**보다 **soft auxiliary constraint와 gradient geometry**에 두어야 함을 보여준다.

---

## 10. CE 제거 후에는 KD strength를 별도로 조절해야 한다

### 10.1 Legacy $T_{KD}=0.5$ fixed-$\lambda$ sweep

C100, KD-only + feature, single logged runs의 Last-30이다.

| Partition | $\lambda=0$ | 0.01 | 0.1 | 0.3 | 1 | 3 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 69.367 | 69.655 | 69.773 | 71.421 | **72.370** | 71.961 | 70.982 |
| $\beta=0.5$ | 66.988 | 67.184 | 67.388 | **68.530** | 68.049 | 68.503 | 65.685 |
| $\beta=0.3$ | 63.412 | 64.146 | 64.441 | **66.226** | 66.030 | 65.495 | 63.994 |
| $\beta=0.1$ | **54.999** | 54.424 | 54.507 | 54.407 | 53.601 | 53.714 | 51.076 |

Best $\lambda$는 IID의 1에서 $\beta=0.5/0.3$의 0.3, $\beta=0.1$의 0으로 이동한다. 특히 $\lambda=10$은 모든 partition에서 악화된다.

### 10.2 $T_{KD}=1$ compact fixed-$\lambda$ rerun

Adaptive implementation과 temperature를 맞추기 위해 teacher/student KD temperature를 모두 1로 고정한 seed-0 rerun이다. 모든 $\lambda$는 round 0부터 고정되며 warm-up이 없다.

| Partition | $\lambda=0.1$ | 0.3 | 0.5 | 0.7 | 1.0 | 3.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 71.396 | **72.679** | 71.888 | 71.848 | 71.486 | 70.276 | 70.262 |
| $\beta=0.5$ | **68.462** | 68.192 | 68.036 | 67.648 | 67.666 | 67.305 | 66.513 |
| $\beta=0.3$ | 65.811 | **66.121** | 65.273 | 65.351 | 65.071 | 64.706 | 63.542 |
| $\beta=0.1$ | **55.230** | 53.433 | 52.252 | 52.443 | 52.382 | 51.986 | 50.419 |

$T=1$에서도 큰 $\lambda$가 무너지는 현상과 severe non-IID에서 작은 $\lambda$가 필요한 현상은 유지된다. 다만 최적값과 절대 정확도는 $T=0.5$와 달라진다.

### 10.3 `alpha`와 `lambda`를 혼동하면 안 되는 이유

- Blend loss에서 $\alpha=1$은 branch CE를 없애고 branch KD만 남긴다.
- KD-only loss에서 $\lambda$는 남은 KD의 절대 gradient strength를 조절한다.
- 따라서 “$\alpha=1$이 좋다”는 결과는 **hard branch CE 제거**를 지지한다.
- “큰 $\lambda$가 좋다”는 결론은 따로 검증해야 하며, 실제로 severe non-IID에서는 반대이다.

Factorial 실험에서 $T_s=0.5$와 1.0 사이에 약 0.6–1.0 point 차이가 났으므로, temperature가 달라진 실험의 $\lambda$를 같은 축으로 직접 비교해서는 안 된다.

---

## 11. 논문용 통합 해석

### 11.1 단순 task를 연구의 주 적용 범위에서 제외하는 이유

CIFAR-10과 Fashion-MNIST에서는 intermediate feature가 비교적 이른 depth부터 높은 선형 분리 가능성을 가진다. CIFAR-10 frozen probe에서 B2와 B3는 각각 73.305%, 89.095%를 기록했다. 이 환경에서 teacher의 soft target이 제공하는 추가 구조는 필수적이지 않으며, non-IID에서는 branch prediction loss가 shared feature를 과도하게 바꿀 수 있다.

실제로 CIFAR-10 $\beta=0.5$에서 feature-only는 77.109%, CE+feature는 72.989%, KD+feature는 70.775%였다. 즉 CE와 KD 모두 성능을 낮추고 KD가 더 크게 낮춘다. 따라서 단순 task에서 관찰된 $\alpha$ 감소 선호를 “CE가 본질적으로 좋은 supervision”이라고 해석해서는 안 된다. **단순 task에서는 shallow feature가 이미 충분히 판별적이어서 추가 soft relation supervision의 필요성이 낮고, branch prediction objective 자체가 불필요할 수 있다**고 정리하는 것이 현재 증거와 가장 잘 맞는다.

### 11.2 복잡한 task에서 branch CE를 제거하는 이유

CIFAR-100에서는 frozen B1/B2/B3 probe가 36.745%, 48.030%, 60.895%로 final teacher 67.970%보다 낮다. 즉 shallow exit가 final과 같은 one-hot 판별을 수행하기에는 더 큰 capacity gap이 남아 있다.

이 상황에서 branch CE는 모든 sample에 $p_b-e_y$를 부과해 shallow prefix가 즉시 정답 class 하나를 분리하도록 요구한다. 반면 KD는 $p_b^{(T)}-q_T^{(T)}$를 사용하므로 target과 gradient scale이 더 완화된다. C100 $\beta=0.5$에서 CE+feature는 67.864%, KD+feature는 68.551%였고, $\alpha$ sweep에서도 모든 partition에서 $\alpha=1>\alpha=0$였다.

Detach control은 이 차이가 branch classifier 자체가 아니라 shared backbone gradient를 통해 final model에 전달됨을 보인다. 동시에 paired-gradient probe는 KD가 teacher CE와 더 같은 방향이어서 좋은 것이 아님을 보여준다. 따라서 현재의 가장 정확한 해석은 다음과 같다.

> 복잡한 task에서는 shallow branch를 별도의 hard-label local classifier로 최적화하는 것보다, teacher로부터 얻은 soft하고 scale이 조절된 auxiliary gradient를 shared prefix에 전달하는 것이 final representation에 더 적절하다.

### 11.3 Non-IID class frequency가 이 설명에 더하는 내용

Locally rare class는 frequent class보다 client 간 prediction JS가 크며, $\beta$가 작을수록 그 차이가 커진다. C100 $\beta=0.1$에서 branch Low−High JS gap은 $\alpha=0$일 때 0.260이었다. $\alpha=1$에서는 0.083으로 감소했다.

동시에 branch entropy는 크게 증가하지만 teacher entropy와 teacher JS gap은 거의 유지된다. 따라서 KD-only branch supervision은 local teacher를 모두 동일하게 만드는 것이 아니라, **불완전하고 client-dependent한 local signal을 branch가 one-hot으로 증폭하지 않도록 완화하는 역할**을 한다고 해석할 수 있다.

다만 같은 dispersion 감소가 CIFAR-10에서도 나타나면서 정확도는 하락한다. 그러므로 rare-class stabilization은 KD의 보조 메커니즘이지, dataset에 무관한 성능 향상 법칙은 아니다.

### 11.4 최종 design implication

복잡한 task에 대한 설계는 두 단계로 분리해야 한다.

1. **Supervision type:** branch hard-label CE를 제거하고 KD-dominant 또는 KD-only objective를 사용한다.
2. **Supervision strength:** 남은 KD의 $\lambda$, temperature, loss scale을 heterogeneity에 맞게 조절한다.

Moderate non-IID에서는 KD가 CE와 feature-only를 모두 넘지만, severe non-IID에서는 feature-only가 더 좋을 수 있다. 따라서 최종 방법은 “KD를 항상 강하게 적용”하는 방식보다, client가 접근 가능한 reliability/skew signal을 이용해 KD strength를 줄이는 방향이 타당하다. Confidence 하나만으로 reliability를 정의하는 것은 target ablation 결과상 안전하지 않다.

---

## 12. 주장 가능 범위 체크리스트

| 주장 | 현재 판정 | 근거 또는 제한 |
|---|---|---|
| 복잡한 task에서 branch CE 제거 방향이 유리하다 | **직접 지지** | C100, TinyImageNet, ImageNet100-64에서 $\alpha=1>\alpha=0$ |
| 모든 복잡한 non-IID에서 $\alpha=1$이 plain보다 좋다 | **거짓** | C100과 ImageNet100-64의 $\beta=0.1$에서 plain 미달 |
| 단순 task에서는 branch CE가 final 성능을 높인다 | **일반적으로 지지되지 않음** | C10 non-IID에서 feature-only > CE+feature |
| 단순 task에서는 CE가 KD보다 낫다 | **non-IID에서 직접 지지** | C10/FMNIST의 $\alpha$ 경향 및 target ablation |
| Rare class prediction은 client마다 더 다르다 | **직접 지지** | 모든 non-IID 표에서 Low JS > High JS |
| $\alpha=1$은 branch prediction을 더 soft하게 만든다 | **직접 지지** | Branch entropy 증가와 $p_y$ 감소 |
| $\alpha=1$은 client 간 branch probability dispersion을 줄인다 | **직접 지지** | Branch JS-to-mean 감소, 특히 Low group |
| $\alpha=1$은 client teacher들도 서로 같게 만든다 | **지지되지 않음** | Teacher JS Low−High gap이 거의 유지됨 |
| KD는 branch를 teacher와 더 같게 만든다 | **endpoint 결과상 지지되지 않음** | $\alpha=1$의 branch–teacher JS가 $\alpha=0$보다 낮지 않음 |
| B3는 B1보다 teacher output을 잘 근사한다 | **직접 지지** | 두 dataset과 두 $\alpha$에서 B3 cosine↑, JS↓ |
| Branch loss의 final 효과는 shared backbone gradient를 통한다 | **인과적으로 지지** | Detached CE/KD final curve가 off와 bitwise-identical |
| KD가 좋은 이유는 teacher CE gradient와 더 잘 정렬되기 때문이다 | **반증** | Branch CE–teacher CE cosine이 KD–teacher CE보다 큼 |
| KD의 세부 non-target class relation이 C100 이득의 핵심이다 | **지지되지 않음** | Uniform non-target와 full KD가 유사 |
| Sample별 teacher softness가 핵심이다 | **지지되지 않음** | Batch-mean target이 adaptive target보다 높음 |
| Temperature와 gradient scale이 중요하다 | **직접 지지** | C100 factorial에서 0.4–1.0 point 차이 |
| Confidence는 universal KD reliability proxy이다 | **반증** | Confidence filtering 효과가 C10/C100에서 다르고 C100 correct-only보다 낮음 |
| CE 제거 후 KD를 강하게 할수록 좋다 | **반증** | Fixed-$\lambda$ sweep에서 큰 $\lambda$ 붕괴 |

---

## 13. 결과 출처와 집계 기준

| 분석 | 주요 로그 또는 요약 | 집계 |
|---|---|---|
| Main $\alpha$ sweep | `logs/alpha/logs_blend_alpha_generalization`, `logs/generalization/logs_dataset_model_generalization`, `logs/alpha/logs_fmnist_alpha_r500` | 주로 Last-30 |
| Plain baseline | `logs/baseline/logs_plain_dataset_generalization`, `logs/baseline/logs_plain_imagenet100_64` | 같은 horizon의 Last-30 |
| Class-count | `logs/analysis/logs_cifar100_class_count_alpha_r500` | 3-seed Last-30 mean ± SD |
| Frozen shallow probe | `logs/analysis/logs_frozen_shallow_representation_probe_r500` | 2-seed probe accuracy mean |
| CE mechanism | `logs/analysis/logs_branch_ce_mechanism_ablation_r500` | 2-seed Last-30 mean |
| Target ablation | `logs/analysis/logs_branch_target_ablation_r500` | 2-seed Last-30 mean ± SD; partition extension은 seed 0 |
| Shared-backbone causal control | `logs/analysis/logs_branch_supervision_causal_suite_r500`, `logs/analysis/logs_cifar10_branch_detach_control_r500` | 2-seed Last-30 mean ± SD |
| Branch accuracy/logit probe | `logs/analysis/logs_alpha_branch_logit_probe_r500` | Last-30 |
| Full post-local logits | `logs/analysis/logs_postlocal_branch_distribution_full_logits_r500` | rounds 470/480/490 × seeds 0/1 |
| Full-logit derived summary | `logs/analysis/full_branch_logit_relation_summary.csv` | 6-checkpoint macro |
| Paired branch gradient | `logs/analysis/logs_paired_branch_gradient_probe_r500` | rounds 50–450, 2-seed mean |
| Decomposed client gradient | `logs/analysis/logs_alpha_branch_ce_kd_gradient_probe_r500` | rounds 50–450, seed 0 |
| KD target/temperature/scale factorial | `logs/analysis/logs_cifar100_kd_factorial_clean_r500` | 2-seed Last-30 mean ± SD |
| Legacy fixed $\lambda$, $T=0.5$ | `logs/lambda/adaptive/logs_kd_lambda_sweep`; $\beta=0.1$의 누락 값은 `logs/alpha/logs_byot_beta_kd_only_alpha` | Last-30 |
| Fixed $\lambda$, $T=1$ | `logs/lambda/analysis/logs_cifar100_fixed_lambda_t1_compact`, `logs/lambda/adaptive/logs_cifar100_fixed_lambda1_no_warmup` | seed-0 Last-30 |

### 이 문서에서 의도적으로 제외한 motivation 실험

- Client 수 또는 client당 local sample 수에 따른 intermediate representation 저하
- Fixed centralized checkpoint에서 local $n$을 바꾼 probe
- End-to-end local-$n$ FL 및 matched-$K\times n$ 실험
- Round-budget / checkpoint-fixed CKA 및 local-probe refit 분석

이 실험들은 “왜 FL에서 intermediate representation supervision이 필요한가”라는 motivation을 위한 것이므로 본 analysis 초안에는 포함하지 않았다.

---

## 14. 논문에 넣을 때의 권장 표현

### 권장 핵심 문장

> Our results show that the optimal form of branch supervision is task- and heterogeneity-dependent. On simple non-IID tasks, both hard-label CE and KD can be detrimental relative to feature-only supervision, although CE remains less harmful than KD. In contrast, on complex tasks, replacing hard branch labels with KD consistently improves over CE-supervised BYOT. Post-local diagnostics show that KD softens branch predictions and reduces client-level probability dispersion, particularly for locally rare classes, while leaving teacher disagreement largely unchanged. Causal detach controls confirm that these effects reach the final classifier through shared-backbone gradients. Additional target and gradient ablations indicate that the gain is better explained by a softened, temperature- and scale-controlled auxiliary gradient than by fine-grained non-target class relations alone.

### 피해야 할 과도한 문장

- “Branch CE는 FL에서 항상 나쁘다.”
- “단순 dataset에서는 branch CE가 항상 성능을 높인다.”
- “KD가 client teacher를 동일하게 만든다.”
- “KD의 이득은 dark knowledge class relation 때문이다.”
- “Teacher confidence가 높을수록 KD는 항상 신뢰할 수 있다.”
- “$\alpha=1$이 좋으므로 $\lambda$도 클수록 좋다.”
