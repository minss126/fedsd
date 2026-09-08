# Federated CE/KD Gradient-Route Probe

하나의 고정된 global checkpoint에서 final CE와 auxiliary CE/KD가 shared
prefix에 주는 gradient의 방향과 크기를 비교하는 진단 코드다. optimizer
step은 수행하지 않으며 test label은 분석 reference에만 사용한다.

## 측정 시점과 데이터

각 지정 round의 aggregation이 완료된 동일 파라미터 `theta_t`에서 측정한다.

```text
selected client train subsets -> LCE, Aux-LCE, LKD
official test reference       -> GCE, Aux-GCE, GKD
```

Local route는 client별 전체 subset의 sample-mean gradient를 구한 뒤 실제로
전달된 FedAvg weight로 합친다. Global route는 official test set 전체의
sample-mean gradient다. 모델 파라미터와 BN statistics는 측정 중 바뀌지 않는다.

## Objective

Final logit을 `f(x)`, branch `i`의 logit을 `b_i(x)`라 하면 다음 route를
독립적으로 계산한다.

```text
Main CE_i = CE(f(x), y)
Aux CE_i  = CE(b_i(x), y)
Aux KD_i  = T^2 KL(softmax(f(x)/T).detach() || softmax(b_i(x)/T))
```

기본 temperature는 teacher와 student 모두 `T=1.0`이다. B1/B2/B3 loss를
각각 미분하므로 branch 효과가 섞이지 않는다. `All`은 별도 backward 없이
세 독립 gradient의 합으로 복원한다.

```text
g_Aux-All = g_Aux-B1 + g_Aux-B2 + g_Aux-B3
```

`branch_reduction=mean`이면 위 합을 3으로 나눈다.

## Primary analyses

모든 비교는 B1/B2/B3/All에 대해 계산된다. B1/B2/B3는 각 exit까지의
cumulative shared prefix이며 All은 B1+B2+B3가 도달할 수 있는 전체
`stem+layer1+layer2+layer3`다.

| 분석 | CE | KD |
|---|---|---|
| Global objective alignment | Aux-GCE vs Main-GCE | GKD vs Main-GCE |
| Local objective alignment | Aux-LCE vs Main-LCE | LKD vs Main-LCE |
| Local-to-global consistency | Aux-LCE vs Aux-GCE | LKD vs GKD |

## 저장 지표

각 gradient pair에 대해 다음 scalar를 모두 저장한다.

- cosine similarity와 angle
- 두 gradient의 absolute L2 norm
- norm ratio
- dot product
- signed projection
- reference-normalized projection
- negative-cosine 여부

또한 다음 세 수준을 모두 기록한다.

- FedAvg gradient를 먼저 합친 뒤 계산한 aggregate comparison
- 각 client의 comparison
- client weighted/unweighted mean, standard deviation, normal-approximation
  95% CI, min/max, negative-cosine rate

모든 local/global route 사이의 full pairwise-stat matrix와 incremental shared
stage별 absolute norm도 저장한다. 실행 시간 증가가 거의 없는 scalar만
JSON에 저장하고 raw gradient tensor는 저장하지 않는다.

## DXFL 실행

저장소 루트에서 다음을 실행한다.

```bash
scripts/experiments/analysis/run_gradient_route_probe_4gpu.sh
```

기본 조건은 경향성 확인을 위한 CIFAR-10/CIFAR-100, IID/Dirichlet
beta=0.1, ResNet-18, 500 rounds, CE+feature/KD+feature, seed 0이다.
Training과 diagnostic KD temperature를 모두 `T=1.0`으로 명시한다.
50 round마다 전체 참여 client의 full local subset과 전체 official test set을
사용한다.

주요 override 예시는 다음과 같다.

```bash
SEEDS_OVERRIDE="0 1 2" PARTITIONS_OVERRIDE="iid beta_0.5 beta_0.1" \
PROBE_ROUNDS_OVERRIDE="100,250,500" \
scripts/experiments/analysis/run_gradient_route_probe_4gpu.sh
```

`PROBE_CLIENTS`, `LOCAL_MAX_BATCHES`, `GLOBAL_MAX_BATCHES`는 빠른 smoke run을
위한 cap이며 기본값 0은 전부 사용한다. 결과는 기본적으로 아래에 생성된다.

```text
logs/analysis/logs_gradient_route_probe_t1_r500/
  <dataset>_resnet18/beta_0.5/fedavg/seed<seed>/
    <variant>_gradient_routes/round_0050.json
    ...
    <variant>_gradient_routes/round_0500.json
```

## 해석상 주의

- cosine이 높다는 사실만으로 해당 objective가 accuracy를 높인다고 결론낼 수
  없으므로 norm ratio와 최종 성능을 같이 봐야 한다.
- client-level 평균과 FedAvg aggregate-gradient 결과는 서로 다른 통계다.
- local reference는 실제 train transform, global reference는 clean test
  transform을 쓴다. 따라서 local-to-global 값은 label skew뿐 아니라 실제
  FL에서 존재하는 train/test transform 차이까지 포함한 operational mismatch다.
- official test label은 학습, gate 선택, hyperparameter selection에 사용하면
  안 된다. 본 실험에서는 post-hoc diagnostic reference로만 사용한다.
