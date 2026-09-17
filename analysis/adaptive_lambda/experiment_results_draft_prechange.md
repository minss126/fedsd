# Experiment Results Draft

> 임시 초안: 아래 수치는 최종 JS-client/no-feature 재실험 이전의 soft-b 기반 결과를 포함한다. 최종 논문 표에서는 현재 재실험 결과로 교체해야 한다. Accuracy는 별도 표기가 없는 한 마지막 30 communication rounds의 평균(%)이다.

## 0. Experimental Setup

### 0.1 Datasets and training horizon

| Dataset | Classes | Input resolution | Communication rounds | Adaptive warm-up | Initial learning rate |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 10 | $32\times32$ | 500 | 250 | 0.1 |
| CIFAR-100 | 100 | $32\times32$ | 500 | 250 | 0.1 |
| TinyImageNet | 200 | $64\times64$ | 100 | 50 | 0.01 |
| ImageNet100-64 | 100 | $64\times64$ | 100 | 50 | 0.01 |

### 0.2 Federated learning protocol

| Item | Setting |
|---|---|
| Number of clients | 100 |
| Default client participation | 0.1 (10 clients per round) |
| Default local epochs | 5 |
| Data partition | IID and Dirichlet non-IID with $\beta\in\{0.3,0.1\}$ |
| Minimum client samples | 64 (`min_require_size=64`) |
| Aggregation | FedAvg weighted by the number of local samples |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight decay | 0.001 |
| Learning-rate schedule | Round-wise exponential decay, $\eta_t=\eta_0\cdot0.998^t$ |
| Train batch size | 64 |
| Test batch size | 512 |
| Default seed | 0 |
| Evaluation | Global test accuracy averaged over the last 30 rounds |

### 0.3 Backbone and auxiliary branches

| Item | Setting |
|---|---|
| Default backbone | ResNet18 |
| Additional backbone | MobileNetV2 |
| ResNet18 branch locations | B1/B2/B3 after `layer1`/`layer2`/`layer3` |
| Final classifier | After `layer4` |
| Branch adapter | Convolutional bottleneck, global average pooling, linear classifier |
| Main-path objective | Hard-label cross-entropy |
| Branch objective | KD only; branch hard-label CE is not used |
| Teacher | The final classifier of the same local model |
| Feature imitation | Not used in the final protocol ($\beta_{\mathrm{feature}}=0$) |
| Branch loss reduction | Sum over B1/B2/B3 |
| KD temperature | $T_{\mathrm{KD}}=1$ |
| Proxy temperature | $T_{\mathrm{proxy}}=1$ |

The local objective of the BYOT models is

$$
\mathcal L_k
=
\mathcal L_{\mathrm{CE}}^{\mathrm{final}}
+
\sum_{b=1}^{B}\lambda_{k,t}\,
\mathcal L_{\mathrm{KD}}^{(b)},
$$

where $B=3$ for the evaluated ResNet18-BYOT and MobileNetV2-BYOT models. Auxiliary branches are used only during local training; evaluation uses the final classifier.

### 0.4 Compared methods

| Method | Model/training configuration | KD coefficient |
|---|---|---|
| Plain | Standard backbone trained with final CE only | $0$ |
| Fixed $\lambda=0.3$ | BYOT backbone, final CE + branch KD | Constant $0.3$ from the first round; no warm-up |
| Adaptive | BYOT backbone, final CE + branch KD | Client- and round-dependent $\lambda_{k,t}$ |

For the final adaptive protocol,

$$
\lambda_{k,t}
=
\lambda_t\,r_{k,t}\,s_{k,t}\,g_{k,t},
$$

where:

| Component | Definition/configuration |
|---|---|
| Round schedule $\lambda_t$ | Linear increase from 0 to $\lambda_{\max}=1$ during the first 50% of rounds |
| Teacher reliability $r_{k,t}$ | Mean true-label probability of the local final classifier; power 1 |
| Bias correction $s_{k,t}$ | Client prediction-entropy soft relaxation; power 2, threshold $\tau=0.85$, temperature 0.05 |
| JS-client gate $g_{k,t}$ | Mean normalized teacher–branch JS divergence over active branches; gain 1, minimum gate 0 |
| Granularity | One client-wise coefficient is shared by all active branches |

### 0.5 FL mechanisms

| Mechanism | Configuration |
|---|---|
| FedAvg | Default aggregation mechanism |
| FedProx | Proximal coefficient $\mu=0.01$ |
| MOON | Contrastive coefficient $\mu=0.01$, MOON temperature 0.5 |

### 0.6 One-factor-at-a-time extensions

| Axis | Values | Other axes fixed to |
|---|---|---|
| Dataset | CIFAR-10, CIFAR-100, TinyImageNet, ImageNet100-64 | ResNet18, FedAvg, $E=5$, participation 0.1 |
| Model | ResNet18, MobileNetV2 | CIFAR-100, FedAvg, $E=5$, participation 0.1 |
| FL mechanism | FedAvg, FedProx, MOON | CIFAR-100, ResNet18, $E=5$, participation 0.1 |
| Local epochs | 1, 5, 10 | CIFAR-100, ResNet18, FedAvg, participation 0.1 |
| Client participation | 0.05, 0.1, 0.2 | CIFAR-100, ResNet18, FedAvg, $E=5$ |

## 1. 확장 실험

### 1.1 Dataset

| Dataset | Partition | Plain | Fixed $\lambda=0.3$ | Adaptive |
|---|---|---:|---:|---:|
| CIFAR-100 | IID | 70.336 | **72.679** | 72.210 |
| CIFAR-100 | $\beta=0.3$ | 65.153 | **66.121** | 65.732 |
| CIFAR-100 | $\beta=0.1$ | **55.359** | 53.433 | 54.779 |
| TinyImageNet | IID | 41.298 | 45.582 | **46.151** |
| TinyImageNet | $\beta=0.3$ | 38.268 | 41.552 | **42.414** |
| TinyImageNet | $\beta=0.1$ | 33.850 | 35.301 | **36.400** |
| ImageNet100-64 | IID | 56.794 | 61.620 | **61.687** |
| ImageNet100-64 | $\beta=0.3$ | 52.010 | 54.635 | **55.611** |
| ImageNet100-64 | $\beta=0.1$ | 44.014 | 44.383 | **45.872** |

### 1.2 Model — CIFAR-100

| Model | Partition | Plain | Fixed $\lambda=0.3$ | Adaptive |
|---|---|---:|---:|---:|
| ResNet18 | IID | 70.336 | **72.679** | 72.210 |
| ResNet18 | $\beta=0.3$ | 65.153 | **66.121** | 65.732 |
| ResNet18 | $\beta=0.1$ | **55.359** | 53.433 | 54.779 |
| MobileNetV2 | IID | 60.699 | 61.727 | **62.165** |
| MobileNetV2 | $\beta=0.3$ | 51.849 | 50.060 | **52.516** |
| MobileNetV2 | $\beta=0.1$ | 35.654 | 31.038 | **35.847** |

### 1.3 FL mechanism — CIFAR-100

| FL mechanism | Partition | Plain | Fixed $\lambda=0.3$ | Adaptive |
|---|---|---:|---:|---:|
| FedAvg | IID | 70.336 | **72.679** | 72.210 |
| FedAvg | $\beta=0.3$ | 65.153 | **66.121** | 65.732 |
| FedAvg | $\beta=0.1$ | **55.359** | 53.433 | 54.779 |
| FedProx | IID | 70.451 | **72.603** | 72.340 |
| FedProx | $\beta=0.3$ | 65.875 | 66.890 | **66.929** |
| FedProx | $\beta=0.1$ | **56.685** | 55.028 | 56.245 |
| MOON | IID | 70.306 | **72.624** | 72.117 |
| MOON | $\beta=0.3$ | 65.077 | **66.009** | 65.638 |
| MOON | $\beta=0.1$ | **55.822** | 53.572 | 54.247 |

### 1.4 Local epoch — CIFAR-100

| Local epoch | Partition | Plain | Fixed $\lambda=0.3$ | Adaptive |
|---:|---|---:|---:|---:|
| 1 | IID | 65.801 | **66.880** | 66.518 |
| 1 | $\beta=0.3$ | **59.747** | 58.385 | 59.123 |
| 1 | $\beta=0.1$ | **51.512** | 45.621 | 48.947 |
| 5 | IID | 70.336 | **72.679** | 72.210 |
| 5 | $\beta=0.3$ | 65.153 | **66.121** | 65.732 |
| 5 | $\beta=0.1$ | **55.359** | 53.433 | 54.779 |
| 10 | IID | 67.650 | **69.362** | 68.950 |
| 10 | $\beta=0.3$ | 60.783 | 61.083 | **61.243** |
| 10 | $\beta=0.1$ | **49.287** | 46.511 | 48.880 |

### 1.5 Client participation — CIFAR-100

| Participation | Partition | Plain | Fixed $\lambda=0.3$ | Adaptive |
|---:|---|---:|---:|---:|
| 0.05 | IID | 67.114 | **69.248** | 68.664 |
| 0.05 | $\beta=0.3$ | 58.425 | **59.091** | 58.845 |
| 0.05 | $\beta=0.1$ | **45.097** | 43.095 | 43.488 |
| 0.10 | IID | 70.336 | **72.679** | 72.210 |
| 0.10 | $\beta=0.3$ | 65.153 | **66.121** | 65.732 |
| 0.10 | $\beta=0.1$ | **55.359** | 53.433 | 54.779 |
| 0.20 | IID | 71.796 | **74.200** | 74.110 |
| 0.20 | $\beta=0.3$ | 69.041 | 69.689 | **70.027** |
| 0.20 | $\beta=0.1$ | 61.975 | 60.106 | **62.139** |

## 2. Effective $\lambda$

### 2.1 Round-wise 변화

| Dataset | Partition | Round 1 | Warm-up 종료 | Final round | Last-30 mean |
|---|---|---:|---:|---:|---:|
| CIFAR-100 | IID | 0.0000 | 0.6743 | 0.7987 | 0.7915 |
| CIFAR-100 | $\beta=0.3$ | 0.0000 | 0.5092 | 0.5846 | 0.5699 |
| CIFAR-100 | $\beta=0.1$ | 0.0000 | 0.2723 | 0.3140 | 0.3140 |
| TinyImageNet | IID | 0.0001 | 0.3177 | 0.4373 | 0.3996 |
| TinyImageNet | $\beta=0.3$ | 0.0001 | 0.2465 | 0.3222 | 0.3143 |
| TinyImageNet | $\beta=0.1$ | 0.0001 | 0.1575 | 0.2482 | 0.2165 |
| ImageNet100-64 | IID | 0.0002 | 0.5264 | 0.6593 | 0.6321 |
| ImageNet100-64 | $\beta=0.3$ | 0.0002 | 0.4124 | 0.5051 | 0.4869 |
| ImageNet100-64 | $\beta=0.1$ | 0.0002 | 0.2479 | 0.3337 | 0.3148 |

### 2.2 CIFAR-100 client-wise 분포

| Partition | Mean | Std. | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| IID | 0.785 | 0.033 | 0.694 | 0.784 | 0.864 |
| $\beta=0.1$ | 0.317 | 0.045 | 0.194 | 0.317 | 0.412 |

## 3. Ablation

### 기존 soft-b 결과를 이용한 임시 표

| Method | IID | $\beta=0.3$ | $\beta=0.1$ |
|---|---:|---:|---:|
| Full adaptive | **72.210** | 65.732 | **54.779** |
| w/o warm-up | 71.848 | 65.587 | 53.592 |
| w/o reliability | 71.710 | 65.415 | 52.403 |
| w/o bias correction | 72.087 | **65.894** | 54.709 |
| w/o JS-client | — | — | — |
