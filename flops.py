import torch
from thop import profile
# 모델 파일 경로가 다르다면 아래 import 경로를 수정해주세요.
from models.resnet_byot import multi_resnet18_kd  

# ==========================================
# [설정] CIFAR-10 기준
# ==========================================
NUM_CLASSES = 10       # 클래스 개수
INPUT_SIZE = (32, 32)  # 이미지 크기 (가로, 세로)
# ==========================================

# 1. 모델 준비
# CIFAR-10에 맞는 클래스 개수로 초기화
model = multi_resnet18_kd(num_classes=NUM_CLASSES) 
model.eval()

# 2. 가짜 데이터 1장 생성 (배치 크기=1, 채널=3, 32x32)
input_data = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])

print("----------------------------------------------------------------")
print(f" 🖥️  FLOPs 및 파라미터 측정 중... (Dataset: CIFAR-10)")
print("----------------------------------------------------------------")

# 3. 측정 (thop 라이브러리가 계산)
macs, params = profile(model, inputs=(input_data, ), verbose=False)

# 4. 결과 출력
flops = macs * 2  # FLOPs = MACs * 2
print(f"✅ 모델: ResNet18 (Proposed SD)")
print(f"📉 파라미터 수 (Params): {params / 1e6:.2f} M (백만)")
print(f"⚙️ 연산량 (FLOPs): {flops / 1e9:.4f} G (십억)")
print("----------------------------------------------------------------")