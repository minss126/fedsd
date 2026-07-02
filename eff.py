import matplotlib.pyplot as plt
import re
import os

# ==============================================================================
# [설정] 로그 파일 경로 매핑
# ※ 만약 파일명이 다르다면 여기서 수정해주세요!
# ==============================================================================
LOG_MAP = {
    "CIFAR-10": {
        "Base": "logs/log_cifar10_base.txt",
        "Fixed SD (Old)": "logs/log_cifar10_sd.txt",       # 아까 실패했던 버전
        "Adaptive SD (New)": "logs/log_cifar10_adaptive.txt" # 지금 돌리는 버전
    },
    "Tiny-ImageNet": {
        "Base": "logs/log_tinyimagenet_base.txt",
        "Fixed SD (Old)": "logs/log_tinyimagenet_sd.txt",
        "Adaptive SD (New)": "logs/log_tiny_adaptive.txt"
    },
    "EMNIST": {
        "Base": "logs/log_emnist_base.txt",
        "Fixed SD (Old)": "logs/log_emnist_sd.txt",
        "Adaptive SD (New)": "logs/log_emnist_adaptive.txt"
    }
}

COLORS = {"Base": "black", "Fixed SD (Old)": "red", "Adaptive SD (New)": "blue"}
STYLES = {"Base": "--", "Fixed SD (Old)": ":", "Adaptive SD (New)": "-"}

def parse_log(filepath):
    """로그 파일에서 Accuracy 리스트 추출"""
    accuracies = []
    if not os.path.exists(filepath):
        return None  # 파일 없으면 무시
    
    with open(filepath, 'r') as f:
        for line in f:
            # "Round 10 result: Acc=12.34" 패턴 찾기
            match = re.search(r"Round \d+ result: Acc=([\d\.]+)", line)
            if match:
                accuracies.append(float(match.group(1)))
    return accuracies

def plot_all():
    print(f"{'Dataset':<15} | {'Method':<20} | {'Max Acc':<10} | {'Last Acc':<10}")
    print("=" * 70)

    # 데이터셋별로 그래프 그리기
    for dataset, methods in LOG_MAP.items():
        plt.figure(figsize=(8, 5))
        has_data = False
        
        for label, filepath in methods.items():
            accs = parse_log(filepath)
            
            if accs:
                has_data = True
                rounds = range(len(accs))
                current = accs[-1]
                best = max(accs)
                
                # 그래프 Plot
                plt.plot(rounds, accs, label=f"{label} (Max: {best:.2f}%)", 
                         color=COLORS.get(label, 'gray'), 
                         linestyle=STYLES.get(label, '-'), linewidth=2)
                
                print(f"{dataset:<15} | {label:<20} | {best:>9.2f}% | {current:>9.2f}%")
            else:
                # 파일이 없을 경우 (아직 안 돌렸거나 삭제됨)
                pass

        if has_data:
            plt.title(f"{dataset} Performance Comparison")
            plt.xlabel("Round")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 이미지 저장
            filename = f"comparison_{dataset.lower().replace('-', '')}_final.png"
            plt.savefig(filename)
            print(f"   └─ 📈 Graph saved: {filename}")
            print("-" * 70)
        else:
            plt.close() # 데이터 없으면 캔버스 닫기

if __name__ == "__main__":
    plot_all()