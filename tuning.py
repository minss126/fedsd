import os
import pickle
import numpy as np
import pandas as pd

# ==============================================================================
# [설정] 비교할 실험 경로 및 파일명
# ==============================================================================
EXPERIMENTS = [
    # Reference (기존 실험들 - seed0 폴더 있음)
    ("FedBYOT (Base)",      "./logs/noniid/beta0.5",     "fedbyot"),
    ("Ours (Original)",     "./logs/noniid/beta0.5",     "fedbyot_selective"),
    
    # Tuning Candidates (새로운 튜닝 실험들 - seed0 폴더 없을 수 있음)
    ("Tuning: Thresh 0.35", "./logs/tuning/th0.35",      "ours_th0.35"),
    ("Tuning: Thresh 0.30", "./logs/tuning/th0.30",      "ours_th0.30"),
    ("Tuning: Temp 3.0",    "./logs/tuning/temp3.0",     "ours_temp3.0"),
    ("Tuning: Beta 0.1",    "./logs/tuning/beta0.1",     "ours_beta0.1"),
]

SEED = 0

def load_data(base_dir, file_prefix):
    # Case 1: 기존 구조 (base_dir/seed0/filename.pkl)
    path_v1 = os.path.join(base_dir, f"seed{SEED}", f"{file_prefix}.pkl")
    
    # Case 2: 튜닝 구조 (base_dir/filename.pkl) - seed 폴더 없이 바로 저장된 경우
    path_v2 = os.path.join(base_dir, f"{file_prefix}.pkl")
    
    final_path = None
    
    if os.path.exists(path_v1):
        final_path = path_v1
    elif os.path.exists(path_v2):
        final_path = path_v2
    else:
        # 파일이 아직 생성되지 않음 (학습 진행 중)
        return None
    
    try:
        with open(final_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        return None

def analyze():
    results = []
    base_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"   [Hyperparameter Tuning Championship] (Seed {SEED})")
    print(f"{'='*60}")

    # ... (나머지 로직은 그대로 유지) ...
    # 아래는 복붙 편의를 위해 전체 코드 흐름 유지
    
    for label, path, fname in EXPERIMENTS:
        data = load_data(path, fname)
        
        if data is None:
            results.append({"Method": label, "Acc": "-", "Cost": "-", "Gap": "-"})
            continue
            
        acc_list = data.get('acc', [])
        if not acc_list: acc_list = data.get('history', {}).get('acc', [])
        
        if len(acc_list) > 0:
            final_acc = np.mean(acc_list[-10:])
        else:
            final_acc = 0.0

        if "Base" in label:
            base_acc = final_acc

        feat_list = data.get('feat_ratio', [])
        if not feat_list: feat_list = data.get('efficiency', [])
        
        if feat_list and len(feat_list) > 0:
            if isinstance(feat_list[0], list):
                feat_vals = [x[0] if isinstance(x, list) else x for x in feat_list]
                final_cost = np.mean(feat_vals) * 100
            else:
                final_cost = np.mean(feat_list) * 100
        else:
            final_cost = 100.0

        results.append({
            "Method": label,
            "Acc": final_acc,
            "Cost": final_cost,
            "Gap": 0.0
        })

    df_rows = []
    for res in results:
        method = res["Method"]
        if res["Acc"] == "-":
            df_rows.append([method, "Running...", "-", "-"]) # Not Found -> Running으로 변경
            continue
            
        acc = res["Acc"]
        cost = res["Cost"]
        gap = acc - base_acc
        
        if "Base" in method: gap_str = "Ref"
        else: gap_str = f"{gap:+.2f}%"
            
        acc_str = f"{acc:.2f}%"
        cost_str = f"{cost:.1f}%"
        df_rows.append([method, acc_str, cost_str, gap_str])

    df = pd.DataFrame(df_rows, columns=["Method", "Accuracy", "Cost (Feat)", "Gap vs Base"])
    print(df.to_string(index=False))
    print(f"{'='*60}")
    
    # 승자 추천 로직
    print("\n[📢 AI Analysis & Recommendation]")
    best_cand = None
    max_acc = -1.0
    
    for res in results:
        if res["Acc"] == "-" or "Base" in res["Method"]: continue
        if res["Acc"] > base_acc:
            if res["Acc"] > max_acc:
                max_acc = res["Acc"]
                best_cand = res
    
    if best_cand:
        print(f"🏆 WINNER: [{best_cand['Method']}]")
        print(f"   - 정확도: {base_acc:.2f}% -> {best_cand['Acc']:.2f}% (+{best_cand['Acc'] - base_acc:.2f}%)")
        print(f"   - 비용: {best_cand['Cost']:.1f}%")
        if "Temp" in best_cand['Method'] or "Beta" in best_cand['Method']:
            print("⚠️ (참고) Temp/Beta 변경 모델이 1등입니다. 시간 되면 Base도 같은 설정으로 확인해보세요.")
        else:
            print("✅ Threshold 튜닝 성공! 베이스라인 재실험 없이 바로 채택하세요.")
    else:
        print("Running... (아직 실험이 진행 중이거나, 파일 생성 전입니다.)")

if __name__ == "__main__":
    analyze()