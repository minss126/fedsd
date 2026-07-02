import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 저장 폴더 설정
SAVE_DIR = 'results_analysis/plots_combined_methods'
os.makedirs(SAVE_DIR, exist_ok=True)

def parse_log_file(filepath):
    """단일 로그 파일에서 Acc, Time, Efficiency 평균값을 추출"""
    if not os.path.exists(filepath):
        return None 
        
    acc_list, time_list, eff_list = [], [], []
    
    # 정규표현식 패턴
    result_pattern = re.compile(r"Round\s+(\d+)\s+result:\s+Acc=([0-9.]+)")
    time_eff_pattern = re.compile(r"1 Round train time:\s*([0-9.]+)\s*\|\s*Efficiency:\s*([0-9.]+)")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                te_match = time_eff_pattern.search(line)
                if te_match:
                    time_list.append(float(te_match.group(1)))
                    eff_list.append(float(te_match.group(2)) * 100)
                    
                result_match = result_pattern.search(line)
                if result_match:
                    acc_list.append(float(result_match.group(2)))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
        
    if not acc_list: 
        return None
        
    # 마지막 10라운드 평균 Accuracy, 전체 평균 Time/Efficiency 계산
    final_10_avg_acc = sum(acc_list[-10:]) / len(acc_list[-10:])
    avg_time_per_round = sum(time_list) / len(time_list) if time_list else 0.0
    avg_eff = sum(eff_list) / len(eff_list) if eff_list else 100.0
    
    return {'Acc': final_10_avg_acc, 'Time_Per_Round': avg_time_per_round, 'Efficiency': avg_eff}

def analyze_and_plot():
    data = []
    
    # 환경 매핑: (그래프 표시명, logs_tuning 폴더명, logs 폴더명)
    partition_map = [
        ('IID', 'iid', 'iid'),
        ('Beta 0.3', 'beta_0.3', 'beta_0.3'),
        ('Beta 0.5', 'beta_0.5', 'beta_0.5'),
        ('Grouping', 'noniid_grouping', 'noniid_grouping')
    ]
    
    algs = ['fedavg', 'fedprox', 'moon', 'fedrcl', 'feddecorr']
    methods = ['baseline', 'fedsd', 'selective', 'warmup']

    # 데이터 추출 루프
    for disp_partition, tuned_folder, orig_folder in partition_map:
        for alg in algs:
            for method in methods:
                # logs_tuning 폴더와 logs 폴더의 각각 다른 폴더명 반영
                tuned_log_path = os.path.join('logs_tuning', tuned_folder, alg, f"{method}.log")
                orig_log_path = os.path.join('logs', orig_folder, alg, f"{method}.log")
                
                parsed = None
                
                # 1순위: logs_tuning 폴더의 결과
                if os.path.exists(tuned_log_path):
                    parsed = parse_log_file(tuned_log_path)
                    if parsed:
                        print(f"Loaded (NEW): {tuned_log_path}")
                # 2순위: logs 폴더의 기존 결과
                elif os.path.exists(orig_log_path):
                    parsed = parse_log_file(orig_log_path)
                    if parsed:
                        print(f"Loaded (OLD): {orig_log_path}")
                
                if parsed:
                    data.append({
                        'Partition': disp_partition,
                        'Algorithm': alg.upper(),
                        'Method': method,
                        'Accuracy (%)': parsed['Acc'],
                        'Time / Round (s)': parsed['Time_Per_Round'],
                        'Data Efficiency (%)': parsed['Efficiency']
                    })

    df = pd.DataFrame(data)
    if df.empty:
        print("데이터를 찾을 수 없습니다. 경로 구조를 확인해 주세요.")
        return

    # 지표별 (파일명 접두사, y축 범위)
    metrics = {
        'Accuracy (%)': ('Acc', (40, 80)),
        'Time / Round (s)': ('Time', None),
        'Data Efficiency (%)': ('Efficiency', (50, 105))
    }
    
    sns.set_theme(style="whitegrid")
    
    # 그래프 생성 루프
    for metric_label, (file_prefix, y_limit) in metrics.items():
        for partition in df['Partition'].unique():
            plt.figure(figsize=(9, 5))
            subset = df[df['Partition'] == partition]
            
            ax = sns.barplot(
                data=subset, x='Algorithm', y=metric_label, hue='Method',
                palette='muted', hue_order=methods
            )
            
            if y_limit:
                plt.ylim(y_limit)
            
            plt.title(f'{metric_label} - {partition}', fontsize=12, fontweight='bold')
            plt.xlabel('', fontsize=0)
            plt.ylabel(metric_label, fontsize=10)
            plt.xticks(rotation=15, fontsize=9)
            plt.yticks(fontsize=9)
            
            # 범례를 막대그래프 외부(우측)로 분리
            plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            filename = f"{file_prefix}_{partition.replace(' ', '_')}.png"
            filepath = os.path.join(SAVE_DIR, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            
            print(f"✅ 그래프 저장 완료: {filepath}")

if __name__ == '__main__':
    analyze_and_plot()