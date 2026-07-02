import os
import shutil

LOG_DIR = 'logs'
partitions = ['iid', 'beta_0.3', 'beta_0.5', 'noniid_grouping']
algorithms = ['fedavg', 'fedprox', 'moon', 'fedrcl', 'feddecorr', 'fedlc', 'fedrs']

def reorganize_logs():
    if not os.path.exists(LOG_DIR):
        print("logs 폴더가 존재하지 않습니다.")
        return

    moved_count = 0
    for p in partitions:
        p_path = os.path.join(LOG_DIR, p)
        if not os.path.exists(p_path):
            continue
            
        for file in os.listdir(p_path):
            file_path = os.path.join(p_path, file)
            
            # 폴더는 건너뛰고 파일인 경우에만 처리
            if not os.path.isfile(file_path):
                continue
                
            for alg in algorithms:
                prefix = alg + '_'
                # 파일명이 '알고리즘명_' 으로 시작하는지 확인
                if file.startswith(prefix):
                    # 알고리즘명 이후의 문자열(비교알고리즘명 + 확장자)을 추출
                    # 예: feddecorr_baseline_err.log -> baseline_err.log
                    method_and_ext = file[len(prefix):]
                    
                    new_alg_dir = os.path.join(p_path, alg)
                    os.makedirs(new_alg_dir, exist_ok=True)
                    
                    new_file_path = os.path.join(new_alg_dir, method_and_ext)
                    
                    # 파일 이동
                    shutil.move(file_path, new_file_path)
                    print(f"이동: {p}/{file} -> {p}/{alg}/{method_and_ext}")
                    moved_count += 1
                    break
                    
    print(f"\n정리 완료! 총 {moved_count}개의 파일이 이동되었습니다.")

if __name__ == '__main__':
    reorganize_logs()