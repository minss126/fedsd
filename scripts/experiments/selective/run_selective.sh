#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "🚀 FedLC 및 FedRS 융합 실험 실행 시작..."

JOB_CMD=()

get_args() {
    local alg=$1
    local method=$2
    local dataset=$3  
    local total_ep=$4 

    # 기존 실험과 동일한 베이스 파라미터 적용 (Epoch 10 기준)
    local base="--lr 0.1 --round 500 --byot_alpha 0.3 --byot_beta 0.1 --mu 0.01 --dataset $dataset --epochs $total_ep"

    if [ "$method" == "baseline" ]; then
        if [ "$alg" == "fedlc" ]; then 
            echo "$base --alg fedlc --model resnet18 --calibration_temp 1.0"
        elif [ "$alg" == "fedrs" ]; then 
            echo "$base --alg fedrs --model resnet18 --fedrs_alpha 0.5"
        fi
    elif [ "$method" == "selective" ]; then
        if [ "$alg" == "fedlc" ]; then 
            echo "$base --alg fedbyot_lc_selective --model resnet18_byot --calibration_temp 1.0"
        elif [ "$alg" == "fedrs" ]; then 
            echo "$base --alg fedbyot_rs_selective --model resnet18_byot --fedrs_alpha 0.5"
        fi
    elif [ "$method" == "warmup" ]; then
        if [ "$alg" == "fedlc" ]; then 
            echo "$base --alg fedbyot_lc_greedy --model resnet18_byot --warmup_epochs 2 --calibration_temp 1.0"
        elif [ "$alg" == "fedrs" ]; then 
            echo "$base --alg fedbyot_rs_greedy --model resnet18_byot --warmup_epochs 2 --fedrs_alpha 0.5"
        fi
    fi
}

# 파티션 폴더 매핑
declare -A partitions=( 
    ["iid"]="--partition iid" 
    ["beta_0.3"]="--partition noniid --beta 0.3" 
    ["beta_0.5"]="--partition noniid --beta 0.5" 
    ["noniid_grouping"]="--partition noniid_grouping --partition_groups 8" 
)

# 배열 순서 고정 출력을 위해 직접 리스트 명시
for p_name in "iid" "beta_0.3" "beta_0.5" "noniid_grouping"; do
    for alg in fedlc fedrs; do
        for method in baseline selective warmup; do
            args=$(get_args $alg $method "cifar100" 10)
            JOB_CMD+=("python main.py $args ${partitions[$p_name]} --log_file_name ${p_name}/${alg}_${method}")
        done
    done
done

# =======================================================
# [수정됨] 3대의 GPU (0, 1, 2)에만 분배하도록 변경
# =======================================================
total_jobs=${#JOB_CMD[@]}
echo "총 $total_jobs 개의 작업을 3대의 GPU (0, 1, 2)에 분배합니다."

i=0
for cmd in "${JOB_CMD[@]}"; do
    GPU_ID=$(( i % 3 )) # 0, 1, 2 번 GPU만 할당
    log_file=$(echo $cmd | grep -oP '(?<=--log_file_name )[^ ]+')
    
    mkdir -p "logs/$(dirname $log_file)"
    
    echo "[Job $((i+1))/$total_jobs] GPU $GPU_ID -> $log_file"
    eval "$cmd --device cuda:$GPU_ID > logs/${log_file}_err.log 2>&1" &
    
    i=$((i+1))
    # 3개 작업이 실행되면 대기
    if (( i % 3 == 0 )); then wait; fi
done

wait
echo "✅ FedLC & FedRS 모든 실험 완료!"