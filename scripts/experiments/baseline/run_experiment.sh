#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# 사용할 GPU 리스트 (기존처럼 4개 동시 할당으로 속도 최적화)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

# ---------------------------------------------------------
# 🚨 글로벌 파라미터 세팅 (업로드된 원본 기준 동기화)
# ---------------------------------------------------------
MU_VAL="0.01"        
TEMP_VAL="0.5"       
ALPHA_VAL="0.05"     
BETA_VAL="0.01"      

# 공통 실행 함수
run_job() {
    local env_name=$1
    local env_flags=$2
    local method_name=$3
    local method_flags=$4
    local algo_name=$5
    local algo_flags=$6

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}

    # 폴더 구조: logs_tuning/환경/알고리즘
    local log_dir="logs_tuning/${env_name}/${algo_name}"
    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] 시작: 환경=${env_name} | 알고리즘=${algo_name} | 기법=${method_name}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed 0 \
        --device "cuda:${gpu_id}" \
        --logdir "logs_tuning" \
        --log_file_name "${env_name}/${algo_name}/${method_name}" \
        $env_flags $method_flags $algo_flags \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "✅ 현재 4개 배치 완료. 다음 배치 시작..."
    fi
}

# ---------------------------------------------------------
# 1. 환경(Environment) 설정 (폴더명 및 인자 동기화)
# ---------------------------------------------------------
declare -A ENV_MAP
ENV_MAP["iid"]="--partition iid"
ENV_MAP["beta_0.5"]="--partition noniid --beta 0.5"
ENV_MAP["beta_0.3"]="--partition noniid --beta 0.3"
ENV_MAP["noniid_grouping"]="--partition noniid_grouping --partition_groups 8"

# ---------------------------------------------------------
# 2. 알고리즘(Algorithm) 설정 (원본 기준 파라미터 정리)
# ---------------------------------------------------------
declare -A ALGO_MAP
ALGO_MAP["fedavg"]=""
ALGO_MAP["fedprox"]="--use_fedprox --mu ${MU_VAL}"
ALGO_MAP["moon"]="--use_moon --mu ${MU_VAL} --temperature ${TEMP_VAL}"
ALGO_MAP["fedrcl"]="--use_fedrcl"       
ALGO_MAP["feddecorr"]="--feddecorr"     

# ---------------------------------------------------------
# 3. [작업 1] Baseline 전체 재실행
# ---------------------------------------------------------
echo "========== [작업 1] Baseline 재실행 =========="

for env in "iid" "beta_0.3" "beta_0.5" "noniid_grouping"; do
    for algo in "fedavg" "fedprox" "moon" "fedrcl" "feddecorr"; do
        
        BASELINE_FLAGS="--model resnet18 --alg $algo"
        
        run_job "$env" "${ENV_MAP[$env]}" "baseline" "$BASELINE_FLAGS" "$algo" "${ALGO_MAP[$algo]}"
    done
done

wait
echo "✅ Baseline 전체 실행 완료"

# ---------------------------------------------------------
# 4. [작업 2] BYOT 증류 기법 전체 재실행
# ---------------------------------------------------------
echo "========== [작업 2] 증류 기법 재실행 =========="
declare -A METHOD_MAP
METHOD_MAP["fedsd"]="--model resnet18_byot --alg fedbyot --kd_conf_threshold 0.0 --byot_alpha ${ALPHA_VAL} --byot_beta ${BETA_VAL}"
METHOD_MAP["selective"]="--model resnet18_byot --alg fedbyot_selective --kd_conf_threshold 0.8 --byot_alpha ${ALPHA_VAL} --byot_beta ${BETA_VAL}"
METHOD_MAP["warmup"]="--model resnet18_byot --alg fedbyot_selective_greedy --kd_conf_threshold 0.8 --min_threshold 0.3 --warmup_epochs 2 --byot_alpha ${ALPHA_VAL} --byot_beta ${BETA_VAL}"

for env in "beta_0.3" "beta_0.5" "noniid_grouping"; do
    # 원본 파일 논리에 따라 fedavg는 제외하고 증류 기법 적용
    for algo in "fedprox" "moon" "fedrcl" "feddecorr"; do
        for method in "fedsd" "selective" "warmup"; do
            run_job "$env" "${ENV_MAP[$env]}" "$method" "${METHOD_MAP[$method]}" "$algo" "${ALGO_MAP[$algo]}"
        done
    done
done

wait
echo "🎉 모든 실험이 완벽하게 종료되었습니다!"