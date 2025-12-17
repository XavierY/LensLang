#!/bin/bash
#SBATCH --job-name=fsv_nolenslang_train
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gres=gpu:1              # 如无GPU节点，可先注释本行
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

# -------- 只能在 Slurm 作业里跑，防止误在登录节点直接 bash --------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[ERROR] This script must be submitted with: sbatch run_nolenslang_fakesv_train.sh"
  exit 1
fi

# -------- 项目与日志目录 --------
PROJECT_ROOT="/data/hyan671/yhproject/FakingRecipe"
cd "${PROJECT_ROOT}"

LOG_ROOT="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

JID="${SLURM_JOB_ID}"
TS="$(date +%Y%m%d_%H%M%S)"

RUN_DIR="${LOG_ROOT}/run_nolenslang_fakesv_train_${JID}_${TS}"
mkdir -p "${RUN_DIR}"

echo "[INFO] RUN_DIR = ${RUN_DIR}"
echo "[INFO] SLURM_JOB_ID = ${JID}"

# -------- 激活 conda 环境 --------
source /data/hyan671/anaconda3/etc/profile.d/conda.sh
conda activate lenslang

echo "[ENV] which python: $(which python)"
python -V

# -------- 启动资源监控（后台） --------
# GPU 监控
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used \
             --format=csv -l 5 > "${RUN_DIR}/fsv_nolenslang_gpu_${JID}.csv" &
  GPU_MON=$!
else
  GPU_MON=""
fi

# CPU / 内存监控
vmstat 1 > "${RUN_DIR}/fsv_nolenslang_vmstat_${JID}.log" &
VM_MON=$!

# -------- 运行无 LensLang 的基线模型（训练） --------
stdbuf -oL -eL python -u main.py \
  --dataset fakesv \
  --mode train \
  --batch_size 128 \
  --gpu 0 \
  --lr 1e-4 \
  --alpha 1.0 \
  --beta 1.0 \
  2>&1 | tee "${RUN_DIR}/nolenslang_fakesv_train_full_${JID}.log"

EC=${PIPESTATUS[0]}
echo "PYTHON_EXIT_CODE=${EC}" | tee -a "${RUN_DIR}/nolenslang_fakesv_train_full_${JID}.log"

# -------- 清理监控进程 --------
if [ -n "${GPU_MON:-}" ]; then
  kill "${GPU_MON}" 2>/dev/null || true
fi
kill "${VM_MON}" 2>/dev/null || true

exit "${EC}"