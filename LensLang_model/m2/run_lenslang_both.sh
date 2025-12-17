#!/bin/bash
#SBATCH --job-name=lenslang_both_train
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gres=gpu:1              # 如无GPU节点，可先注释本行
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00           # 两个数据集合计时间，按需调整

set -euo pipefail

# -------- 只能在 Slurm 作业里跑，防止误在登录节点直接 bash --------
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "[ERROR] This script must be submitted with: sbatch run_lenslang_both.sh"
  exit 1
fi

# -------- 项目与日志目录 --------
PROJECT_ROOT="/data/hyan671/yhproject/FakingRecipe"
cd "${PROJECT_ROOT}"

LOG_ROOT="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

JID="${SLURM_JOB_ID}"

echo "[INFO] SLURM_JOB_ID = ${JID}"
echo "[INFO] PROJECT_ROOT = ${PROJECT_ROOT}"

# -------- 激活 conda 环境 --------
source /data/hyan671/anaconda3/etc/profile.d/conda.sh
conda activate lenslang

echo "[ENV] which python: $(which python)"
python -V

############################################
#           Block 1: FakeSV 训练           #
############################################
TS_FSV="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_FSV="${LOG_ROOT}/run_lenslang_fakesv_${JID}_${TS_FSV}"
mkdir -p "${RUN_DIR_FSV}"

echo "[INFO] ===== FakeSV training start ====="
echo "[INFO] RUN_DIR_FSV = ${RUN_DIR_FSV}"

# GPU 监控（FakeSV）
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used \
             --format=csv -l 5 > "${RUN_DIR_FSV}/fsv_gpu_${JID}.csv" &
  GPU_MON_FSV=$!
else
  GPU_MON_FSV=""
fi

# CPU / 内存监控（FakeSV）
vmstat 1 > "${RUN_DIR_FSV}/fsv_vmstat_${JID}.log" &
VM_MON_FSV=$!

# 运行 FakeSV LensLang 版本（训练）
stdbuf -oL -eL python -u main_lenslang.py \
  --dataset fakesv \
  --mode train \
  --use_lenslang \
  --lenslang_root /data/hyan671/yhproject/FakingRecipe/fea/fakesv/preprocess_lenslang \
  --batch_size 128 \
  --gpu 0 \
  --lr 1e-4 \
  --alpha 1.0 \
  --beta 1.0 \
  2>&1 | tee "${RUN_DIR_FSV}/lenslang_fakesv_full_${JID}.log"

EC_FSV=${PIPESTATUS[0]}
echo "PYTHON_EXIT_CODE_FAKESV=${EC_FSV}" | tee -a "${RUN_DIR_FSV}/lenslang_fakesv_full_${JID}.log"

# 清理 FakeSV 监控进程
if [ -n "${GPU_MON_FSV:-}" ]; then
  kill "${GPU_MON_FSV}" 2>/dev/null || true
fi
kill "${VM_MON_FSV}" 2>/dev/null || true

if [ "${EC_FSV}" -ne 0 ]; then
  echo "[ERROR] FakeSV training failed with code ${EC_FSV}, aborting FakeTT."
  exit "${EC_FSV}"
fi

echo "[INFO] ===== FakeSV training finished OK ====="

############################################
#           Block 2: FakeTT 训练           #
############################################
TS_FTT="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_FTT="${LOG_ROOT}/run_lenslang_fakett_${JID}_${TS_FTT}"
mkdir -p "${RUN_DIR_FTT}"

echo "[INFO] ===== FakeTT training start ====="
echo "[INFO] RUN_DIR_FTT = ${RUN_DIR_FTT}"

# GPU 监控（FakeTT）
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used \
             --format=csv -l 5 > "${RUN_DIR_FTT}/ftt_gpu_${JID}.csv" &
  GPU_MON_FTT=$!
else
  GPU_MON_FTT=""
fi

# CPU / 内存监控（FakeTT）
vmstat 1 > "${RUN_DIR_FTT}/ftt_vmstat_${JID}.log" &
VM_MON_FTT=$!

# 运行 FakeTT LensLang 版本（训练）
stdbuf -oL -eL python -u main_lenslang_fakett.py \
  --dataset fakett \
  --mode train \
  --use_lenslang \
  --lenslang_root /data/hyan671/yhproject/FakingRecipe/fea/fakett/preprocess_lenslang \
  --batch_size 128 \
  --gpu 0 \
  --lr 1e-4 \
  --alpha 1.0 \
  --beta 1.0 \
  2>&1 | tee "${RUN_DIR_FTT}/lenslang_fakett_full_${JID}.log"

EC_FTT=${PIPESTATUS[0]}
echo "PYTHON_EXIT_CODE_FAKETT=${EC_FTT}" | tee -a "${RUN_DIR_FTT}/lenslang_fakett_full_${JID}.log"

# 清理 FakeTT 监控进程
if [ -n "${GPU_MON_FTT:-}" ]; then
  kill "${GPU_MON_FTT}" 2>/dev/null || true
fi
kill "${VM_MON_FTT}" 2>/dev/null || true

if [ "${EC_FTT}" -ne 0 ]; then
  echo "[ERROR] FakeTT training failed with code ${EC_FTT}."
  exit "${EC_FTT}"
fi

echo "[INFO] ===== FakeTT training finished OK ====="
echo "[INFO] All tasks finished successfully."

exit 0