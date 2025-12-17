#!/usr/bin/env bash
set -euo pipefail

# ===== 唯一需要改的 5 个路径 =====
# FakeSV
# VIDEO_DIR="/data/hyan671/yhproject/FakingRecipe/dataio/FakeSV/videos"
# OUT_DIR="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/out_lang"

# FakeTT
VIDEO_DIR="/data/hyan671/yhproject/FakingRecipe/dataio/FakeTT/video"
OUT_DIR="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeTT_json"

MODEL_PATH="/data/hyan671/models/DeepSeek-R1-Distill-Qwen-32B"
SCRIPT="/data/hyan671/yhproject/FakingRecipe/lenslang/scripts/annotate_lenslang_local32b.py"
LOG_DIR="/data/hyan671/yhproject/FakingRecipe/lenslang/logs"

# ===== 资源与作业名 =====
GPUS=${GPUS:-1}
CPUS=${CPUS:-8}
MEM=${MEM:-48G}
TIME=${TIME:-0}
JOB_NAME=${JOB_NAME:-lenslang-vllm-auto}

JOB_DIR="/data/hyan671/yhproject/FakingRecipe/lenslang/jobs"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$JOB_DIR" /data/hyan671/.cache /data/hyan671/.config

JOB_FILE="${JOB_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).sh"

# 不加引号，让上面的变量在生成时写死；\${...} 会保留到作业运行时再展开
cat > "$JOB_FILE" <<BATCH
#!/bin/bash
set -euo pipefail
umask 002

echo "[INFO] Node: \$(hostname)"
echo "[INFO] Start: \$(date)"
nvidia-smi || true

export HF_HOME=/data/hyan671/.cache/huggingface
export XDG_CACHE_HOME=/data/hyan671/.cache
export XDG_CONFIG_HOME=/data/hyan671/.config
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

# 每个作业自己的日志目录：logs/<JOB_NAME>_<JOBID>/
LOG_TASK_DIR="${LOG_DIR}/${JOB_NAME}_\${SLURM_JOB_ID:-manual}"
mkdir -p "\$LOG_TASK_DIR" "$OUT_DIR"

# 选最空闲 GPU
NGPU=\$(nvidia-smi --list-gpus | wc -l)
if [[ "\$NGPU" -eq 0 ]]; then echo "[ERR] No GPU visible"; exit 1; fi
BEST_IDX=0; BEST_FREE=0
for ((i=0;i<NGPU;i++)); do
  FREE=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "\$i" 2>/dev/null | head -n1 || echo 0)
  [[ -z "\$FREE" ]] && FREE=0
  if (( FREE > BEST_FREE )); then BEST_FREE=\$FREE; BEST_IDX=\$i; fi
done
export CUDA_VISIBLE_DEVICES=\$BEST_IDX
echo "[INFO] Picked GPU \$BEST_IDX with ~\${BEST_FREE}MiB free"

GPU_MEM_UTIL=0.80; MAX_MODEL_LEN=3072; SWAP_GB=8
if   (( BEST_FREE >= 72000 )); then GPU_MEM_UTIL=0.85; MAX_MODEL_LEN=4096; SWAP_GB=8;
elif (( BEST_FREE >= 62000 )); then GPU_MEM_UTIL=0.80; MAX_MODEL_LEN=3072; SWAP_GB=8;
elif (( BEST_FREE >= 52000 )); then GPU_MEM_UTIL=0.74; MAX_MODEL_LEN=2048; SWAP_GB=12;
else                                GPU_MEM_UTIL=0.70; MAX_MODEL_LEN=1536; SWAP_GB=16;
fi

run_once () {
  srun conda run --no-capture-output -n lenslang python "$SCRIPT" \
    --video_dir "$VIDEO_DIR" \
    --out_dir   "$OUT_DIR" \
    --model     "$MODEL_PATH" \
    --report_dir "\$LOG_TASK_DIR" \
    --job_id "\${SLURM_JOB_ID:-manual}" \
    --max_tokens 512 \
    --temperature 0.2 \
    --top_p 0.9 \
    --max_model_len "\$MAX_MODEL_LEN" \
    --gpu_mem_util  "\$GPU_MEM_UTIL" \
    --tensor_parallel_size 1 \
    --swap_space "\$SWAP_GB" \
    --continue_on_error
}

if run_once; then
  echo "[OK] First attempt succeeded."
else
  echo "[WARN] First attempt failed. Retrying with conservative settings..."
  MAX_MODEL_LEN=\$(( MAX_MODEL_LEN * 3 / 4 )); (( MAX_MODEL_LEN < 1536 )) && MAX_MODEL_LEN=1536
  GPU_MEM_UTIL=0.72; SWAP_GB=\$(( SWAP_GB + 8 ))
  if run_once; then
    echo "[OK] Second attempt succeeded."
  else
    echo "[WARN] Final ultra-conservative retry..."
    MAX_MODEL_LEN=1536; GPU_MEM_UTIL=0.70; SWAP_GB=20
    run_once
  fi
fi

echo "[INFO] End: \$(date)"
BATCH

chmod +x "$JOB_FILE"
echo "[SUBMIT] $JOB_FILE"

# ===== 提交：stdout/err 定向到 logs/<JOB_NAME>_<JOBID>/*.out|*.err =====
mkdir -p "$LOG_DIR"
JOBID=$(sbatch --parsable --job-name="$JOB_NAME" \
       --gres=gpu:${GPUS} \
       --cpus-per-task=${CPUS} \
       --mem=${MEM} \
       --time=${TIME} \
       --output="${LOG_DIR}/${JOB_NAME}_%j/%x-%j.out" \
       --error="${LOG_DIR}/${JOB_NAME}_%j/%x-%j.err" \
       "$JOB_FILE")

# 预创建该作业的日志目录，避免极小概率竞态
mkdir -p "${LOG_DIR}/${JOB_NAME}_${JOBID}"

echo "[OK] Submitted as JobID=$JOBID"
echo "[LOGS] ${LOG_DIR}/${JOB_NAME}_${JOBID}/%x-${JOBID}.{out,err}"