#!/bin/bash
# =============================================================================
# Step-level DPO 실험 자동화
#
# 1. 데이터 구축 완료 대기
# 2. DPO 학습 (SFT v2 base, epoch 1)
# 3. LoRA Merge
# 4. 평가
#
# Usage:
#   nohup bash scripts/run_step_dpo_experiment.sh > logs/step_dpo_experiment.log 2>&1 &
# =============================================================================

set -euo pipefail

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL="Qwen/Qwen2.5-7B-Instruct"
DPO_DATA="outputs/step_dpo_5000q_full.jsonl"
DPO_OUTPUT="outputs/dpo_step_level_v1_base"
WANDB_NAME="step_dpo_v1_base_good_bad_ep1"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NUMEXPR_MAX_THREADS=64

echo "=============================================================="
echo "  Step-level DPO Experiment"
echo "  Base: Qwen2.5-7B-Instruct (base, no SFT)"
echo "  Data: Step-level GOOD vs BAD + Regen K=3"
echo "  Epoch: 1"
echo "=============================================================="

# ---------------------------------------------------------------
# Step 1: 데이터 구축 완료 대기
# ---------------------------------------------------------------
BUILD_PID=$(ps aux | grep "build_step_level_dpo" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$BUILD_PID" ]; then
    echo ""
    echo "[Step 1] 데이터 구축 완료 대기 (PID: $BUILD_PID)..."
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        sleep 60
        echo "  $(date '+%Y-%m-%d %H:%M:%S') — 아직 빌드 중..."
    done
    echo "[Step 1] 데이터 구축 완료!"
else
    echo "[Step 1] 빌드 프로세스 없음 — 데이터 이미 완료된 것으로 간주"
fi

# 데이터 확인
if [ ! -f "$DPO_DATA" ]; then
    echo "[ERROR] DPO 데이터 없음: $DPO_DATA"
    exit 1
fi
echo "  DPO 데이터: $(wc -l < "$DPO_DATA") pairs"
echo ""

# ---------------------------------------------------------------
# Step 2: DPO 학습 (SFT v2 base, epoch 1)
# ---------------------------------------------------------------
echo "[Step 2] DPO 학습 시작..."
echo "  Model: $SFT_MODEL"
echo "  Data:  $DPO_DATA"
echo "  Output: $DPO_OUTPUT"
echo ""

torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
    --dpo-dataset "$DPO_DATA" \
    --model-name "$SFT_MODEL" \
    --output-dir "$DPO_OUTPUT" \
    --num-epochs 1 \
    --wandb-run-name "$WANDB_NAME"

echo "[Step 2] DPO 학습 완료!"
echo ""

# ---------------------------------------------------------------
# Step 3: LoRA Merge
# ---------------------------------------------------------------
echo "[Step 3] LoRA Merge..."

python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
cache = '$HF_HOME'
base = '$SFT_MODEL'
lora = '${DPO_OUTPUT}/final_model'
out = '${DPO_OUTPUT}/merged_model'
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, cache_dir=cache, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()
model.save_pretrained(out)
tokenizer = AutoTokenizer.from_pretrained(base, cache_dir=cache, trust_remote_code=True)
tokenizer.save_pretrained(out)
print(f'Merged → {out}')
"

echo "[Step 3] Merge 완료: ${DPO_OUTPUT}/merged_model"
echo ""

# ---------------------------------------------------------------
# Step 4: 평가
# ---------------------------------------------------------------
echo "[Step 4] 평가 시작..."

python scripts/eval_flashrag.py \
    --model "${DPO_OUTPUT}/merged_model" \
    --tag "step_dpo_v1_good_bad" \
    --top-k 3 \
    --limit 7405 \
    --temperature 0.8 \
    --gpu-memory 0.88

echo ""
echo "=============================================================="
echo "  Step-level DPO 실험 완료!"
echo "=============================================================="
echo ""
echo "결과 확인:"
echo "  모델: ${DPO_OUTPUT}/merged_model"
echo "  평가: outputs/flashrag_step_dpo_v1_good_bad/"
echo "=============================================================="
