#!/bin/bash
# =============================================================================
# 학습 실험 스크립트
#
# 데이터 생성(run_round2_3_data.sh)이 완료된 후 사용
# 다양한 학습 설정을 실험하기 위한 스크립트
#
# Usage:
#   # 특정 실험만 실행
#   bash scripts/run_training_experiments.sh exp7
#   bash scripts/run_training_experiments.sh exp8
#
#   # 전체 실행
#   nohup bash scripts/run_training_experiments.sh all > logs/training_experiments.log 2>&1 &
# =============================================================================

set -euo pipefail

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NUMEXPR_MAX_THREADS=64

# ---------------------------------------------------------------
# Helper: SFT + Merge + Eval
# ---------------------------------------------------------------
run_sft() {
    local INPUT=$1
    local OUTPUT_DIR=$2
    local WANDB_NAME=$3
    local EPOCHS=${4:-1}

    echo "  [SFT] $WANDB_NAME — input: $INPUT"

    # SFT 학습
    torchrun --nproc_per_node=2 scripts/train_sft_policy.py \
        --input "$INPUT" \
        --model-name "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --num-epochs "$EPOCHS" \
        --wandb-run-name "$WANDB_NAME"

    # LoRA Merge
    echo "  [Merge] $OUTPUT_DIR/merged_model"
    python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
cache = '$HF_HOME'
base = '$BASE_MODEL'
lora = '${OUTPUT_DIR}/final_model'
out = '${OUTPUT_DIR}/merged_model'
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, cache_dir=cache, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()
model.save_pretrained(out)
tokenizer = AutoTokenizer.from_pretrained(base, cache_dir=cache, trust_remote_code=True)
tokenizer.save_pretrained(out)
print(f'Merged → {out}')
"
}

run_dpo() {
    local DPO_DATA=$1
    local SFT_MODEL=$2
    local OUTPUT_DIR=$3
    local WANDB_NAME=$4
    local EPOCHS=${5:-1}

    echo "  [DPO] $WANDB_NAME — epochs: $EPOCHS"

    torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
        --dpo-dataset "$DPO_DATA" \
        --model-name "$SFT_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --num-epochs "$EPOCHS" \
        --wandb-run-name "$WANDB_NAME"
}

run_eval() {
    local MODEL=$1
    local TAG=$2
    local LORA=${3:-""}

    echo "  [Eval] $TAG"

    local LORA_ARG=""
    if [ -n "$LORA" ]; then
        LORA_ARG="--lora $LORA"
    fi

    python scripts/eval_flashrag.py \
        --model "$MODEL" \
        $LORA_ARG \
        --tag "$TAG" \
        --top-k 3 \
        --limit 7405 \
        --temperature 0.8 \
        --gpu-util 0.85

    echo "  결과:"
    cat "outputs/flashrag_${TAG}/hotpotqa_"*"/metric_score.txt" 2>/dev/null || echo "  (결과 파일 없음)"
}

# ---------------------------------------------------------------
# 실험 정의
# ---------------------------------------------------------------

# Exp 7: SFT 4,000q only
exp7() {
    echo "======================================================"
    echo "  Exp 7: SFT 4,000q only"
    echo "======================================================"
    run_sft "outputs/all_4000q_sft_input.jsonl" "outputs/sft_policy_v3_4000q" "sft_v3_4000q"
    run_eval "outputs/sft_policy_v3_4000q/merged_model" "sft_v3_4000q"
}

# Exp 8: SFT 5,000q only
exp8() {
    echo "======================================================"
    echo "  Exp 8: SFT 5,000q only"
    echo "======================================================"
    run_sft "outputs/all_5000q_sft_input.jsonl" "outputs/sft_policy_v4_5000q" "sft_v4_5000q"
    run_eval "outputs/sft_policy_v4_5000q/merged_model" "sft_v4_5000q"
}

# Exp 9: SFT 5,000q + DPO ep1
exp9() {
    echo "======================================================"
    echo "  Exp 9: SFT 5,000q + DPO ep1 (prompt 통일)"
    echo "======================================================"
    # SFT가 이미 있으면 skip
    if [ ! -f "outputs/sft_policy_v4_5000q/merged_model/config.json" ]; then
        run_sft "outputs/all_5000q_sft_input.jsonl" "outputs/sft_policy_v4_5000q" "sft_v4_5000q"
    fi
    run_dpo "outputs/dpo_dataset_v4_5000q_bva.jsonl" "outputs/sft_policy_v4_5000q/merged_model" \
        "outputs/dpo_policy_v4_5000q_ep1" "dpo_v4_5000q_ep1" 1
    run_eval "outputs/sft_policy_v4_5000q/merged_model" "dpo_v4_5000q_ep1" \
        "outputs/dpo_policy_v4_5000q_ep1/final_model"
}

# Exp 10: SFT 5,000q + DPO ep3
exp10() {
    echo "======================================================"
    echo "  Exp 10: SFT 5,000q + DPO ep3 (prompt 통일)"
    echo "======================================================"
    if [ ! -f "outputs/sft_policy_v4_5000q/merged_model/config.json" ]; then
        run_sft "outputs/all_5000q_sft_input.jsonl" "outputs/sft_policy_v4_5000q" "sft_v4_5000q"
    fi
    run_dpo "outputs/dpo_dataset_v4_5000q_bva.jsonl" "outputs/sft_policy_v4_5000q/merged_model" \
        "outputs/dpo_policy_v4_5000q_ep3" "dpo_v4_5000q_ep3" 3
    run_eval "outputs/sft_policy_v4_5000q/merged_model" "dpo_v4_5000q_ep3" \
        "outputs/dpo_policy_v4_5000q_ep3/final_model"
}

# Exp 11: SFT 4,000q + DPO ep1
exp11() {
    echo "======================================================"
    echo "  Exp 11: SFT 4,000q + DPO ep1 (prompt 통일)"
    echo "======================================================"
    if [ ! -f "outputs/sft_policy_v3_4000q/merged_model/config.json" ]; then
        run_sft "outputs/all_4000q_sft_input.jsonl" "outputs/sft_policy_v3_4000q" "sft_v3_4000q"
    fi
    run_dpo "outputs/dpo_dataset_v3_4000q_bva.jsonl" "outputs/sft_policy_v3_4000q/merged_model" \
        "outputs/dpo_policy_v3_4000q_ep1" "dpo_v3_4000q_ep1" 1
    run_eval "outputs/sft_policy_v3_4000q/merged_model" "dpo_v3_4000q_ep1" \
        "outputs/dpo_policy_v3_4000q_ep1/final_model"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
EXPERIMENT=${1:-"all"}

case "$EXPERIMENT" in
    exp7)  exp7 ;;
    exp8)  exp8 ;;
    exp9)  exp9 ;;
    exp10) exp10 ;;
    exp11) exp11 ;;
    all)
        exp7
        exp8
        exp9
        exp10
        exp11
        ;;
    *)
        echo "Usage: $0 {exp7|exp8|exp9|exp10|exp11|all}"
        exit 1
        ;;
esac

echo ""
echo "======================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================"
