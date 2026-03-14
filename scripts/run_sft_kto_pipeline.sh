#!/bin/bash
# SFT + KTO from DPO chosen/rejected → merge → eval
# Same data as V1 DPO (mcts_dpo_5000q_v4b.jsonl, 14,239 pairs)
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

export NUMEXPR_MAX_THREADS=64
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface

DPO_DATA="outputs/mcts_dpo_5000q_v4b.jsonl"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASETS="popqa hotpotqa 2wikimultihopqa bamboogle musique"

# ============================================================
# PHASE 1: SFT Training (chosen only)
# ============================================================
SFT_DIR="outputs/sft_from_dpo_v1"
if [ -f "$SFT_DIR/final_model/adapter_config.json" ]; then
    echo "[SKIP] SFT training — already done"
else
    echo "=============================================="
    echo "PHASE 1: SFT Training (DPO chosen)"
    echo "=============================================="
    torchrun --nproc_per_node=2 scripts/train_sft_from_dpo.py \
        --dpo-dataset "$DPO_DATA" \
        --model-name "$BASE_MODEL" \
        --output-dir "$SFT_DIR" \
        --num-epochs 1 \
        --batch-size 1 \
        --gradient-accumulation 8 \
        --learning-rate 2e-5 \
        --lora-r 64 --lora-alpha 128 \
        --wandb-run-name sft_from_dpo_v1
    echo "[DONE] SFT Training"
fi

# ============================================================
# PHASE 2: SFT Merge
# ============================================================
SFT_MERGED="$SFT_DIR/merged_model"
if [ -f "$SFT_MERGED/config.json" ]; then
    echo "[SKIP] SFT merge — already done"
else
    echo "=============================================="
    echo "PHASE 2: SFT Merge LoRA"
    echo "=============================================="
    python scripts/merge_lora.py \
        --base-model "$BASE_MODEL" \
        --lora-path "$SFT_DIR/final_model" \
        --output-dir "$SFT_MERGED"
    echo "[DONE] SFT Merge"
fi

# ============================================================
# PHASE 3: SFT Eval
# ============================================================
echo ""
echo "=============================================="
echo "PHASE 3: SFT Eval"
echo "=============================================="
for ds in $DATASETS; do
    tag="sft_dpo_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] SFT $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] SFT eval $ds"
    python scripts/eval_flashrag.py \
        --model "$SFT_MERGED" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0
    echo "[DONE] SFT eval $ds"
    cat "$result_dir"/*/metric_score.txt
done

# ============================================================
# PHASE 4: KTO Training (chosen=desirable, rejected=undesirable)
# ============================================================
KTO_DIR="outputs/kto_from_dpo_v1"
if [ -f "$KTO_DIR/final_model/adapter_config.json" ]; then
    echo "[SKIP] KTO training — already done"
else
    echo ""
    echo "=============================================="
    echo "PHASE 4: KTO Training (DPO chosen/rejected)"
    echo "=============================================="
    torchrun --nproc_per_node=2 scripts/train_kto_from_dpo.py \
        --dpo-dataset "$DPO_DATA" \
        --model-name "$BASE_MODEL" \
        --output-dir "$KTO_DIR" \
        --beta 0.1 \
        --num-epochs 1 \
        --batch-size 1 \
        --gradient-accumulation 8 \
        --learning-rate 5e-6 \
        --lora-r 64 --lora-alpha 128 \
        --wandb-run-name kto_from_dpo_v1
    echo "[DONE] KTO Training"
fi

# ============================================================
# PHASE 5: KTO Merge
# ============================================================
KTO_MERGED="$KTO_DIR/merged_model"
if [ -f "$KTO_MERGED/config.json" ]; then
    echo "[SKIP] KTO merge — already done"
else
    echo ""
    echo "=============================================="
    echo "PHASE 5: KTO Merge LoRA"
    echo "=============================================="
    python scripts/merge_lora.py \
        --base-model "$BASE_MODEL" \
        --lora-path "$KTO_DIR/final_model" \
        --output-dir "$KTO_MERGED"
    echo "[DONE] KTO Merge"
fi

# ============================================================
# PHASE 6: KTO Eval
# ============================================================
echo ""
echo "=============================================="
echo "PHASE 6: KTO Eval"
echo "=============================================="
for ds in $DATASETS; do
    tag="kto_dpo_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] KTO $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] KTO eval $ds"
    python scripts/eval_flashrag.py \
        --model "$KTO_MERGED" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0
    echo "[DONE] KTO eval $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "ALL SFT + KTO PIPELINE COMPLETE"
echo "=============================================="
