#!/bin/bash
# Auto-evaluation after DPO training completes.
# Polls logs/train_dpo.log until "DPO TRAINING COMPLETED" appears,
# then evaluates DPO model vs SFT baseline using E5 retriever on HotpotQA val.
#
# Usage:
#   nohup bash scripts/auto_eval_after_dpo.sh > logs/auto_eval.log 2>&1 &

set -euo pipefail

VAL_DATA="data/raw/questions/hotpotqa_validation.jsonl"
TRAIN_LOG="logs/train_dpo.log"
EVAL_LOG="logs/auto_eval.log"
NUM_Q=500

DPO_MODEL="outputs/dpo_policy_v1/final_model"
SFT_MODEL="outputs/sft_policy_v1/merged_model"

DPO_OUT="outputs/eval_dpo_bva_e5_hotpotqa_val${NUM_Q}.jsonl"
SFT_OUT="outputs/eval_sft_v1_e5_hotpotqa_val${NUM_Q}.jsonl"

mkdir -p logs

echo "============================================================"
echo "[auto_eval] Started at $(date)"
echo "[auto_eval] Waiting for DPO training to complete..."
echo "============================================================"

# ---------------------------------------------------------------
# Wait for DPO training to complete
# ---------------------------------------------------------------
while true; do
    if grep -q "DPO TRAINING COMPLETED" "$TRAIN_LOG" 2>/dev/null; then
        echo "[auto_eval] DPO training completed! ($(date))"
        break
    fi
    if ! pgrep -f "train_dpo_policy.py" > /dev/null 2>&1; then
        if grep -q "DPO TRAINING COMPLETED" "$TRAIN_LOG" 2>/dev/null; then
            echo "[auto_eval] DPO training completed! ($(date))"
            break
        fi
        echo "[auto_eval] ERROR: training process gone but no completion marker found."
        tail -30 "$TRAIN_LOG"
        exit 1
    fi
    sleep 60
done

# ---------------------------------------------------------------
# Verify final model exists
# ---------------------------------------------------------------
if [ ! -d "$DPO_MODEL" ]; then
    echo "[auto_eval] ERROR: $DPO_MODEL not found. Checking for checkpoints..."
    ls outputs/dpo_policy_v1/
    # fallback to latest checkpoint
    LATEST_CKPT=$(ls -d outputs/dpo_policy_v1/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "[auto_eval] Using latest checkpoint: $LATEST_CKPT"
        DPO_MODEL="$LATEST_CKPT"
    else
        exit 1
    fi
fi
echo "[auto_eval] DPO model: $DPO_MODEL"

# ---------------------------------------------------------------
# [1] Evaluate DPO model with E5 retriever
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "[1/2] Evaluating DPO model + E5 retriever"
echo "  Model:   $DPO_MODEL (adapter on $SFT_MODEL)"
echo "  Output:  $DPO_OUT"
echo "============================================================"

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
python scripts/eval_policy.py \
    --data "$VAL_DATA" \
    --output "$DPO_OUT" \
    --policy-model "$SFT_MODEL" \
    --lora-adapter "$DPO_MODEL" \
    --retriever e5 \
    --limit "$NUM_Q" \
    --temperature 0.0 \
    --max-steps 10 \
    --batch-size 50 \
    --gpu-memory-utilization 0.85

echo "[1/2] DPO eval done."

# ---------------------------------------------------------------
# [2] Evaluate SFT baseline with E5 retriever
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "[2/2] Evaluating SFT baseline + E5 retriever"
echo "  Model:   $SFT_MODEL"
echo "  Output:  $SFT_OUT"
echo "============================================================"

if [ -f "$SFT_OUT" ]; then
    echo "[2/2] Already exists, skipping: $SFT_OUT"
else
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    python scripts/eval_policy.py \
        --data "$VAL_DATA" \
        --output "$SFT_OUT" \
        --policy-model "$SFT_MODEL" \
        --retriever e5 \
        --limit "$NUM_Q" \
        --temperature 0.0 \
        --max-steps 10 \
        --batch-size 50 \
        --gpu-memory-utilization 0.85

    echo "[2/2] SFT eval done."
fi

# ---------------------------------------------------------------
# [3] Comparison
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "RESULTS COMPARISON (HotpotQA val ${NUM_Q}q, E5 retriever)"
echo "============================================================"

SFT_OUT="$SFT_OUT" DPO_OUT="$DPO_OUT" python3 - <<'EOF'
import json, os

def load(path):
    if not path or not os.path.exists(path):
        return 0, 0
    lines = [l for l in open(path) if l.strip()]
    total = len(lines)
    correct = sum(1 for l in lines if json.loads(l).get("is_correct", False))
    return total, correct

sft_t, sft_c = load(os.environ["SFT_OUT"])
dpo_t, dpo_c = load(os.environ["DPO_OUT"])

print(f"\n{'Model':<32} {'Total':>7} {'Correct':>8} {'Accuracy':>10}")
print("-" * 62)
if sft_t:
    sft_acc = sft_c / sft_t * 100
    print(f"{'SFT baseline':<32} {sft_t:>7} {sft_c:>8} {sft_acc:>9.2f}%")
else:
    print(f"{'SFT baseline':<32} {'N/A':>7}")
    sft_acc = None

if dpo_t:
    dpo_acc = dpo_c / dpo_t * 100
    print(f"{'DPO (best-vs-all)':<32} {dpo_t:>7} {dpo_c:>8} {dpo_acc:>9.2f}%")
else:
    print(f"{'DPO (best-vs-all)':<32} {'N/A':>7}")
    dpo_acc = None

if sft_acc is not None and dpo_acc is not None:
    delta = dpo_acc - sft_acc
    print(f"\nDelta (DPO - SFT): {delta:+.2f}%")
EOF

echo ""
echo "[auto_eval] Finished at $(date)"
