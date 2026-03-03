#!/bin/bash
# Auto-evaluation after DPO training completes.
# Evaluates DPO model vs Qwen base using E5 retriever on HotpotQA val.
#
# Usage:
#   nohup bash scripts/auto_eval_after_dpo.sh > logs/auto_eval.log 2>&1 &

set -euo pipefail

VAL_DATA="data/raw/questions/hotpotqa_validation.jsonl"
NUM_Q=500

DPO_MODEL="outputs/dpo_policy_v1/final_model"
DPO_BASE_MODEL="outputs/sft_policy_v1/merged_model"   # DPO LoRA was trained on this
BASELINE_MODEL="Qwen/Qwen2.5-7B-Instruct"             # Baseline comparison

DPO_OUT="outputs/eval_dpo_bva_e5_hotpotqa_val${NUM_Q}.jsonl"
BASE_OUT="outputs/eval_base_e5_hotpotqa_val${NUM_Q}.jsonl"

mkdir -p logs

# ---------------------------------------------------------------
# [1] Evaluate DPO model (SFT + DPO LoRA) with E5 retriever
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "[1/2] Evaluating DPO model + E5 retriever"
echo "  Base:    $DPO_BASE_MODEL"
echo "  LoRA:    $DPO_MODEL"
echo "  Output:  $DPO_OUT"
echo "============================================================"

if [ -f "$DPO_OUT" ] && [ "$(wc -l < "$DPO_OUT")" -ge "$NUM_Q" ]; then
    echo "[1/2] Already complete ($DPO_OUT). Skipping."
else
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    python scripts/eval_policy.py \
        --data "$VAL_DATA" \
        --output "$DPO_OUT" \
        --policy-model "$DPO_BASE_MODEL" \
        --lora-adapter "$DPO_MODEL" \
        --retriever e5 \
        --limit "$NUM_Q" \
        --temperature 0.0 \
        --max-steps 10 \
        --batch-size 50 \
        --gpu-memory-utilization 0.85 \
        --resume
    echo "[1/2] DPO eval done."
fi

# ---------------------------------------------------------------
# [2] Evaluate Qwen base (baseline) with E5 retriever
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "[2/2] Evaluating Qwen base (baseline) + E5 retriever"
echo "  Model:   $BASELINE_MODEL"
echo "  Output:  $BASE_OUT"
echo "============================================================"

if [ -f "$BASE_OUT" ] && [ "$(wc -l < "$BASE_OUT")" -ge "$NUM_Q" ]; then
    echo "[2/2] Already complete ($BASE_OUT). Skipping."
else
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    python scripts/eval_policy.py \
        --data "$VAL_DATA" \
        --output "$BASE_OUT" \
        --policy-model "$BASELINE_MODEL" \
        --retriever e5 \
        --limit "$NUM_Q" \
        --temperature 0.0 \
        --max-steps 10 \
        --batch-size 50 \
        --gpu-memory-utilization 0.85 \
        --resume
    echo "[2/2] Baseline eval done."
fi

# ---------------------------------------------------------------
# [3] Comparison
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "RESULTS COMPARISON (HotpotQA val ${NUM_Q}q, E5 retriever)"
echo "============================================================"

BASE_OUT="$BASE_OUT" DPO_OUT="$DPO_OUT" python3 - <<'EOF'
import json, os

def load(path):
    if not path or not os.path.exists(path):
        return 0, 0
    lines = [l for l in open(path) if l.strip()]
    total = len(lines)
    correct = sum(1 for l in lines if json.loads(l).get("is_correct", False))
    return total, correct

base_t, base_c = load(os.environ["BASE_OUT"])
dpo_t, dpo_c = load(os.environ["DPO_OUT"])

print(f"\n{'Model':<32} {'Total':>7} {'Correct':>8} {'Accuracy':>10}")
print("-" * 62)
if base_t:
    base_acc = base_c / base_t * 100
    print(f"{'Qwen2.5-7B-Instruct (base)':<32} {base_t:>7} {base_c:>8} {base_acc:>9.2f}%")
else:
    print(f"{'Qwen2.5-7B-Instruct (base)':<32} {'N/A':>7}")
    base_acc = None

if dpo_t:
    dpo_acc = dpo_c / dpo_t * 100
    print(f"{'DPO (best-vs-all)':<32} {dpo_t:>7} {dpo_c:>8} {dpo_acc:>9.2f}%")
else:
    print(f"{'DPO (best-vs-all)':<32} {'N/A':>7}")
    dpo_acc = None

if base_acc is not None and dpo_acc is not None:
    delta = dpo_acc - base_acc
    print(f"\nDelta (DPO - Base): {delta:+.2f}%")
EOF

echo ""
echo "[auto_eval] Finished at $(date)"
