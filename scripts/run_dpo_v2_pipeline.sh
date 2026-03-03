#!/bin/bash
# Full pipeline: Wait for eval → DPO v2 training (on Qwen base) → Eval
#
# DPO v1 was trained on SFT model (bad). v2 trains on Qwen2.5-7B-Instruct.
#
# Usage:
#   nohup bash scripts/run_dpo_v2_pipeline.sh > logs/dpo_v2_pipeline.log 2>&1 &

set -euo pipefail

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DPO_DATASET="outputs/dpo_dataset_bva_messages.jsonl"
DPO_OUTPUT="outputs/dpo_policy_v2"
DPO_WANDB_NAME="dpo_policy_v2_qwen_base"

NUM_Q=7405
TOP_K=3
TEMP=0.8
GPU_UTIL=0.85

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

mkdir -p logs

# ---------------------------------------------------------------
# [0/4] Wait for current full eval to finish
# ---------------------------------------------------------------
EVAL_WRAPPER_PID=1799569
if kill -0 $EVAL_WRAPPER_PID 2>/dev/null; then
    echo "[0/4] Waiting for full eval pipeline (PID $EVAL_WRAPPER_PID) to finish..."
    while kill -0 $EVAL_WRAPPER_PID 2>/dev/null; do
        sleep 60
        echo "  Still waiting... $(date '+%H:%M:%S')"
    done
    echo "[0/4] Full eval finished at $(date)"
else
    echo "[0/4] Full eval already finished."
fi

echo ""
echo "============================================================"
echo "DPO v2 PIPELINE — Training on Qwen2.5-7B-Instruct base"
echo "============================================================"
echo "  Base model:  $BASE_MODEL"
echo "  Dataset:     $DPO_DATASET ($(wc -l < $DPO_DATASET) pairs)"
echo "  Output:      $DPO_OUTPUT"
echo "============================================================"

# ---------------------------------------------------------------
# [1/4] DPO v2 Training
# ---------------------------------------------------------------
DPO_DONE="${DPO_OUTPUT}/.done"
if [ -f "$DPO_DONE" ]; then
    echo "[1/4] DPO v2 training already done. Skipping."
else
    echo ""
    echo "[1/4] Starting DPO v2 training..."

    torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
        --dpo-dataset "$DPO_DATASET" \
        --model-name "$BASE_MODEL" \
        --output-dir "$DPO_OUTPUT" \
        --wandb-run-name "$DPO_WANDB_NAME" \
        --beta 0.1 \
        --loss-type sigmoid \
        --learning-rate 5e-7 \
        --num-epochs 3 \
        --batch-size 1 \
        --gradient-accumulation 16 \
        --lora-r 16 \
        --lora-alpha 32 \
        --max-length 4096 \
        --logging-steps 10 \
        --save-steps 200

    touch "$DPO_DONE"
    echo "[1/4] DPO v2 training done."
fi

# ---------------------------------------------------------------
# [2/4] Evaluate DPO v2 model (Qwen base + DPO v2 LoRA)
# ---------------------------------------------------------------
DPO_V2_TAG="dpo_v2_full"
DPO_V2_EVAL_DONE="outputs/flashrag_${DPO_V2_TAG}/.done"

if [ -f "$DPO_V2_EVAL_DONE" ]; then
    echo "[2/4] DPO v2 eval already done. Skipping."
else
    echo ""
    echo "[2/4] Evaluating DPO v2 model (${NUM_Q}q)..."

    python scripts/eval_flashrag.py \
        --model "$BASE_MODEL" \
        --lora "${DPO_OUTPUT}/final_model" \
        --tag "$DPO_V2_TAG" \
        --top-k "$TOP_K" \
        --limit "$NUM_Q" \
        --temperature "$TEMP" \
        --gpu-util "$GPU_UTIL"

    touch "$DPO_V2_EVAL_DONE"
    echo "[2/4] DPO v2 eval done."
fi

# ---------------------------------------------------------------
# [3/4] Evaluate Qwen base (reuse if already exists from run_full_eval)
# ---------------------------------------------------------------
BASE_TAG="qwen_base_full"
BASE_EVAL_DONE="outputs/flashrag_${BASE_TAG}/.done"

if [ -f "$BASE_EVAL_DONE" ]; then
    echo "[3/4] Qwen base eval already done. Reusing."
else
    echo ""
    echo "[3/4] Evaluating Qwen base (${NUM_Q}q)..."

    python scripts/eval_flashrag.py \
        --model "$BASE_MODEL" \
        --tag "$BASE_TAG" \
        --top-k "$TOP_K" \
        --limit "$NUM_Q" \
        --temperature "$TEMP" \
        --gpu-util "$GPU_UTIL"

    touch "$BASE_EVAL_DONE"
    echo "[3/4] Qwen base eval done."
fi

# ---------------------------------------------------------------
# [4/4] Compare all three: Base / DPO v1 (SFT+LoRA) / DPO v2 (Qwen+LoRA)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "[4/4] Computing metrics (EM / Cover EM / F1)"
echo "============================================================"

DPO_V1_TAG="dpo_bva_full" DPO_V2_TAG="$DPO_V2_TAG" BASE_TAG="$BASE_TAG" python3 - << 'PYEOF'
import json, os, re, string, glob

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(s.split()).strip()

def check_cover_em(gold_answers, retrieval_results):
    all_text = ""
    for step_data in retrieval_results.values():
        if 'docs' in step_data:
            for doc in step_data['docs']:
                all_text += " " + doc.get('contents', '')
    all_text_norm = normalize_answer(all_text)
    return any(normalize_answer(a) in all_text_norm for a in gold_answers)

def find_latest(tag):
    base = f"outputs/flashrag_{tag}"
    dirs = sorted(glob.glob(os.path.join(base, "hotpotqa_*_experiment")))
    return os.path.join(dirs[-1], "intermediate_data.json") if dirs else None

def compute(path):
    if not path or not os.path.exists(path):
        return None
    data = json.load(open(path))
    total = len(data)
    em = cover = no_ret = 0
    f1_sum = 0.0
    total_steps = 0
    for item in data:
        out = item['output']
        golds = item['golden_answers']
        ret = out.get('retrieval_results', {})
        if out.get('metric_score', {}).get('em', 0) == 1.0:
            em += 1
        f1_sum += out.get('metric_score', {}).get('f1', 0.0)
        if not ret:
            no_ret += 1
        elif check_cover_em(golds, ret):
            cover += 1
        total_steps += out.get('retrieved_times', 0)
    return dict(
        total=total, em=em, em_r=em/total*100,
        cover=cover, cover_r=cover/total*100,
        f1=f1_sum/total*100,
        no_ret=no_ret,
        avg_steps=total_steps/total,
    )

# Load results
models = {}
for tag, label in [
    (os.environ["BASE_TAG"],   "Qwen2.5-7B Base"),
    (os.environ["DPO_V1_TAG"], "DPO v1 (SFT+LoRA)"),
    (os.environ["DPO_V2_TAG"], "DPO v2 (Qwen+LoRA)"),
]:
    path = find_latest(tag)
    m = compute(path) if path else None
    if m:
        models[label] = m
    else:
        print(f"  [SKIP] {label}: no data found for tag={tag}")

if not models:
    print("ERROR: No results found!")
    exit(1)

base = models.get("Qwen2.5-7B Base")

W = 88
print("=" * W)
n = base['total'] if base else '?'
print(f"RESULTS — HotpotQA ({n}q) | e5-base | top_k=3 | temp=0.8")
print("=" * W)

header = f"{'Metric':<22}"
for label in models:
    header += f" {label:>20}"
if base:
    header += f" {'v2 Delta':>10}"
print(header)
print("-" * W)

for metric, key, fmt in [
    ("EM",             "em_r",     ".2f"),
    ("F1",             "f1",       ".2f"),
    ("Cover EM",       "cover_r",  ".2f"),
    ("Avg Ret. Steps", "avg_steps", ".2f"),
    ("No Retrieval",   "no_ret",   "d"),
]:
    row = f"{metric:<22}"
    v2_val = None
    for label, m in models.items():
        val = m[key]
        if fmt == "d":
            row += f" {val:>20d}"
        elif key in ("em_r", "f1", "cover_r"):
            row += f" {val:>19.2f}%"
        else:
            row += f" {val:>20.2f}"
        if "v2" in label:
            v2_val = val
    if base and v2_val is not None and key in ("em_r", "f1", "cover_r"):
        delta = v2_val - base[key]
        row += f" {delta:>+9.2f}%"
    print(row)

print("=" * W)

# Save summary
summary = {"models": {}, "dataset": "hotpotqa", "top_k": 3, "temp": 0.8}
for label, m in models.items():
    summary["models"][label] = {
        "em": m["em_r"], "f1": m["f1"], "cover_em": m["cover_r"],
        "avg_steps": m["avg_steps"], "total": m["total"],
    }
out_path = "outputs/flashrag_full_comparison_v2.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out_path}")
PYEOF

echo ""
echo "[dpo_v2_pipeline] All done at $(date)"
