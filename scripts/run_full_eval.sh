#!/bin/bash
# Full evaluation pipeline: DPO -> Base -> Compare
# Runs all 7405 HotpotQA questions, computes EM / Cover EM / F1
#
# Usage:
#   nohup bash scripts/run_full_eval.sh > logs/run_full_eval.log 2>&1 &

set -euo pipefail

NUM_Q=7405
TOP_K=3
TEMP=0.8
GPU_UTIL=0.85

DPO_BASE="outputs/sft_policy_v1/merged_model"
DPO_LORA="outputs/dpo_policy_v1/final_model"
QWEN_BASE="Qwen/Qwen2.5-7B-Instruct"

DPO_TAG="dpo_bva_full"
BASE_TAG="qwen_base_full"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

# ---------------------------------------------------------------
# [1/3] DPO model evaluation
# ---------------------------------------------------------------
DPO_DONE="outputs/flashrag_${DPO_TAG}/.done"
if [ -f "$DPO_DONE" ]; then
    echo "[1/3] DPO eval already done. Skipping."
else
    echo ""
    echo "============================================================"
    echo "[1/3] Evaluating DPO model (${NUM_Q}q, top_k=${TOP_K}, temp=${TEMP})"
    echo "  Base:  $DPO_BASE"
    echo "  LoRA:  $DPO_LORA"
    echo "============================================================"

    python scripts/eval_flashrag.py \
        --model "$DPO_BASE" \
        --lora "$DPO_LORA" \
        --tag "$DPO_TAG" \
        --top-k "$TOP_K" \
        --limit "$NUM_Q" \
        --temperature "$TEMP" \
        --gpu-util "$GPU_UTIL"

    touch "$DPO_DONE"
    echo "[1/3] DPO eval done."
fi

# ---------------------------------------------------------------
# [2/3] Qwen base evaluation
# ---------------------------------------------------------------
BASE_DONE="outputs/flashrag_${BASE_TAG}/.done"
if [ -f "$BASE_DONE" ]; then
    echo "[2/3] Base eval already done. Skipping."
else
    echo ""
    echo "============================================================"
    echo "[2/3] Evaluating Qwen base (${NUM_Q}q, top_k=${TOP_K}, temp=${TEMP})"
    echo "  Model: $QWEN_BASE"
    echo "============================================================"

    python scripts/eval_flashrag.py \
        --model "$QWEN_BASE" \
        --tag "$BASE_TAG" \
        --top-k "$TOP_K" \
        --limit "$NUM_Q" \
        --temperature "$TEMP" \
        --gpu-util "$GPU_UTIL"

    touch "$BASE_DONE"
    echo "[2/3] Base eval done."
fi

# ---------------------------------------------------------------
# [3/3] Compare results
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "[3/3] Computing metrics (EM / Cover EM / F1)"
echo "============================================================"

DPO_TAG="$DPO_TAG" BASE_TAG="$BASE_TAG" python3 - << 'PYEOF'
import json, os, re, string, glob

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(s.split()).strip()

def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)

def check_cover_em(gold_answers, retrieval_results):
    all_text = ""
    for step_data in retrieval_results.values():
        if 'docs' in step_data:
            for doc in step_data['docs']:
                all_text += " " + doc.get('contents', '')
    all_text_norm = normalize_answer(all_text)
    return any(normalize_answer(a) in all_text_norm for a in gold_answers)

def find_latest_experiment(tag):
    base = f"outputs/flashrag_{tag}"
    dirs = sorted(glob.glob(os.path.join(base, "hotpotqa_*_experiment")))
    if not dirs:
        return None
    return os.path.join(dirs[-1], "intermediate_data.json")

def compute(path):
    data = json.load(open(path))
    total = len(data)
    em = cover = no_ret = 0
    f1_sum = 0.0
    total_steps = 0

    for item in data:
        out = item['output']
        golds = item['golden_answers']
        ret = out.get('retrieval_results', {})

        # EM
        if out.get('metric_score', {}).get('em', 0) == 1.0:
            em += 1

        # F1
        f1_sum += out.get('metric_score', {}).get('f1', 0.0)

        # Cover EM
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

dpo_path = find_latest_experiment(os.environ["DPO_TAG"])
base_path = find_latest_experiment(os.environ["BASE_TAG"])

if not dpo_path or not base_path:
    print("ERROR: Could not find experiment data!")
    if not dpo_path: print(f"  Missing: DPO ({os.environ['DPO_TAG']})")
    if not base_path: print(f"  Missing: Base ({os.environ['BASE_TAG']})")
    exit(1)

d = compute(dpo_path)
b = compute(base_path)

W = 76
print("=" * W)
print(f"RESULTS — HotpotQA full ({b['total']}q) | e5-base | top_k=3 | temp=0.8")
print("=" * W)
print(f"{'Metric':<25} {'Qwen2.5-7B Base':>18} {'DPO (SFT+LoRA)':>18} {'Delta':>12}")
print("-" * W)
print(f"{'EM':<25} {b['em_r']:>17.2f}% {d['em_r']:>17.2f}% {d['em_r']-b['em_r']:>+11.2f}%")
print(f"{'F1':<25} {b['f1']:>17.2f}% {d['f1']:>17.2f}% {d['f1']-b['f1']:>+11.2f}%")
print(f"{'Cover EM':<25} {b['cover_r']:>17.2f}% {d['cover_r']:>17.2f}% {d['cover_r']-b['cover_r']:>+11.2f}%")
print(f"{'Avg Retrieval Steps':<25} {b['avg_steps']:>18.2f} {d['avg_steps']:>18.2f} {d['avg_steps']-b['avg_steps']:>+12.2f}")
print(f"{'No Retrieval':<25} {b['no_ret']:>18d} {d['no_ret']:>18d}")
print("=" * W)

# Save to JSON
summary = {
    "dataset": "hotpotqa", "num_questions": b["total"],
    "retriever": "e5-base-v2", "top_k": 3, "temperature": 0.8,
    "base": {"em": b["em_r"], "f1": b["f1"], "cover_em": b["cover_r"], "avg_steps": b["avg_steps"]},
    "dpo":  {"em": d["em_r"], "f1": d["f1"], "cover_em": d["cover_r"], "avg_steps": d["avg_steps"]},
    "delta": {"em": d["em_r"]-b["em_r"], "f1": d["f1"]-b["f1"], "cover_em": d["cover_r"]-b["cover_r"]},
}
out_path = "outputs/flashrag_full_comparison.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out_path}")
PYEOF

echo ""
echo "[run_full_eval] Finished at $(date)"
