#!/usr/bin/env python3
"""
Run VersaPRM on all trajectories and save per-trajectory scores.
Output: outputs/versaprm_per_trajectory.jsonl
  Each line: { question, trajectory_id, predicted_answer, gold_answer,
               is_correct, versaprm_step_scores, versaprm_avg, versaprm_min }
"""

import sys, json, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── Config ────────────────────────────────────────────────────────────────
PER_TRAJ_IN  = "outputs/voting_hotpotqa_per_trajectory.jsonl"   # has question/answer/steps
ORIG_TRAJ    = "outputs/hotpotqa_val_500q_128s.jsonl"           # original (has documents)
OUTPUT_FILE  = "outputs/versaprm_per_trajectory.jsonl"
MODEL_ID     = "UW-Madison-Lee-Lab/VersaPRM-Base-8B"
CACHE_DIR    = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
BATCH_SIZE   = 16

# ─── Load VersaPRM ─────────────────────────────────────────────────────────
print(f"Loading VersaPRM from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR,
)
model.eval()
print("✓ VersaPRM loaded")

CANDIDATE_TOKENS = [12, 10]   # token IDs for "+" and "-" in VersaPRM
STEP_TOKEN_ID    = 23535      # token ID for "ки" (step separator)

# ─── Score one batch of trajectory texts ──────────────────────────────────
def score_batch(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        logits = model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).logits[:, :, CANDIDATE_TOKENS]
        scores = logits.softmax(dim=-1)[:, :, 1]   # P("+")

    batch_step_scores = []
    for j in range(len(texts)):
        input_ids  = inputs["input_ids"][j]
        item_scores = scores[j]
        step_mask  = (input_ids == STEP_TOKEN_ID)
        # Take score at each step-token position
        step_positions = step_mask.nonzero(as_tuple=True)[0]
        if len(step_positions) == 0:
            # No step tokens found – use mean of all positions
            s = [item_scores.mean().item()]
        else:
            s = [item_scores[pos].item() for pos in step_positions]
        batch_step_scores.append(s)
    return batch_step_scores


# ─── Build input text for each trajectory ─────────────────────────────────
def build_versaprm_text(question: str, steps: list) -> str:
    step_texts = []
    for i, step in enumerate(steps):
        parts = []
        if step.get("think"):
            parts.append(f"<think>{step['think'][:300]}</think>")
        if step.get("search"):
            parts.append(f"<search>{step['search']}</search>")
        elif step.get("answer"):
            parts.append(f"<answer>{step['answer']}</answer>")
        if step.get("documents"):
            parts.append(f"<documents>{step['documents'][:200]}</documents>")
        step_texts.append(f"Step {i+1}: " + " ".join(parts))

    sep = " \n\n\n\n"
    return f"Question: {question} \n\n" + sep.join(step_texts) + sep


# ─── Check answer (Cover EM) ───────────────────────────────────────────────
def check_answer(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    return gold.strip().lower() in pred.strip().lower()


# ─── Load trajectories ─────────────────────────────────────────────────────
print("Loading trajectories...")

# Use per_trajectory.jsonl as it already has step data merged with critic scores
trajs = []
with open(PER_TRAJ_IN) as f:
    for line in f:
        trajs.append(json.loads(line))

print(f"  {len(trajs)} trajectories loaded")

# ─── Resume support: skip already-processed trajectories ──────────────────
done_ids = set()
if Path(OUTPUT_FILE).exists():
    with open(OUTPUT_FILE) as f:
        for line in f:
            rec = json.loads(line)
            done_ids.add(rec.get("trajectory_id", ""))
    print(f"  Resuming: {len(done_ids)} already done, {len(trajs)-len(done_ids)} remaining")

remaining = [t for t in trajs if t.get("trajectory_id", "") not in done_ids]

# ─── Batch process ────────────────────────────────────────────────────────
out_f = open(OUTPUT_FILE, "a")

texts    = []
meta     = []

def flush_batch(texts, meta, out_f):
    if not texts:
        return
    step_scores_batch = score_batch(texts)
    for rec, step_scores in zip(meta, step_scores_batch):
        rec["versaprm_step_scores"] = step_scores
        rec["versaprm_avg"] = sum(step_scores) / len(step_scores) if step_scores else 0.0
        rec["versaprm_min"] = min(step_scores) if step_scores else 0.0
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_f.flush()

for traj in tqdm(remaining, desc="VersaPRM scoring"):
    text = build_versaprm_text(traj["question"], traj["steps"])
    texts.append(text)
    meta.append({
        "trajectory_id":   traj.get("trajectory_id", ""),
        "question":        traj["question"],
        "gold_answer":     traj["gold_answer"],
        "predicted_answer": traj["predicted_answer"],
        "is_correct":      traj["is_correct"],
        "critic_step_scores": traj.get("critic_step_scores", []),
        "critic_avg":      float(sum(traj.get("critic_step_scores",[]))/len(traj.get("critic_step_scores",[]))) if traj.get("critic_step_scores") else 0.0,
        "critic_min":      traj.get("critic_min", 0.0),
    })

    if len(texts) >= BATCH_SIZE:
        flush_batch(texts, meta, out_f)
        texts, meta = [], []

flush_batch(texts, meta, out_f)
out_f.close()
print(f"\n✓ Saved → {OUTPUT_FILE}")
print(f"  Total lines: {len(trajs)}")
