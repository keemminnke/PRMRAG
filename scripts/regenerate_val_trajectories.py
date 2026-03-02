#!/usr/bin/env python3
"""Critic-scored eval trajectory regeneration pipeline.

Flow:
  Phase 1 – Score:   Load eval trajectories → critic model scores every step
                     → outputs/eval_base_scored.jsonl
  Phase 2 – Regen:   Filter wrong+BAD → truncate at first BAD → regenerate
                     using critic_reasoning as feedback → outputs/eval_base_regenerated.jsonl

Auto-resume: if scored/regenerated files already exist, skip processed items.

Usage:
    # Full pipeline (Phase 1 + Phase 2)
    python scripts/regenerate_val_trajectories.py \
        --input  outputs/eval_base_hotpotqa_val500.jsonl \
        --scored outputs/eval_base_scored.jsonl \
        --output outputs/eval_base_regenerated.jsonl

    # Phase 1 only (scoring)
    python scripts/regenerate_val_trajectories.py \
        --input  outputs/eval_base_hotpotqa_val500.jsonl \
        --scored outputs/eval_base_scored.jsonl \
        --score-only

    # Phase 2 only (regeneration, requires scored file)
    python scripts/regenerate_val_trajectories.py \
        --input  outputs/eval_base_hotpotqa_val500.jsonl \
        --scored outputs/eval_base_scored.jsonl \
        --output outputs/eval_base_regenerated.jsonl \
        --regen-only
"""

import sys
import os
import re
import gc
import json
import math
import signal
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

_shutdown = False


def _sig(signum, frame):
    global _shutdown
    print(f"\n  [!] {signal.Signals(signum).name}: finishing current item then exiting...")
    _shutdown = True


# ─────────────────────────────────────────────────────────────────────────────
# Critic prompt builder  (reused from score_regenerated.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_critic_prompt(tokenizer, question: str, steps: List[Dict], step_idx: int) -> str:
    system = (
        "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
        "The trajectory uses XML tags: <think> for reasoning, <search> for queries, "
        "<answer> for final answers, <documents> for retrieved passages. "
        "Analyze each step's logical soundness and evidence grounding. "
        "First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."
    )
    parts = [f"Question: {question}", ""]
    if step_idx > 0:
        parts.append("Previous Steps:")
        for j, s in enumerate(steps[:step_idx]):
            parts.append(f"## Step {j+1}")
            if s.get("think"):
                parts.append(f"<think>{s['think']}</think>")
            if s.get("search"):
                parts.append(f"<search>{s['search']}</search>")
            elif s.get("answer"):
                parts.append(f"<answer>{s['answer']}</answer>")
            if s.get("documents"):
                parts.append(f"<documents>{s['documents']}</documents>")
            parts.append("")
    cur = steps[step_idx]
    parts.append("Current Step to Evaluate:")
    parts.append(f"## Step {step_idx + 1}")
    if cur.get("think"):
        parts.append(f"<think>{cur['think']}</think>")
    if cur.get("search"):
        parts.append(f"<search>{cur['search']}</search>")
    elif cur.get("answer"):
        parts.append(f"<answer>{cur['answer']}</answer>")
    if cur.get("documents"):
        parts.append(f"<documents>{cur['documents']}</documents>")
    parts.append("")
    parts.append("Task: Evaluate the quality of the Current Step. "
                 "Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _soft_score(logprobs_list, tid_1, tid_0, binary_label):
    for i in range(len(logprobs_list) - 1, max(len(logprobs_list) - 6, -1), -1):
        if i < 0 or logprobs_list[i] is None:
            continue
        lp1 = lp0 = None
        for tid, lp in logprobs_list[i].items():
            if tid == tid_1:
                lp1 = lp.logprob
            elif tid == tid_0:
                lp0 = lp.logprob
        if lp1 is not None or lp0 is not None:
            if lp1 is not None and lp0 is not None:
                p1 = math.exp(lp1)
                p0 = math.exp(lp0)
                return p1 / (p1 + p0)
            return (1.0 - 1e-6) if lp1 is not None else 1e-6
    return 1.0 if binary_label == 1 else (0.0 if binary_label == 0 else 0.5)


def extract_reasoning(text: str) -> str:
    m = re.search(r'\[REASONING\]\s*(.*?)\s*\[/REASONING\]', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "Label:" in text:
        return text.split("Label:")[0].strip()
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Score all trajectories with critic
# ─────────────────────────────────────────────────────────────────────────────

def phase1_score(args):
    """Score every step in the eval file with the critic model."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # Load trajectories
    print("[Phase 1] Loading eval trajectories...")
    with open(args.input) as f:
        trajs = [json.loads(l) for l in f if l.strip()]
    print(f"  Total: {len(trajs)} trajectories, "
          f"{sum(len(t['steps']) for t in trajs)} steps")

    # Resume: skip already-scored trajectories
    processed_ids = set()
    if os.path.exists(args.scored):
        with open(args.scored) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        processed_ids.add(r["trajectory_id"])
                    except Exception:
                        break
        print(f"  Resume: {len(processed_ids)} already scored, skipping")
    to_score = [t for t in trajs if t["trajectory_id"] not in processed_ids]
    if not to_score:
        print("  All already scored. Skip Phase 1.")
        return
    print(f"  To score: {len(to_score)} trajectories")

    # Load critic model
    print(f"\n  Loading critic tokenizer: {args.critic_base}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.critic_base, trust_remote_code=True, cache_dir=HF_CACHE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading critic LLM (vLLM, tensor_parallel=2)...")
    llm = LLM(
        model=args.critic_base,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=args.critic_gpu,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
        tensor_parallel_size=2,
    )
    lora_request = LoRARequest("critic", 1, args.critic_model)

    tid_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
    tid_0 = tokenizer.encode("0", add_special_tokens=False)[-1]

    # Build prompts for all (traj, step) pairs
    all_prompts = []
    prompt_info = []  # (traj_idx, step_idx)
    for ti, traj in enumerate(to_score):
        for si in range(len(traj["steps"])):
            all_prompts.append(build_critic_prompt(tokenizer, traj["question"], traj["steps"], si))
            prompt_info.append((ti, si))

    total_p = len(all_prompts)
    print(f"  Total prompts: {total_p}")

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        logprobs=20,
        stop=["Label: 1", "Label: 0", "Label:1", "Label:0"],
        include_stop_str_in_output=True,
    )

    # results per traj_idx
    results = {i: {"labels": [], "scores": [], "reasonings": []} for i in range(len(to_score))}

    batch_size = args.critic_batch
    for b_start in range(0, total_p, batch_size):
        b_end = min(b_start + batch_size, total_p)
        print(f"    Batch {b_start//batch_size + 1}: {b_start+1}-{b_end}/{total_p}")
        outputs = llm.generate(all_prompts[b_start:b_end], sampling_params, lora_request=lora_request)
        for (ti, si), out in zip(prompt_info[b_start:b_end], outputs):
            text = out.outputs[0].text
            label = -1
            if "Label: 1" in text or "Label:1" in text or text.strip().endswith("1"):
                label = 1
            elif "Label: 0" in text or "Label:0" in text or text.strip().endswith("0"):
                label = 0
            score = _soft_score(out.outputs[0].logprobs, tid_1, tid_0, label) \
                if out.outputs[0].logprobs else (1.0 if label == 1 else (0.0 if label == 0 else 0.5))
            reasoning = extract_reasoning(text)
            results[ti]["labels"].append(label)
            results[ti]["scores"].append(score)
            results[ti]["reasonings"].append(reasoning)

    # Unload critic
    del llm, tokenizer, lora_request
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("  Critic unloaded")

    # Write scored trajectories
    os.makedirs(os.path.dirname(args.scored) or ".", exist_ok=True)
    good_total = bad_total = 0
    with open(args.scored, "a") as f:
        for ti, traj in enumerate(to_score):
            scored = dict(traj)
            labels = results[ti]["labels"]
            scores = results[ti]["scores"]
            reasonings = results[ti]["reasonings"]
            for i, step in enumerate(scored["steps"]):
                if i < len(labels):
                    step["critic_label"] = labels[i]
                    step["critic_score"] = scores[i]
                    step["critic_reasoning"] = reasonings[i]
            scored["critic_step_labels"] = labels
            scored["critic_step_scores"] = scores
            scored["critic_min"] = min(scores) if scores else 0.0
            # normalize field name
            if "final_answer" in scored and "predicted_answer" not in scored:
                scored["predicted_answer"] = scored["final_answer"]
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")
            good_total += labels.count(1)
            bad_total += labels.count(0)

    print(f"\n  Scored {len(to_score)} trajectories → {args.scored}")
    print(f"  GOOD steps: {good_total}, BAD steps: {bad_total}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Filter wrong+BAD → truncate → regenerate
# ─────────────────────────────────────────────────────────────────────────────

def phase2_regen(args):
    """Regenerate wrong trajectories that have at least one BAD step."""
    global _shutdown
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    # Load scored trajectories
    print(f"\n[Phase 2] Loading scored trajectories from {args.scored}...")
    with open(args.scored) as f:
        all_scored = [json.loads(l) for l in f if l.strip()]
    print(f"  Total scored: {len(all_scored)}")

    # Filter: wrong AND has BAD step
    wrong_bad = []
    for t in all_scored:
        if t.get("is_correct"):
            continue
        labels = t.get("critic_step_labels", [])
        if any(l == 0 for l in labels):
            wrong_bad.append(t)

    correct_no_bad = sum(1 for t in all_scored if t.get("is_correct"))
    wrong_all_good = sum(1 for t in all_scored
                        if not t.get("is_correct") and all(l == 1 for l in t.get("critic_step_labels", [])))
    print(f"  Correct (skip): {correct_no_bad}")
    print(f"  Wrong + all-GOOD steps (skip): {wrong_all_good}")
    print(f"  Wrong + has BAD (regenerate): {len(wrong_bad)}")

    if args.limit:
        wrong_bad = wrong_bad[:args.limit]
        print(f"  Limited to: {len(wrong_bad)}")

    # Resume
    processed_ids = set()
    correct_count = total_count = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        processed_ids.add(r.get("trajectory_id", "").replace("_regen", ""))
                        total_count += 1
                        if r.get("is_correct"):
                            correct_count += 1
                    except Exception:
                        break
        if processed_ids:
            print(f"  Resume: {len(processed_ids)} done (correct: {correct_count}/{total_count})")
    to_regen = [t for t in wrong_bad if t["trajectory_id"] not in processed_ids]
    print(f"  To regenerate: {len(to_regen)}")

    if not to_regen:
        print("  Nothing to regenerate.")
        return

    # Truncation info
    trunc_list = []
    for t in to_regen:
        labels = t.get("critic_step_labels", [])
        steps = t.get("steps", [])
        bad_idx = next((i for i, l in enumerate(labels) if l == 0), len(steps) - 1)
        good_steps = steps[:bad_idx]
        bad_step = steps[bad_idx]
        reasoning = bad_step.get("critic_reasoning", "")
        trunc_list.append({
            "trajectory_id": t["trajectory_id"],
            "question": t["question"],
            "gold_answer": t.get("gold_answer", ""),
            "good_steps": good_steps,
            "bad_step": bad_step,
            "bad_step_idx": bad_idx,
            "critic_reasoning": reasoning,
            "all_original_steps": steps,
            "is_correct": t.get("is_correct", False),
        })

    avg_good = sum(len(t["good_steps"]) for t in trunc_list) / len(trunc_list)
    zero_good = sum(1 for t in trunc_list if not t["good_steps"])
    print(f"  Avg good steps kept: {avg_good:.1f}, zero-good: {zero_good}")

    # Load corpus + retriever
    import pickle
    data_dir = Path(__file__).parent.parent / "data"
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
    cache_path = corpus_file.parent / (corpus_file.stem + "_parsed.pkl")

    print(f"\n  Loading corpus...")
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            corpus = pickle.load(f)
    else:
        corpus = []
        with open(corpus_file) as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("id", doc.get("_id"))
                title = doc.get("title", doc.get("wikipedia_title"))
                text = doc["text"]
                if isinstance(text, list):
                    text = " ".join(text)
                corpus.append({"id": doc_id, "title": title, "text": text[:1200]})
        with open(cache_path, "wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Corpus: {len(corpus):,} documents")

    from prmrag.retrieval.bge_retriever import BGERetriever
    retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=str(data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"),
        device="cpu",
        faiss_index_path=str(data_dir / "indexes" / "kilt_wikipedia_bge_m3.faiss"),
    )

    # Load policy model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from prmrag.models import load_policy_model
    print(f"\n  Loading policy model: {args.policy_model}...")
    policy_model = load_policy_model({
        "model_name": args.policy_model,
        "temperature": args.temperature,
        "gpu_memory_utilization": args.policy_gpu,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": 2,
    })

    from prmrag.regeneration.regenerator import CriticGuidedRegenerator
    from prmrag.regeneration.truncator import TruncationResult

    regenerator = CriticGuidedRegenerator(
        policy_model=policy_model,
        retriever=retriever,
        max_steps=args.max_steps,
        top_k_passages=args.top_k,
        temperature=args.temperature,
    )

    print(f"\n  Regenerating {len(trunc_list)} trajectories...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "a") as f:
        for i, info in enumerate(trunc_list):
            if _shutdown:
                print(f"\n  [!] Graceful shutdown after {total_count} trajectories.")
                break

            tr = TruncationResult(
                trajectory_id=info["trajectory_id"],
                question_id=info["trajectory_id"].rsplit("_sample_", 1)[0],
                question=info["question"],
                gold_answer=info["gold_answer"],
                good_steps=info["good_steps"],
                bad_step=info["bad_step"],
                all_original_steps=info["all_original_steps"],
                bad_step_id=info["bad_step_idx"] + 1,
                bad_step_critic_score=info["bad_step"].get("critic_score", 0.0),
                original_num_steps=len(info["all_original_steps"]),
                original_predicted_answer="",
                original_is_correct=info["is_correct"],
                critic_reasoning=info["critic_reasoning"],
            )

            print(f"  [{i+1}/{len(trunc_list)}] {tr.trajectory_id}: "
                  f"bad@step{tr.bad_step_id}, good={len(tr.good_steps)}, "
                  f"reasoning={tr.critic_reasoning[:60]}...")

            try:
                result = regenerator.regenerate_single(tr)
            except Exception as e:
                print(f"    -> ERROR: {e}")
                result = {
                    "trajectory_id": tr.trajectory_id + "_regen",
                    "question_id": tr.question_id,
                    "question": tr.question,
                    "gold_answer": tr.gold_answer,
                    "predicted_answer": "",
                    "is_correct": False,
                    "case": "error",
                    "steps": [],
                    "metadata": {"error": str(e)},
                }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

            total_count += 1
            if result.get("is_correct"):
                correct_count += 1
                print(f"    -> CORRECT: '{result.get('predicted_answer', '')}'")
            else:
                print(f"    -> wrong: '{result.get('predicted_answer', '')}' "
                      f"(gold: '{tr.gold_answer}')")

            if (i + 1) % 50 == 0:
                print(f"\n  === Progress {total_count} done | "
                      f"correct: {correct_count}/{total_count} "
                      f"({correct_count/total_count*100:.1f}%) ===\n")

    print()
    print("=" * 70)
    print("Phase 2 Complete" if not _shutdown else "Phase 2 Paused (re-run to resume)")
    print("=" * 70)
    print(f"  Regenerated: {total_count}")
    if total_count:
        print(f"  Correct: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"  Output: {args.output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Critic-score then regenerate wrong eval trajectories")

    # I/O
    p.add_argument("--input",  type=str, required=True,
                   help="eval_base_hotpotqa_val500.jsonl (no critic labels)")
    p.add_argument("--scored", type=str, required=True,
                   help="Intermediate: trajectories with critic labels added")
    p.add_argument("--output", type=str, default=None,
                   help="Final regenerated trajectories JSONL")

    # Phase control
    p.add_argument("--score-only", action="store_true",
                   help="Only run Phase 1 (scoring)")
    p.add_argument("--regen-only", action="store_true",
                   help="Only run Phase 2 (regeneration, --scored must exist)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit Phase 2 to N trajectories (for debug)")

    # Critic model
    p.add_argument("--critic-model", type=str,
                   default="outputs/critic_model_v8_2000q/final_model",
                   help="Path to critic LoRA adapter")
    p.add_argument("--critic-base", type=str,
                   default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                   help="Critic base model")
    p.add_argument("--critic-gpu", type=float, default=0.9,
                   help="GPU memory for critic vLLM")
    p.add_argument("--critic-batch", type=int, default=256,
                   help="Batch size for critic scoring")

    # Policy model
    p.add_argument("--policy-model", type=str,
                   default="outputs/sft_policy_v1/merged_model",
                   help="Policy model for regeneration")
    p.add_argument("--policy-gpu", type=float, default=0.88,
                   help="GPU memory for policy vLLM")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-model-len", type=int, default=16384)

    return p.parse_args()


def main():
    args = parse_args()

    # Map hyphenated args to underscore
    args.critic_model = args.critic_model
    args.critic_base = args.critic_base
    args.critic_gpu = args.critic_gpu
    args.critic_batch = args.critic_batch
    args.policy_model = args.policy_model
    args.policy_gpu = args.policy_gpu
    args.max_steps = args.max_steps
    args.top_k = args.top_k
    args.max_model_len = args.max_model_len

    print("=" * 70)
    print("Eval Trajectory Regeneration Pipeline")
    print("=" * 70)
    print(f"  Input:        {args.input}")
    print(f"  Scored:       {args.scored}")
    print(f"  Output:       {args.output}")
    print(f"  Critic model: {args.critic_model}")
    print(f"  Policy model: {args.policy_model}")
    print()

    if not args.regen_only:
        phase1_score(args)

    if not args.score_only:
        if args.output is None:
            print("Error: --output required for Phase 2")
            sys.exit(1)
        phase2_regen(args)


if __name__ == "__main__":
    main()
