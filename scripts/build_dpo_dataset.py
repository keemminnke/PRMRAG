#!/usr/bin/env python3
"""Build DPO dataset with regeneration-augmented chosen pool.

Pipeline:
  Phase 1 – Regen: Load critic-scored trajectories from hotpotqa + musique.
                   Select has-BAD trajectories (prioritize questions with 0 all-GOOD).
                   Truncate at first BAD step → regenerate continuation with policy model.
                   → outputs/dpo_regen_trajectories.jsonl

  Phase 2 – Score: Run critic on regenerated trajectories to get step labels.
                   → outputs/dpo_regen_scored.jsonl

  Phase 3 – Build: Combine original all-GOOD + regenerated all-GOOD as chosen.
                   has-BAD (original) as rejected. Create all-vs-all pairs.
                   → outputs/dpo_dataset.jsonl (prompt/chosen/rejected JSONL)

Usage:
    # Full pipeline
    python scripts/build_dpo_dataset.py \
        --hotpotqa outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
        --musique  outputs/musique_critic_results_v8_per_trajectory.jsonl \
        --regen-out  outputs/dpo_regen_trajectories.jsonl \
        --scored-out outputs/dpo_regen_scored.jsonl \
        --dpo-out    outputs/dpo_dataset.jsonl \
        --policy-model outputs/sft_policy_v1/merged_model \
        --critic-gpu 0.50 --policy-gpu 0.50

    # Phase 3 only (dataset build, no regeneration)
    python scripts/build_dpo_dataset.py \
        --hotpotqa outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
        --musique  outputs/musique_critic_results_v8_per_trajectory.jsonl \
        --dpo-out  outputs/dpo_dataset.jsonl \
        --build-only

    # Phase 1+2 only (regen + score, no dataset build)
    python scripts/build_dpo_dataset.py \
        --hotpotqa outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
        --musique  outputs/musique_critic_results_v8_per_trajectory.jsonl \
        --regen-out  outputs/dpo_regen_trajectories.jsonl \
        --scored-out outputs/dpo_regen_scored.jsonl \
        --regen-only \
        --policy-model outputs/sft_policy_v1/merged_model \
        --critic-gpu 0.50 --policy-gpu 0.50
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
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

_shutdown = False


def _sig(signum, frame):
    global _shutdown
    print(f"\n  [!] {signal.Signals(signum).name}: finishing current item then exiting...")
    _shutdown = True


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_critic_results(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_question_id(trajectory_id: str) -> str:
    """Extract question_id from trajectory_id (remove _sample_N suffix)."""
    parts = trajectory_id.rsplit("_sample_", 1)
    return parts[0] if len(parts) == 2 else trajectory_id


def classify_trajectory(traj: Dict) -> str:
    """Return 'all_good', 'all_bad', or 'has_bad'."""
    labels = traj.get("critic_step_labels", [])
    if not labels:
        return "unknown"
    if all(l == 1 for l in labels):
        return "all_good"
    elif all(l == 0 for l in labels):
        return "all_bad"
    else:
        return "has_bad"


def analyze_datasets(all_trajs: List[Dict]) -> Dict:
    """Group trajectories and identify regeneration targets."""
    by_question = defaultdict(lambda: {"all_good": [], "has_bad": [], "all_bad": []})

    for t in all_trajs:
        qid = get_question_id(t["trajectory_id"])
        cat = classify_trajectory(t)
        if cat == "all_good":
            by_question[qid]["all_good"].append(t)
        elif cat in ("has_bad", "all_bad"):
            by_question[qid]["has_bad"].append(t)

    # Questions with 0 all-GOOD (need regen to have any pairs)
    no_good_qs = {qid: d for qid, d in by_question.items() if not d["all_good"]}
    has_good_qs = {qid: d for qid, d in by_question.items() if d["all_good"]}

    total_trajs = len(all_trajs)
    all_good_count = sum(len(d["all_good"]) for d in by_question.values())
    has_bad_count = sum(len(d["has_bad"]) for d in by_question.values())

    print(f"  Total trajectories: {total_trajs:,}")
    print(f"  Total questions:    {len(by_question):,}")
    print(f"  all-GOOD trajs:     {all_good_count:,} ({100*all_good_count/total_trajs:.1f}%)")
    print(f"  has-BAD trajs:      {has_bad_count:,} ({100*has_bad_count/total_trajs:.1f}%)")
    print(f"")
    print(f"  Questions with ≥1 all-GOOD: {len(has_good_qs):,}")
    print(f"  Questions with 0  all-GOOD: {len(no_good_qs):,}  ← regen priority 1")

    # DPO pair stats (without regen)
    dpo_qs = sum(1 for d in by_question.values() if d["all_good"] and d["has_bad"])
    cross_pairs = sum(
        len(d["all_good"]) * len(d["has_bad"])
        for d in by_question.values()
        if d["all_good"] and d["has_bad"]
    )
    print(f"")
    print(f"  Without regen:")
    print(f"    DPO-pairable questions: {dpo_qs:,}")
    print(f"    all-vs-all pairs:       {cross_pairs:,}")

    return dict(by_question), no_good_qs, has_good_qs


def select_regen_targets(
    by_question: Dict,
    no_good_qs: Dict,
    has_good_qs: Dict,
    max_per_q: int = 16,
    regen_with_good: bool = False,
) -> List[Dict]:
    """Select has-BAD trajectories to regenerate.

    Priority:
    1. Questions with 0 all-GOOD: regenerate up to max_per_q has-BAD
    2. (Optional) Questions with all-GOOD: regenerate if regen_with_good=True
    """
    targets = []

    # Priority 1: 0 all-GOOD questions
    for qid, d in no_good_qs.items():
        has_bad_list = d["has_bad"][:max_per_q]
        for t in has_bad_list:
            targets.append(t)

    # Priority 2: questions with all-GOOD (optional augmentation)
    if regen_with_good:
        for qid, d in has_good_qs.items():
            has_bad_list = d["has_bad"][:max_per_q]
            for t in has_bad_list:
                targets.append(t)

    print(f"  Regen targets: {len(targets):,}")
    print(f"    From 0-all-GOOD questions: "
          f"{sum(len(d['has_bad'][:max_per_q]) for d in no_good_qs.values()):,}")
    if regen_with_good:
        print(f"    From has-all-GOOD questions: "
              f"{sum(len(d['has_bad'][:max_per_q]) for d in has_good_qs.values()):,}")

    return targets


# ─────────────────────────────────────────────────────────────────────────────
# Critic helpers (reused from regenerate_val_trajectories.py)
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
# Phase 1: Regenerate has-BAD trajectories
# ─────────────────────────────────────────────────────────────────────────────

def phase1_regen(targets: List[Dict], args):
    """Regenerate has-BAD trajectories using policy model."""
    global _shutdown
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    # Resume
    processed_ids = set()
    if os.path.exists(args.regen_out):
        with open(args.regen_out) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        orig_id = r.get("trajectory_id", "").replace("_regen", "")
                        processed_ids.add(orig_id)
                    except Exception:
                        pass
        if processed_ids:
            print(f"  Resume: {len(processed_ids)} already regenerated, skipping")

    to_regen = [t for t in targets if t["trajectory_id"] not in processed_ids]
    if not to_regen:
        print("  All targets already regenerated. Skipping Phase 1.")
        return

    if args.limit:
        to_regen = to_regen[:args.limit]
    print(f"  To regenerate: {len(to_regen)}")

    # Build truncation info
    trunc_list = []
    for t in to_regen:
        labels = t.get("critic_step_labels", [])
        steps = t.get("steps", [])
        bad_idx = next((i for i, l in enumerate(labels) if l == 0), len(steps) - 1)
        bad_step = steps[bad_idx]
        reasoning = bad_step.get("critic_reasoning", "")
        trunc_list.append({
            "trajectory_id": t["trajectory_id"],
            "question": t["question"],
            "gold_answer": t.get("gold_answer", ""),
            "good_steps": steps[:bad_idx],
            "bad_step": bad_step,
            "bad_step_idx": bad_idx,
            "critic_reasoning": reasoning,
            "all_original_steps": steps,
            "is_correct": t.get("is_correct", False),
        })

    avg_good = sum(len(t["good_steps"]) for t in trunc_list) / len(trunc_list) if trunc_list else 0
    zero_good = sum(1 for t in trunc_list if not t["good_steps"])
    print(f"  Avg good steps kept before truncation: {avg_good:.1f}, zero-good: {zero_good}")

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
    print(f"  Corpus: {len(corpus):,} docs")

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
    os.makedirs(os.path.dirname(args.regen_out) or ".", exist_ok=True)

    total = correct = 0
    with open(args.regen_out, "a") as f:
        for i, info in enumerate(trunc_list):
            if _shutdown:
                print(f"\n  [!] Graceful shutdown after {total} regenerations.")
                break

            tr = TruncationResult(
                trajectory_id=info["trajectory_id"],
                question_id=get_question_id(info["trajectory_id"]),
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
                  f"bad@step{tr.bad_step_id}, good_kept={len(tr.good_steps)}, "
                  f"reasoning={tr.critic_reasoning[:50]}...")

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

            total += 1
            if result.get("is_correct"):
                correct += 1
                print(f"    -> CORRECT: '{result.get('predicted_answer', '')}'")
            else:
                print(f"    -> wrong: '{result.get('predicted_answer', '')}' "
                      f"(gold: '{tr.gold_answer}')")

            if (i + 1) % 50 == 0:
                print(f"\n  === Progress {total} | correct: {correct}/{total} "
                      f"({correct/total*100:.1f}%) ===\n")

    # Unload policy
    del policy_model, regenerator
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"\n  Regenerated: {total}, correct: {correct}/{total}")
    print(f"  Output: {args.regen_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Score regenerated trajectories with critic
# ─────────────────────────────────────────────────────────────────────────────

def phase2_score(args):
    """Score regenerated trajectories with critic to get step labels."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print(f"\n[Phase 2] Loading regenerated trajectories from {args.regen_out}...")
    with open(args.regen_out) as f:
        regen_trajs = [json.loads(l) for l in f if l.strip()]

    # Skip error/empty trajectories
    regen_trajs = [t for t in regen_trajs if t.get("steps")]
    print(f"  Total (non-empty): {len(regen_trajs)}")

    # Resume
    processed_ids = set()
    if os.path.exists(args.scored_out):
        with open(args.scored_out) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        processed_ids.add(r["trajectory_id"])
                    except Exception:
                        pass
        if processed_ids:
            print(f"  Resume: {len(processed_ids)} already scored, skipping")

    to_score = [t for t in regen_trajs if t["trajectory_id"] not in processed_ids]
    if not to_score:
        print("  All already scored. Skip Phase 2.")
        return
    print(f"  To score: {len(to_score)}")

    # Load critic
    print(f"\n  Loading critic tokenizer: {args.critic_base}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.critic_base, trust_remote_code=True, cache_dir=HF_CACHE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading critic LLM (tensor_parallel=2)...")
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

    # Build prompts
    all_prompts = []
    prompt_info = []
    for ti, traj in enumerate(to_score):
        steps = traj.get("steps", [])
        for si in range(len(steps)):
            all_prompts.append(build_critic_prompt(tokenizer, traj["question"], steps, si))
            prompt_info.append((ti, si))

    print(f"  Total prompts: {len(all_prompts)}")

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        logprobs=20,
        stop=["Label: 1", "Label: 0", "Label:1", "Label:0"],
        include_stop_str_in_output=True,
    )

    results = {i: {"labels": [], "scores": [], "reasonings": []} for i in range(len(to_score))}
    batch_size = args.critic_batch

    for b_start in range(0, len(all_prompts), batch_size):
        b_end = min(b_start + batch_size, len(all_prompts))
        print(f"    Batch {b_start//batch_size + 1}: {b_start+1}-{b_end}/{len(all_prompts)}")
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

    del llm, tokenizer, lora_request
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("  Critic unloaded")

    # Write scored
    os.makedirs(os.path.dirname(args.scored_out) or ".", exist_ok=True)
    all_good_regen = 0
    with open(args.scored_out, "a") as f:
        for ti, traj in enumerate(to_score):
            scored = dict(traj)
            labels = results[ti]["labels"]
            scores = results[ti]["scores"]
            reasonings = results[ti]["reasonings"]
            for i, step in enumerate(scored.get("steps", [])):
                if i < len(labels):
                    step["critic_label"] = labels[i]
                    step["critic_score"] = scores[i]
                    step["critic_reasoning"] = reasonings[i]
            scored["critic_step_labels"] = labels
            scored["critic_step_scores"] = scores
            scored["critic_min"] = min(scores) if scores else 0.0
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")
            if all(l == 1 for l in labels):
                all_good_regen += 1

    print(f"\n  Scored {len(to_score)} regenerated trajectories")
    print(f"  all-GOOD after regen: {all_good_regen}/{len(to_score)} "
          f"({100*all_good_regen/len(to_score):.1f}%)")
    print(f"  Output: {args.scored_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Build DPO dataset
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an advanced AI agent capable of Adaptive RAG (Retrieval-Augmented Generation).
Your goal is to answer questions accurately by combining internal reasoning with external retrieval when needed.

# OUTPUT FORMAT

Use these XML tags for your response:

1. <think>Your reasoning</think>
   - Analyze the question, plan next action, evaluate evidence
   - ALWAYS start each step with <think>

2. <search>query</search>
   - Query external knowledge base
   - Use when you need factual information

3. <answer>final answer</answer>
   - Provide final answer (entity name or short answer only)
   - Use when you have sufficient evidence

After <search>, you will receive:
<documents>Retrieved passages</documents>

# STEP TYPES

- Search step: <think>...</think> followed by <search>...</search>
- Reason step: <think>...</think> only (no search)
- Finish step: <think>...</think> followed by <answer>...</answer>

# EXAMPLE

Question: Who directed the movie that won Best Picture at the 2020 Oscars?

<think>I need to find which movie won Best Picture at the 2020 Oscars, then identify its director.</think>
<search>Best Picture winner 2020 Oscars</search>
<documents>
[1] 92nd Academy Awards: "Parasite" won Best Picture at the 92nd Academy Awards (2020)...
[2] Parasite (2019 film): Directed by Bong Joon-ho, the film also won Best Director...
</documents>
<think>The observation states "Parasite" won and was directed by Bong Joon-ho. I have sufficient evidence.</think>
<answer>Bong Joon-ho</answer>

# RULES

1. One action per step - Either <search> or <answer>, not both
2. Always <think> first - Explain your reasoning before action
3. Search before guessing - If uncertain, use <search>
4. Trust observations - Retrieved information takes priority
5. Concise answer - Output only the entity name in <answer>

Begin."""


def build_trajectory_text(steps: List[Dict]) -> str:
    """Convert steps to XML format."""
    parts = []
    for step in steps:
        think = step.get("think", "")
        if think:
            parts.append(f"<think>{think}</think>")
        search = step.get("search", "")
        answer = step.get("answer", "")
        if search:
            parts.append(f"<search>{search}</search>")
        elif answer:
            parts.append(f"<answer>{answer}</answer>")
        docs = step.get("documents", "")
        if docs:
            parts.append(f"<documents>{docs}</documents>")
    return "\n".join(parts)


def phase3_build(
    original_trajs: List[Dict],
    scored_regen_path: Optional[str],
    output_path: str,
    args,
):
    """Combine original + regenerated all-GOOD as chosen, has-BAD as rejected → all-vs-all pairs."""

    # Load regenerated scored trajectories
    regen_good = []
    if scored_regen_path and os.path.exists(scored_regen_path):
        with open(scored_regen_path) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                labels = t.get("critic_step_labels", [])
                if labels and all(l == 1 for l in labels) and t.get("steps"):
                    regen_good.append(t)
        print(f"  Loaded {len(regen_good)} regenerated all-GOOD trajectories")
    else:
        print("  No scored regenerated file. Using only original trajectories.")

    # Group original trajectories
    by_question = defaultdict(lambda: {"all_good": [], "has_bad": []})
    for t in original_trajs:
        qid = get_question_id(t["trajectory_id"])
        cat = classify_trajectory(t)
        if cat == "all_good":
            by_question[qid]["all_good"].append(t)
        elif cat in ("has_bad", "all_bad"):
            by_question[qid]["has_bad"].append(t)

    # Add regenerated all-GOOD trajectories
    regen_added_qs = set()
    for t in regen_good:
        orig_tid = t["trajectory_id"].replace("_regen", "")
        qid = get_question_id(orig_tid)
        by_question[qid]["all_good"].append(t)
        regen_added_qs.add(qid)

    print(f"  Questions with regen-added chosen: {len(regen_added_qs)}")

    # Messages format: prompt = list[dict], chosen/rejected = list[dict]
    # This avoids tokenization boundary mismatch in TRL DPOTrainer

    def build_prompt_messages(question: str) -> List[Dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]

    def build_response_messages(steps: List[Dict]) -> List[Dict]:
        return [{"role": "assistant", "content": build_trajectory_text(steps)}]

    # Create best-vs-all pairs
    pairs = []
    stats = {
        "total_qs": len(by_question),
        "dpo_qs": 0,
        "new_dpo_qs": 0,  # questions that only got pairs thanks to regen
        "total_pairs": 0,
        "regen_chosen_pairs": 0,
    }

    for qid, d in by_question.items():
        chosen_pool = d["all_good"]
        rejected_pool = d["has_bad"]

        if not chosen_pool or not rejected_pool:
            continue

        stats["dpo_qs"] += 1
        question = chosen_pool[0].get("question") or rejected_pool[0].get("question")
        prompt = build_prompt_messages(question)

        has_orig_good = any(not t["trajectory_id"].endswith("_regen") for t in chosen_pool)
        if not has_orig_good:
            stats["new_dpo_qs"] += 1

        # Best-vs-All: pick the best chosen (highest critic_min)
        best_cho = max(chosen_pool, key=lambda t: t.get("critic_min", 0.0))
        cho_msgs = build_response_messages(best_cho.get("steps", []))
        is_regen_chosen = best_cho["trajectory_id"].endswith("_regen")

        for rej in rejected_pool:
            rej_msgs = build_response_messages(rej.get("steps", []))
            pairs.append({
                "prompt": prompt,
                "chosen": cho_msgs,
                "rejected": rej_msgs,
                "question_id": qid,
                "chosen_trajectory_id": best_cho["trajectory_id"],
                "rejected_trajectory_id": rej["trajectory_id"],
                "regen_chosen": is_regen_chosen,
            })
            if is_regen_chosen:
                stats["regen_chosen_pairs"] += 1

    stats["total_pairs"] = len(pairs)

    print(f"\n  DPO Dataset Stats:")
    print(f"    Total questions:               {stats['total_qs']:,}")
    print(f"    DPO-pairable questions:        {stats['dpo_qs']:,}")
    print(f"    New pairs from regen:          {stats['new_dpo_qs']:,} questions newly added")
    print(f"    Total best-vs-all pairs:       {stats['total_pairs']:,}")
    print(f"    Pairs with regen chosen:       {stats['regen_chosen_pairs']:,}")
    print(f"    Pairs with original chosen:    {stats['total_pairs'] - stats['regen_chosen_pairs']:,}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Saved {len(pairs):,} pairs → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build DPO dataset with regeneration-augmented chosen pool")

    # Input data
    p.add_argument("--hotpotqa", type=str,
                   default="outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl")
    p.add_argument("--musique", type=str,
                   default="outputs/musique_critic_results_v8_per_trajectory.jsonl")

    # I/O
    p.add_argument("--regen-out",  type=str, default="outputs/dpo_regen_trajectories.jsonl")
    p.add_argument("--scored-out", type=str, default="outputs/dpo_regen_scored.jsonl")
    p.add_argument("--dpo-out",    type=str, default="outputs/dpo_dataset.jsonl")

    # Phase control
    p.add_argument("--build-only", action="store_true",
                   help="Skip regen + score; just build DPO dataset from existing files")
    p.add_argument("--regen-only", action="store_true",
                   help="Run Phase 1 (regen) + Phase 2 (score), skip dataset build")
    p.add_argument("--no-regen-with-good", action="store_true",
                   help="Only regen trajectories from questions with 0 all-GOOD (default)")
    p.add_argument("--regen-with-good", action="store_true",
                   help="Also regen trajectories from questions that already have all-GOOD")
    p.add_argument("--max-regen-per-q", type=int, default=16,
                   help="Max has-BAD trajectories to regenerate per question")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit regen to N trajectories (for debug)")

    # Critic model
    p.add_argument("--critic-model", type=str,
                   default="outputs/critic_model_v8_2000q/final_model")
    p.add_argument("--critic-base", type=str,
                   default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    p.add_argument("--critic-gpu", type=float, default=0.50)
    p.add_argument("--critic-batch", type=int, default=256)

    # Policy model
    p.add_argument("--policy-model", type=str,
                   default="outputs/sft_policy_v1/merged_model")
    p.add_argument("--policy-gpu", type=float, default=0.50)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-model-len", type=int, default=16384)

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("DPO Dataset Builder with Regeneration")
    print("=" * 70)

    # Load original critic-scored trajectories
    print("\n[Loading] Original critic-scored trajectories...")
    all_trajs = []
    for path in [args.hotpotqa, args.musique]:
        trajs = load_critic_results(path)
        all_trajs.extend(trajs)
        print(f"  {path}: {len(trajs):,} trajectories")
    print(f"  Total: {len(all_trajs):,} trajectories")

    print("\n[Analysis]")
    by_question, no_good_qs, has_good_qs = analyze_datasets(all_trajs)

    if args.build_only:
        # Skip regen phases
        print("\n[Build-only mode] Skipping Phase 1 and Phase 2")
        print("\n[Phase 3] Building DPO dataset...")
        phase3_build(all_trajs, args.scored_out, args.dpo_out, args)
        return

    # Phase 1: Regen
    print("\n[Phase 1] Regeneration")
    regen_with_good = args.regen_with_good and not args.no_regen_with_good
    targets = select_regen_targets(
        by_question, no_good_qs, has_good_qs,
        max_per_q=args.max_regen_per_q,
        regen_with_good=regen_with_good,
    )

    phase1_regen(targets, args)

    if _shutdown:
        print("\n[!] Shutdown requested. Re-run to resume from checkpoint.")
        return

    # Phase 2: Score
    print("\n[Phase 2] Critic-scoring regenerated trajectories")
    phase2_score(args)

    if args.regen_only:
        print("\n[Regen-only mode] Skipping Phase 3 (dataset build)")
        return

    # Phase 3: Build DPO dataset
    print("\n[Phase 3] Building DPO dataset...")
    phase3_build(all_trajs, args.scored_out, args.dpo_out, args)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"  DPO dataset: {args.dpo_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
