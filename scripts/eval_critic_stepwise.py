#!/usr/bin/env python3
"""Evaluate critic models step-level accuracy against judge labels (ground truth).

Computes:
  1. Step-level: Accuracy, Precision, Recall, F1 for GOOD/BAD
  2. Trajectory-level: BoN selection accuracy (pick best trajectory per question)
  3. Split by train/test: Separate metrics for seen vs unseen questions

Usage:
    # Score with v8 critic and evaluate
    python scripts/eval_critic_stepwise.py \
        --judge-labels outputs/training_data/judge_labels_5000q_merged.jsonl \
        --critic-model outputs/critic_model_v8_2000q/final_model \
        --critic-base deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --train-questions outputs/training_data/judge_labels_2000q_v1.jsonl \
        --output outputs/critic_eval_v8_on_5000q.jsonl \
        --name v8

    # Evaluate v9 from existing scored data (no inference needed)
    python scripts/eval_critic_stepwise.py \
        --judge-labels outputs/training_data/judge_labels_5000q_merged.jsonl \
        --scored-data outputs/all_5000q_critic_v9_scored.jsonl \
        --train-questions outputs/training_data/judge_labels_3000q_merged.jsonl \
        --name v9
"""

import sys
import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

CRITIC_SYSTEM_PROMPT = (
    "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
    "The trajectory uses XML tags: <think> for reasoning, <search> for queries, <answer> for final answers, "
    "<documents> for retrieved passages. "
    "Analyze each step's logical soundness and evidence grounding. "
    "First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."
)


def build_critic_prompt(question, previous_steps, current_step, tokenizer,
                        gold_answer=None, prompt_style="training"):
    """Build critic prompt in different styles for comparison.

    Styles:
      - "training": matches CriticDataFormatter (no gold answer, simple task)
      - "mcts": matches build_mcts_dpo.py (has gold answer, simple task)
      - "judge": matches JudgeLabeler (has gold answer, detailed R1-R6 criteria)
    """
    if prompt_style == "judge":
        return _build_judge_style_prompt(
            question, previous_steps, current_step, tokenizer, gold_answer
        )

    # training / mcts style (same structure, differ only in gold answer)
    input_parts = [f"Question: {question}"]
    if prompt_style == "mcts" and gold_answer:
        input_parts.append(f"Gold Answer: {gold_answer}")
    input_parts.append("")

    if previous_steps:
        input_parts.append("Previous Steps:")
        for i, prev in enumerate(previous_steps, 1):
            input_parts.append(f"## Step {i}")
            if prev.get("think"):
                input_parts.append(f"<think>{prev['think']}</think>")
            if prev.get("search"):
                input_parts.append(f"<search>{prev['search']}</search>")
            elif prev.get("answer"):
                input_parts.append(f"<answer>{prev['answer']}</answer>")
            if prev.get("documents"):
                input_parts.append(f"<documents>{prev['documents']}</documents>")
            input_parts.append("")

    input_parts.append("Current Step to Evaluate:")
    input_parts.append(f"## Step {len(previous_steps) + 1}")
    if current_step.get("think"):
        input_parts.append(f"<think>{current_step['think']}</think>")
    if current_step.get("search"):
        input_parts.append(f"<search>{current_step['search']}</search>")
    elif current_step.get("answer"):
        input_parts.append(f"<answer>{current_step['answer']}</answer>")
    if current_step.get("documents"):
        input_parts.append(f"<documents>{current_step['documents']}</documents>")
    input_parts.append("")
    input_parts.append(
        "Task: Evaluate the quality of the Current Step. "
        "Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad)."
    )

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(input_parts)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


JUDGE_SYSTEM_PROMPT = """You are a strict process supervisor for a multi-hop question answering agent that uses retrieval-augmented generation (RAG). Your task is to evaluate each step of the agent's trajectory and assign a binary label: GOOD or BAD.

The agent operates using four XML-tagged actions:
- <think>...</think>  Internal reasoning
- <search>...</search>  Retrieval query
- <documents>...</documents>  Retrieved passages (system-provided)
- <answer>...</answer>  Final answer submission

---

# Labeling Principles

1. The default label is BAD. Assign GOOD only when a step makes a clear, verifiable contribution toward answering the question.
2. No information gain means BAD. Steps that restate the question, repeat prior reasoning, or produce generic plans without new insight are BAD.
3. Do not use your own world knowledge. Any factual claim not grounded in <documents> is treated as hallucination.

---

# Evaluation Criteria

**R1. Entity and Relation Grounding**
- BAD if <think> misidentifies an entity (e.g., fictional vs. real person), reverses a relation (e.g., son vs. father), or drifts to a namesake.
- GOOD if the step maintains correct entity and relation grounding consistent with the question.

**R2. Search Steps (<search>)**
- GOOD if: (a) the query is specific and grounded in the question's entities and relations, (b) the returned <documents> contain information directly relevant to the question, and (c) if a prior search failed, the agent uses a meaningfully different strategy.
- BAD if: the query repeats a failed attempt, is too vague, targets the wrong entity, is derived from a hallucinated claim, or the returned <documents> are empty or irrelevant.

**R3. Reasoning Steps (<think> only)**
- GOOD if the step extracts new useful information from prior <documents> that advances toward the answer, or identifies a specific knowledge gap with a concrete next search plan.
- BAD if the step merely restates the question, repeats prior reasoning, makes a generic plan, summarizes without new insight, or asserts unsupported factual claims.

**R4. Answer Steps (<answer> or final step)**
This rule applies to any step with <answer> and to the last step of the trajectory.
- BAD if any of the following hold:
  (a) The answer is empty, uncertain, or a non-response (e.g., "Unknown", "N/A").
  (b) The reasoning expresses uncertainty (e.g., "cannot determine", "not sure").
  (c) The answer requires a logical leap not supported by <documents>.
  (d) The agent concludes despite insufficient evidence (premature termination).
  (e) The answer contradicts information in <documents>.
- GOOD if and only if: the answer is specific, logically derivable from the accumulated <documents>, and the reasoning chain is sound.

**R5. Recovery from Retrieval Failure**
When the preceding <documents> were irrelevant or empty:
- GOOD if the agent issues a new <search> with a meaningfully different query.
- BAD if the agent proceeds to <answer> as if the search succeeded, or retries with the same query.

**R6. Unsupported Answer (Overconfidence)**
If the agent produces <answer> without any prior successful search:
- BAD. Correct answers without evidence are penalized as lucky guesses.

---

First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."""


def _build_judge_style_prompt(question, previous_steps, current_step, tokenizer,
                              gold_answer=None):
    """Build critic prompt with exact judge system prompt + step-by-step eval."""
    input_parts = [f"**Question:** {question}"]
    if gold_answer:
        input_parts.append(
            f"**Ground Truth Answer (for reference only - do NOT use this to directly judge correctness):** {gold_answer}"
        )
    input_parts.append("")

    if previous_steps:
        input_parts.append("Previous Steps:")
        for i, prev in enumerate(previous_steps, 1):
            input_parts.append(f"## Step {i}")
            if prev.get("think"):
                input_parts.append(f"<think>{prev['think']}</think>")
            if prev.get("search"):
                input_parts.append(f"<search>{prev['search']}</search>")
            elif prev.get("answer"):
                input_parts.append(f"<answer>{prev['answer']}</answer>")
            if prev.get("documents"):
                input_parts.append(f"<documents>{prev['documents']}</documents>")
            input_parts.append("")

    input_parts.append("Current Step to Evaluate:")
    input_parts.append(f"## Step {len(previous_steps) + 1}")
    if current_step.get("think"):
        input_parts.append(f"<think>{current_step['think']}</think>")
    if current_step.get("search"):
        input_parts.append(f"<search>{current_step['search']}</search>")
    elif current_step.get("answer"):
        input_parts.append(f"<answer>{current_step['answer']}</answer>")
    if current_step.get("documents"):
        input_parts.append(f"<documents>{current_step['documents']}</documents>")
    input_parts.append("")
    input_parts.append(
        "Evaluate the quality of the Current Step using the criteria. "
        "Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad)."
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(input_parts)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_critic_output(text):
    """Parse critic output → (binary_label, score, feedback)."""
    text = text.strip()
    reasoning_match = re.search(
        r"\[REASONING\](.*?)(?:\[/REASONING\]|$)", text, re.DOTALL
    )
    feedback = reasoning_match.group(1).strip() if reasoning_match else text
    label_match = re.search(r"Label:\s*([01])", text)
    if label_match:
        label = int(label_match.group(1))
    else:
        label = 1 if "1" in text else 0
    return label, feedback


def run_critic_inference(judge_data, critic_model_path, critic_base, output_path,
                         gpu_memory=0.85, prompt_style="training"):
    """Run vLLM inference with critic model on all steps."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"\n[Inference] Loading critic: {critic_base} + LoRA: {critic_model_path}")
    print(f"[Inference] Prompt style: {prompt_style}")
    llm = LLM(
        model=critic_base,
        tensor_parallel_size=2,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=gpu_memory,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
    )
    tokenizer = llm.get_tokenizer()
    lora_request = LoRARequest("critic", 1, os.path.abspath(critic_model_path))
    sampling = SamplingParams(temperature=0.0, max_tokens=512)

    # Build all prompts
    print("[Inference] Building prompts...")
    all_prompts = []
    prompt_index = []  # (traj_idx, step_idx)

    for t_idx, traj in enumerate(tqdm(judge_data, desc="Building prompts")):
        question = traj["question"]
        gold_answer = traj.get("gold_answer", "")
        steps = traj["steps"]
        for s_idx, step in enumerate(steps):
            prev_steps = steps[:s_idx]
            prompt = build_critic_prompt(
                question, prev_steps, step, tokenizer,
                gold_answer=gold_answer, prompt_style=prompt_style,
            )
            all_prompts.append(prompt)
            prompt_index.append((t_idx, s_idx))

    print(f"[Inference] Total prompts: {len(all_prompts):,}")

    # Batch inference
    BATCH_SIZE = 5000
    all_results = [None] * len(all_prompts)

    for batch_start in range(0, len(all_prompts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]
        print(f"[Inference] Batch {batch_start//BATCH_SIZE + 1}: "
              f"prompts {batch_start+1}-{batch_end}")

        outputs = llm.generate(batch_prompts, sampling, lora_request=lora_request)

        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            label, feedback = parse_critic_output(text)
            all_results[batch_start + i] = {
                "critic_label": label,
                "critic_reasoning": feedback,
            }

    # Build reverse index for O(1) lookup
    prompt_reverse = {}
    for idx, (t_idx, s_idx) in enumerate(prompt_index):
        prompt_reverse[(t_idx, s_idx)] = idx

    # Attach results back to trajectories
    scored_data = []
    for t_idx, traj in enumerate(judge_data):
        scored_traj = {
            "trajectory_id": traj["trajectory_id"],
            "question": traj["question"],
            "gold_answer": traj["gold_answer"],
            "predicted_answer": traj.get("predicted_answer", ""),
            "is_correct": traj.get("is_correct", False),
            "steps": [],
        }
        step_labels = []
        for s_idx, step in enumerate(traj["steps"]):
            result = all_results[prompt_reverse[(t_idx, s_idx)]]
            scored_step = {**step}
            scored_step["critic_label"] = result["critic_label"]
            scored_step["critic_reasoning"] = result["critic_reasoning"]
            scored_traj["steps"].append(scored_step)
            step_labels.append(result["critic_label"])

        scored_traj["critic_step_labels"] = step_labels
        scored_traj["critic_min"] = min(step_labels) if step_labels else 0
        scored_data.append(scored_traj)

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for d in scored_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[Inference] Saved {len(scored_data):,} scored trajectories → {output_path}")

    # Cleanup
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return scored_data


def compute_metrics(judge_data, scored_data, train_qids, name):
    """Compute step-level and trajectory-level metrics."""
    # Build lookups (O(1) access)
    scored_lookup = {d["trajectory_id"]: d for d in scored_data}
    judge_lookup = {d["trajectory_id"]: d for d in judge_data}

    # Categorize trajectories
    all_tids = []
    seen_tids = []
    unseen_tids = []

    for traj in judge_data:
        tid = traj["trajectory_id"]
        if tid not in scored_lookup:
            continue
        all_tids.append(tid)
        qid = tid.rsplit("_sample_", 1)[0]
        if qid in train_qids:
            seen_tids.append(tid)
        else:
            unseen_tids.append(tid)

    def step_metrics(label, tids):
        tp = fp = tn = fn = 0
        total = 0
        for tid in tids:
            j_traj = judge_lookup[tid]
            s_traj = scored_lookup[tid]
            for js, ss in zip(j_traj["steps"], s_traj["steps"]):
                jl = 1 if str(js.get("judge_label", "")).upper() in ("GOOD", "1") else 0
                cl = ss.get("critic_label", -1)
                if cl == -1:
                    continue
                total += 1
                if jl == 1 and cl == 1: tp += 1
                elif jl == 0 and cl == 1: fp += 1
                elif jl == 0 and cl == 0: tn += 1
                elif jl == 1 and cl == 0: fn += 1

        acc = (tp + tn) / total if total else 0
        prec_g = tp / (tp + fp) if (tp + fp) else 0
        rec_g = tp / (tp + fn) if (tp + fn) else 0
        f1_g = 2 * prec_g * rec_g / (prec_g + rec_g) if (prec_g + rec_g) else 0
        prec_b = tn / (tn + fn) if (tn + fn) else 0
        rec_b = tn / (tn + fp) if (tn + fp) else 0
        f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) else 0

        n_q = len(set(t.rsplit("_sample_", 1)[0] for t in tids))
        print(f"\n{'='*60}")
        print(f"[{name}] {label} — {n_q} questions, {len(tids)} trajectories, {total} steps")
        print(f"{'='*60}")
        print(f"  Accuracy:  {acc:.4f} ({tp+tn}/{total})")
        print(f"  GOOD — P: {prec_g:.4f}  R: {rec_g:.4f}  F1: {f1_g:.4f}")
        print(f"  BAD  — P: {prec_b:.4f}  R: {rec_b:.4f}  F1: {f1_b:.4f}")
        print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"  Judge GOOD rate: {(tp+fn)/total:.3f}, Critic GOOD rate: {(tp+fp)/total:.3f}")

        return {
            "label": label, "questions": n_q, "trajectories": len(tids),
            "steps": total, "accuracy": acc,
            "good_precision": prec_g, "good_recall": rec_g, "good_f1": f1_g,
            "bad_precision": prec_b, "bad_recall": rec_b, "bad_f1": f1_b,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    results = {}
    results["all"] = step_metrics("ALL", all_tids)
    if seen_tids:
        results["seen"] = step_metrics("SEEN (train)", seen_tids)
    if unseen_tids:
        results["unseen"] = step_metrics("UNSEEN (test)", unseen_tids)

    # Trajectory-level: BoN selection
    print(f"\n{'='*60}")
    print(f"[{name}] TRAJECTORY SELECTION (BoN / Majority Voting)")
    print(f"{'='*60}")

    def trajectory_selection(label, tids):
        # Group by question
        by_q = defaultdict(list)
        for tid in tids:
            qid = tid.rsplit("_sample_", 1)[0]
            by_q[qid].append(tid)

        majority_correct = 0
        bon_min_correct = 0
        total_q = 0

        for qid, q_tids in by_q.items():
            total_q += 1
            trajs = []
            for tid in q_tids:
                j_traj = judge_lookup[tid]
                s_traj = scored_lookup[tid]
                is_correct = j_traj.get("is_correct", False)
                critic_labels = [s.get("critic_label", 0) for s in s_traj["steps"]]
                critic_min = min(critic_labels) if critic_labels else 0
                critic_avg = sum(critic_labels) / len(critic_labels) if critic_labels else 0
                trajs.append({
                    "tid": tid,
                    "is_correct": is_correct,
                    "pred": j_traj.get("predicted_answer", ""),
                    "critic_min": critic_min,
                    "critic_avg": critic_avg,
                })

            # Majority voting
            from collections import Counter
            vote_counts = Counter()
            for t in trajs:
                if t["pred"]:
                    vote_counts[t["pred"].strip().lower()] += 1
            if vote_counts:
                majority_pred = vote_counts.most_common(1)[0][0]
                gold = judge_lookup[q_tids[0]]["gold_answer"]
                majority_correct += int(majority_pred == gold.strip().lower())

            # BoN (critic_min): pick trajectory with highest min critic score
            best = max(trajs, key=lambda t: (t["critic_min"], t["critic_avg"]))
            bon_min_correct += int(best["is_correct"])

        print(f"  [{label}] {total_q} questions")
        print(f"    Majority voting: {majority_correct}/{total_q} ({100*majority_correct/total_q:.1f}%)")
        print(f"    BoN (critic min): {bon_min_correct}/{total_q} ({100*bon_min_correct/total_q:.1f}%)")

        return {
            "questions": total_q,
            "majority_correct": majority_correct,
            "majority_acc": majority_correct / total_q if total_q else 0,
            "bon_min_correct": bon_min_correct,
            "bon_min_acc": bon_min_correct / total_q if total_q else 0,
        }

    results["traj_all"] = trajectory_selection("ALL", all_tids)
    if unseen_tids:
        results["traj_unseen"] = trajectory_selection("UNSEEN", unseen_tids)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate critic model step-level accuracy")
    parser.add_argument("--judge-labels", type=str, required=True,
                        help="Judge labels JSONL (ground truth)")
    parser.add_argument("--scored-data", type=str, default=None,
                        help="Pre-scored data JSONL (skip inference if provided)")
    parser.add_argument("--critic-model", type=str, default=None,
                        help="Critic LoRA adapter path (for inference)")
    parser.add_argument("--critic-base", type=str,
                        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                        help="Critic base model")
    parser.add_argument("--train-questions", type=str, default=None,
                        help="Training data JSONL (to identify seen/unseen questions)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output scored data JSONL (for inference mode)")
    parser.add_argument("--name", type=str, default="critic",
                        help="Critic model name (for display)")
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-style", type=str, default="training",
                        choices=["training", "mcts", "judge"],
                        help="Prompt style: training (no gold), mcts (gold answer), judge (full judge prompt)")
    args = parser.parse_args()

    # Load judge labels
    print(f"[Data] Loading judge labels: {args.judge_labels}")
    judge_data = []
    with open(args.judge_labels) as f:
        for line in f:
            if line.strip():
                judge_data.append(json.loads(line))
    if args.limit:
        judge_data = judge_data[:args.limit]
    print(f"  {len(judge_data):,} trajectories")

    # Load train question IDs
    train_qids = set()
    if args.train_questions:
        with open(args.train_questions) as f:
            for line in f:
                d = json.loads(line)
                qid = d["trajectory_id"].rsplit("_sample_", 1)[0]
                train_qids.add(qid)
        print(f"  Train questions: {len(train_qids):,}")

    # Get scored data
    if args.scored_data:
        # Load pre-scored data
        print(f"[Data] Loading pre-scored data: {args.scored_data}")
        scored_data = []
        with open(args.scored_data) as f:
            for line in f:
                if line.strip():
                    scored_data.append(json.loads(line))
        if args.limit:
            scored_data = scored_data[:args.limit]
        print(f"  {len(scored_data):,} scored trajectories")
    elif args.critic_model:
        # Run inference
        scored_data = run_critic_inference(
            judge_data, args.critic_model, args.critic_base,
            args.output, args.gpu_memory, prompt_style=args.prompt_style,
        )
    else:
        print("Error: provide either --scored-data or --critic-model")
        sys.exit(1)

    # Compute metrics
    results = compute_metrics(judge_data, scored_data, train_qids, args.name)

    # Save summary
    summary_path = args.output.replace(".jsonl", "_summary.json") if args.output else f"outputs/critic_eval_{args.name}_summary.json"
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"model": args.name, "results": results}, f, indent=2)
    print(f"\n[Done] Summary → {summary_path}")


if __name__ == "__main__":
    main()
