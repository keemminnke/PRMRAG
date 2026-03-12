#!/usr/bin/env python3
"""Critic Feedback-Guided Regeneration (GenPRM style).

For questions where MCTS failed (F1=0), extract critic reasoning from the
failure point and inject it into the base model's prompt to guide regeneration.
The generation model is always base Qwen2.5-7B-Instruct (not DPO).

Pipeline:
  1. Load existing MCTS tree → find F1=0 questions
  2. For each: find best path → identify failure point → extract critic reasoning
  3. Build prompt with good prefix + critic feedback injection
  4. Base model generates new continuation from failure point
  5. Critic PRM scores new steps → extract DPO pairs
  6. Merge new pairs with V1 original

Usage:
    python scripts/regenerate_with_critic.py \
        --tree-cache outputs/mcts_dpo_5000q_tree.jsonl \
        --existing-dpo outputs/mcts_dpo_5000q.jsonl \
        --output outputs/mcts_dpo_5000q_regen.jsonl \
        --critic-model outputs/critic_model_v9_3000q/final_model
"""

import sys
import os
import json
import re
import gc
import math
import string
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Import shared utilities from build_mcts_dpo
from build_mcts_dpo import (
    SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT,
    normalize_answer, compute_f1, format_docs,
    parse_step, parse_critic_output,
    BGERetriever, MCTSTree,
    extract_dpo_pairs, score_and_propagate,
    save_trees,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feedback-guided prompt builder
# ─────────────────────────────────────────────────────────────────────────────


def build_regen_prompt(question, good_prefix_nodes, fail_feedback, tokenizer):
    """Build prompt with good prefix + critic feedback injection.

    Feedback is injected into the user message (not after prefix) for
    better model comprehension in chat format.

    Args:
        question: The question text
        good_prefix_nodes: List of nodes from root's children up to (not including) failure point
        fail_feedback: Critic reasoning string from the failure point
        tokenizer: For chat template formatting
    Returns:
        prompt string ready for vLLM generation
    """
    # Build user message with question + critic feedback
    user_content = f"Question: {question}"
    if fail_feedback:
        user_content += (
            "\n\n[Previous Attempt Feedback]\n"
            "A previous attempt to answer this question failed. "
            "The critic identified the following issue:\n"
            f"\"{fail_feedback[:500]}\"\n"
            "Please avoid this mistake and try a different approach."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Build prefix from good nodes (skip root)
    prefix_parts = []
    for n in good_prefix_nodes:
        if n["think"]:
            prefix_parts.append(f"<think>{n['think']}</think>")
        if n["search_query"]:
            prefix_parts.append(f"<search>{n['search_query']}</search>")
        elif n["answer_text"]:
            prefix_parts.append(f"<answer>{n['answer_text']}</answer>")
        if n.get("documents"):
            prefix_parts.append(n["documents"])

    if prefix_parts:
        prompt += "\n".join(prefix_parts)

    # Continue with think tag for new generation
    prompt += "<think>"
    return prompt


def build_critic_prompt_standalone(question, gold_answer, prev_nodes, current_node, tokenizer):
    """Build critic prompt for a standalone node (not from MCTSTree)."""
    input_parts = [f"Question: {question}", f"Gold Answer: {gold_answer}", ""]

    if prev_nodes:
        input_parts.append("Previous Steps:")
        for j, prev in enumerate(prev_nodes):
            input_parts.append(f"## Step {j+1}")
            if prev["think"]:
                input_parts.append(f"<think>{prev['think']}</think>")
            if prev["search_query"]:
                input_parts.append(f"<search>{prev['search_query']}</search>")
            elif prev["answer_text"]:
                input_parts.append(f"<answer>{prev['answer_text']}</answer>")
            if prev.get("documents"):
                doc_text = prev["documents"].strip()
                doc_text = re.sub(r"^\s*<documents>\s*", "", doc_text)
                doc_text = re.sub(r"\s*</documents>\s*$", "", doc_text)
                input_parts.append(f"<documents>{doc_text}</documents>")
            input_parts.append("")

    input_parts.append("Current Step to Evaluate:")
    input_parts.append(f"## Step {len(prev_nodes) + 1}")
    if current_node["think"]:
        input_parts.append(f"<think>{current_node['think']}</think>")
    if current_node["search_query"]:
        input_parts.append(f"<search>{current_node['search_query']}</search>")
    elif current_node["answer_text"]:
        input_parts.append(f"<answer>{current_node['answer_text']}</answer>")
    if current_node.get("documents"):
        doc_text = current_node["documents"].strip()
        doc_text = re.sub(r"^\s*<documents>\s*", "", doc_text)
        doc_text = re.sub(r"\s*</documents>\s*$", "", doc_text)
        input_parts.append(f"<documents>{doc_text}</documents>")
    input_parts.append("")
    input_parts.append("Task: Evaluate the quality of the Current Step. "
                       "Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(input_parts)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main regeneration logic
# ─────────────────────────────────────────────────────────────────────────────

def extract_regen_targets(tree_path, existing_dpo_path):
    """Extract F1=0 questions with their failure points and critic reasoning."""

    # Load existing DPO pair question IDs
    qid_pair_counts = Counter()
    with open(existing_dpo_path) as f:
        for line in f:
            item = json.loads(line)
            qid_pair_counts[item["question_id"]] += 1

    # Load tree
    tree_by_qid = defaultdict(list)
    with open(tree_path) as f:
        for line in f:
            node = json.loads(line)
            tree_by_qid[node["question_id"]].append(node)

    # Find F1=0 questions (pair 0 or pair 1-2)
    targets = []
    for qid, nodes in tree_by_qid.items():
        pair_count = qid_pair_counts.get(qid, 0)
        if pair_count > 2:
            continue  # enough pairs already

        max_f1 = max((n.get("f1_reward", 0) or 0) for n in nodes)
        if max_f1 > 0:
            continue  # not F1=0

        # Build node index
        node_by_id = {n["id"]: n for n in nodes}

        # Find best path (highest avg critic score)
        terminals = [n for n in nodes if n.get("is_terminal") or n.get("answer_text")]
        if not terminals:
            terminals = [n for n in nodes
                         if not n.get("children_ids") or len(n.get("children_ids", [])) == 0]
        if not terminals:
            continue

        best_path = None
        best_avg = -1
        for term in terminals:
            path = []
            current = term
            while current is not None:
                path.append(current)
                pid = current.get("parent_id")
                current = node_by_id.get(pid) if pid is not None else None
            path.reverse()
            avg = sum((n.get("critic_score", 0) or 0) for n in path) / len(path)
            if avg > best_avg:
                best_avg = avg
                best_path = path

        if not best_path:
            continue

        # Find failure point: first critic_score=0 node (skip root)
        fail_idx = None
        fail_feedback = None
        good_prefix = []

        for i, node in enumerate(best_path):
            if i == 0:  # root
                continue
            if (node.get("critic_score", 1) or 0) == 0 and node.get("critic_feedback"):
                fail_idx = i
                fail_feedback = node["critic_feedback"]
                break
            good_prefix.append(node)

        if not fail_feedback:
            last = best_path[-1]
            if last.get("critic_feedback"):
                fail_idx = len(best_path) - 1
                fail_feedback = last["critic_feedback"]
                good_prefix = best_path[1:-1]  # exclude root and last

        if fail_feedback:
            targets.append({
                "question_id": qid,
                "question": nodes[0]["question"],
                "gold_answer": nodes[0].get("gold_answer", ""),
                "good_prefix": good_prefix,
                "fail_feedback": fail_feedback,
                "fail_step": fail_idx,
                "total_steps": len(best_path),
            })

    return targets


def run_regeneration(targets, retriever, args):
    """Generate new trajectories using critic feedback injection."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch

    # Load policy model (base Qwen, NOT DPO)
    policy_gpu_mem = 0.50
    critic_gpu_mem = 0.35

    print(f"\n[Regen] Loading policy: {args.policy_model}")
    llm = LLM(
        model=args.policy_model,
        tensor_parallel_size=2,
        gpu_memory_utilization=policy_gpu_mem,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
        seed=42,
    )
    tokenizer = llm.get_tokenizer()

    # Load critic
    print(f"[Regen] Loading critic: {args.critic_base} + LoRA")
    critic_llm = LLM(
        model=args.critic_base,
        tensor_parallel_size=2,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=critic_gpu_mem,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
    )
    critic_tokenizer = critic_llm.get_tokenizer()
    critic_model_abs = os.path.abspath(args.critic_model)
    lora_request = LoRARequest("critic", 1, critic_model_abs)

    sampling = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=1024,
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
        n=args.n_samples,  # multiple samples per question for diversity
    )
    critic_sampling = SamplingParams(temperature=0.0, max_tokens=512)

    # Build MCTS trees for regen targets
    trees = {}
    for t in targets:
        tree = MCTSTree(
            t["question_id"], t["question"], t["gold_answer"],
            max_children=args.max_children, max_depth=args.max_depth,
        )
        # Rebuild good prefix into the tree
        parent_id = tree.root_id
        for node_data in t["good_prefix"]:
            step = {
                "think": node_data.get("think", ""),
                "search_query": node_data.get("search_query", ""),
                "answer_text": node_data.get("answer_text", ""),
            }
            child = tree.expand(parent_id, step)
            if child:
                child["documents"] = node_data.get("documents", "")
                child["critic_score"] = node_data.get("critic_score")
                child["critic_feedback"] = node_data.get("critic_feedback", "")
                parent_id = child["id"]
            else:
                break

        trees[t["question_id"]] = (tree, parent_id, t["fail_feedback"])

    print(f"[Regen] {len(trees)} trees with good prefixes rebuilt")

    # Track frontier: nodes to expand next (starts at failure points)
    frontier = {}  # qid -> list of node_ids to expand
    for qid, (tree, expand_from, feedback) in trees.items():
        frontier[qid] = [expand_from]

    # Multi-round generation: each round expands frontier nodes,
    # search nodes become next round's frontier
    for round_num in range(args.max_rounds):
        print(f"\n[Regen] Round {round_num + 1}/{args.max_rounds}")

        # Build prompts for frontier nodes
        prompts = []
        prompt_meta = []  # (qid, parent_id)

        for qid, (tree, _, feedback) in trees.items():
            expand_nodes = frontier.get(qid, [])
            for expand_from in expand_nodes:
                node = tree.nodes[expand_from]
                if node["is_terminal"] or len(node["children_ids"]) >= tree.max_children:
                    continue

                # Build prompt: path to this node + critic feedback
                prefix_node_ids = tree.get_path(expand_from)[1:]  # skip root
                prefix_nodes = [tree.nodes[nid] for nid in prefix_node_ids]

                prompt = build_regen_prompt(
                    tree.question,
                    prefix_nodes,
                    feedback,
                    tokenizer,
                )
                prompts.append(prompt)
                prompt_meta.append((qid, expand_from))

        if not prompts:
            print(f"  No expandable trees, stopping")
            break

        print(f"  Generating for {len(prompts)} questions (n={args.n_samples})...")
        outputs = llm.generate(prompts, sampling)

        # Parse outputs and expand trees
        search_needed = []
        new_nodes = []

        for (qid, parent_id), output in zip(prompt_meta, outputs):
            tree = trees[qid][0]
            for sample in output.outputs:
                step = parse_step(sample.text)
                if not step["think"] and not step["search_query"] and not step["answer_text"]:
                    continue

                child = tree.expand(parent_id, step)
                if child is None:
                    continue

                if child["is_terminal"] and child["answer_text"]:
                    f1 = compute_f1(child["answer_text"], tree.gold_answer)
                    child["f1_reward"] = f1 * (args.beta ** child["depth"])
                    tree.backprop(child["id"], child["f1_reward"])
                elif child["search_query"]:
                    search_needed.append((qid, child["id"]))
                    new_nodes.append((qid, child["id"]))
                    tree.backprop(child["id"], 0.0)
                else:
                    tree.backprop(child["id"], 0.0)

        # Retrieve for search nodes
        if search_needed:
            queries = [trees[qid][0].nodes[nid]["search_query"]
                       for qid, nid in search_needed]
            docs_list = retriever.search(queries, top_k=args.top_k)
            for (qid, nid), docs in zip(search_needed, docs_list):
                trees[qid][0].nodes[nid]["documents"] = format_docs(docs)

        # Critic scoring for new non-terminal nodes
        if new_nodes:
            c_prompts = []
            for qid, nid in new_nodes:
                tree = trees[qid][0]
                node = tree.nodes[nid]
                prev_path = tree.get_path(nid)[1:-1]  # skip root, skip current
                prev_nodes = [tree.nodes[pid] for pid in prev_path]
                cp = build_critic_prompt_standalone(
                    tree.question, tree.gold_answer,
                    prev_nodes, node, critic_tokenizer,
                )
                c_prompts.append(cp)

            c_outputs = critic_llm.generate(
                c_prompts, critic_sampling, lora_request=lora_request
            )
            for (qid, nid), cout in zip(new_nodes, c_outputs):
                score, feedback = parse_critic_output(cout.outputs[0].text)
                tree = trees[qid][0]
                node = tree.nodes[nid]
                node["critic_score"] = score
                node["critic_feedback"] = feedback
                discounted = float(score) * (args.beta ** node["depth"])
                tree.retroactive_update(nid, discounted)

        # Update frontier: search nodes (non-terminal) become next round's expansion points
        next_frontier = defaultdict(list)
        for qid, nid in new_nodes:  # new_nodes = search nodes from this round
            next_frontier[qid].append(nid)
        frontier = next_frontier

        # Stats
        total_nodes = sum(len(t[0].nodes) for t in trees.values())
        terminals = sum(1 for t in trees.values()
                        for n in t[0].nodes.values() if n["is_terminal"])
        f1_pos = sum(1 for t in trees.values()
                     for n in t[0].nodes.values()
                     if n.get("f1_reward") and n["f1_reward"] > 0)
        print(f"  Nodes: {total_nodes}, Terminals: {terminals}, F1>0: {f1_pos}")
        print(f"  Next frontier: {sum(len(v) for v in frontier.values())} nodes")

    # Cleanup
    del llm, critic_llm
    gc.collect()
    torch.cuda.empty_cache()

    # Extract just the trees (drop metadata)
    return {qid: t[0] for qid, t in trees.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Critic Feedback-Guided Regeneration")
    p.add_argument("--tree-cache", type=str, required=True,
                   help="Existing MCTS tree JSONL")
    p.add_argument("--existing-dpo", type=str, required=True,
                   help="Existing DPO dataset (V1)")
    p.add_argument("--output", type=str, required=True,
                   help="Output: merged DPO dataset (V1 + regen)")

    # Models
    p.add_argument("--policy-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--critic-model", type=str, required=True,
                   help="Critic LoRA adapter path")
    p.add_argument("--critic-base", type=str,
                   default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

    # Retriever
    p.add_argument("--retriever-model", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--index-path", type=str,
                   default="data/indexes/bge_Flat.index")
    p.add_argument("--corpus-path", type=str,
                   default="data/kilt/kilt_corpus_flashrag.jsonl")
    p.add_argument("--top-k", type=int, default=3)

    # Regeneration
    p.add_argument("--max-rounds", type=int, default=5,
                   help="Max generation rounds per question")
    p.add_argument("--n-samples", type=int, default=4,
                   help="Samples per generation (for diversity)")
    p.add_argument("--max-children", type=int, default=4,
                   help="Max children per node (higher than MCTS default for regen diversity)")
    p.add_argument("--max-depth", type=int, default=7)

    # Reward
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--min-reward-diff", type=float, default=0.01)
    p.add_argument("--critic-alpha", type=float, default=0.3)

    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Critic Feedback-Guided Regeneration (GenPRM style)")
    print("=" * 70)

    # Step 1: Extract regeneration targets
    print("\n[Step 1] Extracting F1=0 questions with critic feedback...")
    targets = extract_regen_targets(args.tree_cache, args.existing_dpo)
    if args.limit:
        targets = targets[:args.limit]
    print(f"  Targets: {len(targets)} questions")

    fail_step_dist = Counter(t["fail_step"] for t in targets)
    print(f"  Failure step distribution:")
    for k in sorted(fail_step_dist):
        print(f"    step {k}: {fail_step_dist[k]}")

    # Step 2: Load retriever
    print("\n[Step 2] Loading retriever...")
    retriever = BGERetriever(
        model_path=args.retriever_model,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
    )

    # Step 3: Run regeneration
    print("\n[Step 3] Running critic-guided regeneration...")
    regen_trees = run_regeneration(targets, retriever, args)

    del retriever
    gc.collect()

    # Step 4: Score and extract pairs
    print("\n[Step 4] Scoring and extracting DPO pairs...")
    score_and_propagate(regen_trees, args.beta)
    regen_pairs = extract_dpo_pairs(
        regen_trees, args.min_reward_diff, critic_alpha=args.critic_alpha
    )

    # Step 5: Merge with existing V1
    print(f"\n[Step 5] Merging with existing DPO dataset...")
    existing_pairs = []
    with open(args.existing_dpo) as f:
        for line in f:
            existing_pairs.append(json.loads(line))

    merged = existing_pairs + regen_pairs
    print(f"  Existing: {len(existing_pairs):,}")
    print(f"  New (regen): {len(regen_pairs):,}")
    print(f"  Merged: {len(merged):,}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for p in merged:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n[Done] Saved {len(merged):,} pairs → {args.output}")

    # Save regen trees
    regen_tree_path = args.output.replace(".jsonl", "_regen_tree.jsonl")
    save_trees(regen_trees, regen_tree_path)

    # Stats
    regen_qids = set(p["question_id"] for p in regen_pairs)
    print(f"\n[Stats] Questions with new pairs: {len(regen_qids)}")
    if regen_pairs:
        f1_pos = sum(1 for p in regen_pairs if p["chosen_f1"] > 0)
        print(f"  Pairs with F1>0 chosen: {f1_pos}/{len(regen_pairs)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
