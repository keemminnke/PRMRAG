#!/usr/bin/env python3
"""Evaluate policy model on HotpotQA validation set.

Usage:
    # Evaluate DPO model with E5 retriever (greedy)
    python scripts/eval_policy.py \
        --data data/raw/questions/hotpotqa_validation.jsonl \
        --output outputs/eval_dpo_bva_e5_hotpotqa_val500.jsonl \
        --policy-model outputs/sft_policy_v1/merged_model \
        --lora-adapter outputs/dpo_policy_v1/final_model \
        --retriever e5 \
        --limit 500

    # Evaluate SFT baseline with E5 retriever
    python scripts/eval_policy.py \
        --data data/raw/questions/hotpotqa_validation.jsonl \
        --output outputs/eval_sft_v1_e5_hotpotqa_val500.jsonl \
        --policy-model outputs/sft_policy_v1/merged_model \
        --retriever e5 \
        --limit 500
"""

import sys
import os
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HF_CACHE_DIR = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate policy model on HotpotQA")

    parser.add_argument("--data", type=str,
                        default="data/raw/questions/hotpotqa_validation.jsonl",
                        help="HotpotQA validation JSONL path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")

    # Policy model
    parser.add_argument("--policy-model", type=str,
                        default="outputs/sft_policy_v1/merged_model",
                        help="Base policy model path (merged SFT or Qwen base)")
    parser.add_argument("--lora-adapter", type=str, default=None,
                        help="LoRA adapter path (e.g., outputs/dpo_policy_v1/final_model)")

    # Retriever
    parser.add_argument("--retriever", type=str, default="e5",
                        choices=["e5", "bge", "bm25"],
                        help="Retriever type (default: e5)")
    parser.add_argument("--e5-model", type=str, default="intfloat/e5-large-v2",
                        help="E5 model name (default: intfloat/e5-large-v2)")

    # Corpus
    parser.add_argument("--corpus", type=str,
                        default="data/kilt/kilt_knowledgesource.json",
                        help="KILT corpus path (loads from parsed .pkl cache if available)")
    parser.add_argument("--embedding-cache", type=str, default=None,
                        help="Path to save/load corpus embeddings (auto-set based on retriever)")
    parser.add_argument("--index-cache", type=str, default=None,
                        help="Path to save/load FAISS index (auto-set based on retriever)")
    parser.add_argument("--corpus-limit", type=int, default=None,
                        help="Limit corpus size for testing")

    # Generation
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of questions to evaluate (default: 500)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max retrieval steps per question (default: 10)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Passages to retrieve per query (default: 5)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Questions per batch (default: 50)")
    parser.add_argument("--max-tokens-per-step", type=int, default=4096,
                        help="Max tokens to generate per step (default: 4096)")

    # vLLM
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory utilization for vLLM (default: 0.85)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Max model context length (default: 16384)")

    # Misc
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output (skip already processed questions)")

    return parser.parse_args()


def load_corpus(corpus_path: str, corpus_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load KILT corpus. Uses .pkl cache if available."""
    corpus_path = Path(corpus_path)
    cache_path = corpus_path.parent / (corpus_path.stem + "_parsed.pkl")

    if cache_path.exists() and corpus_limit is None:
        print(f"Loading cached corpus from {cache_path}...")
        with open(cache_path, "rb") as f:
            corpus = pickle.load(f)
        print(f"✓ Loaded {len(corpus):,} documents from cache")
        return corpus

    print(f"Loading KILT corpus from {corpus_path}...")
    corpus = []
    with open(corpus_path, "r") as f:
        for i, line in enumerate(f):
            if corpus_limit and i >= corpus_limit:
                break
            doc = json.loads(line)
            doc_id = doc.get("id", doc.get("_id", f"doc_{i}"))
            title = doc.get("title", doc.get("wikipedia_title", ""))
            text_raw = doc.get("text", "")
            if isinstance(text_raw, list):
                text = " ".join(text_raw)[:1200]
            else:
                text = text_raw[:1200]
            corpus.append({"id": doc_id, "title": title, "text": text})
            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:,} documents loaded...")

    print(f"✓ Loaded {len(corpus):,} documents")

    if corpus_limit is None:
        with open(cache_path, "wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved corpus cache to {cache_path}")

    return corpus


def load_questions(data_path: str, limit: int) -> List[Dict[str, Any]]:
    """Load HotpotQA questions from JSONL."""
    questions = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    if limit:
        questions = questions[:limit]
    print(f"Loaded {len(questions)} questions from {data_path}")
    return questions


def get_default_cache_paths(retriever: str, e5_model: str) -> tuple[str, str]:
    data_dir = Path(__file__).parent.parent / "data"
    if retriever == "e5":
        model_tag = e5_model.split("/")[-1].lower().replace("-", "_")
        emb = str(data_dir / "embeddings" / f"kilt_wikipedia_{model_tag}.npy")
        idx = str(data_dir / "indexes" / f"kilt_wikipedia_{model_tag}.faiss")
    elif retriever == "bge":
        emb = str(data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy")
        idx = str(data_dir / "indexes" / "kilt_wikipedia_bge_m3.faiss")
    else:
        emb = None
        idx = None
    return emb, idx


def main():
    args = parse_args()

    print("=" * 70)
    print("POLICY EVALUATION")
    print("=" * 70)
    print(f"Data:          {args.data}")
    print(f"Output:        {args.output}")
    print(f"Policy model:  {args.policy_model}")
    if args.lora_adapter:
        print(f"LoRA adapter:  {args.lora_adapter}")
    print(f"Retriever:     {args.retriever.upper()}")
    print(f"Limit:         {args.limit} questions")
    print(f"Temperature:   {args.temperature}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load questions
    # ---------------------------------------------------------------
    questions = load_questions(args.data, args.limit)

    # Resume: skip already processed questions
    processed_ids = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        tid = d.get("trajectory_id", "")
                        qid = tid.rsplit("_sample_", 1)[0] if "_sample_" in tid else tid
                        processed_ids.add(qid)
                    except json.JSONDecodeError:
                        pass
        print(f"Resume: {len(processed_ids)} questions already processed")

    questions_to_run = [
        q for q in questions
        if q.get("_id", q.get("id", "")) not in processed_ids
    ] if args.resume else questions

    if not questions_to_run:
        print("All questions already processed!")
        return

    print(f"Questions to evaluate: {len(questions_to_run)}")

    # ---------------------------------------------------------------
    # 2. Load corpus & initialize retriever
    # ---------------------------------------------------------------
    print(f"\n[1/3] Loading KILT corpus...")
    corpus = load_corpus(args.corpus, args.corpus_limit)

    emb_cache, idx_cache = get_default_cache_paths(args.retriever, args.e5_model)
    if args.embedding_cache:
        emb_cache = args.embedding_cache
    if args.index_cache:
        idx_cache = args.index_cache

    print(f"\n[2/3] Initializing {args.retriever.upper()} retriever...")
    if args.retriever == "e5":
        from prmrag.retrieval.e5_retriever import E5Retriever
        retriever = E5Retriever(
            corpus=corpus,
            model_name=args.e5_model,
            embedding_cache_path=emb_cache,
            faiss_index_path=idx_cache,
        )
    elif args.retriever == "bge":
        from prmrag.retrieval.bge_retriever import BGERetriever
        retriever = BGERetriever(
            corpus=corpus,
            embedding_cache_path=emb_cache,
            faiss_index_path=idx_cache,
        )
    else:  # bm25
        from prmrag.retrieval import BM25Retriever
        retriever = BM25Retriever(corpus)
    print(f"✓ Retriever ready")

    # ---------------------------------------------------------------
    # 3. Load policy model
    # ---------------------------------------------------------------
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[3/3] Loading policy model...")
    from prmrag.models import load_policy_model
    policy_config = {
        "model_name": args.policy_model,
        "temperature": args.temperature,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    if args.lora_adapter:
        policy_config["lora_adapter"] = args.lora_adapter
    policy_model = load_policy_model(policy_config)
    print("✓ Policy model loaded")

    from prmrag.generation.adaptive_generator import SimpleTrajectoryGenerator
    generator = SimpleTrajectoryGenerator(
        policy_model,
        retriever,
        {
            "max_steps": args.max_steps,
            "top_k_passages": args.top_k,
            "temperature": args.temperature,
            "max_tokens_per_step": args.max_tokens_per_step,
        },
    )

    # ---------------------------------------------------------------
    # 4. Generate & evaluate
    # ---------------------------------------------------------------
    print(f"\nGenerating trajectories...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.resume else "w"

    total_generated = 0
    total_correct = 0

    with open(output_path, file_mode, encoding="utf-8") as f:
        num_batches = (len(questions_to_run) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(questions_to_run))
            batch = questions_to_run[start:end]

            print(f"\n--- Batch {batch_idx + 1}/{num_batches} (q {start+1}~{end}) ---")

            trajectories = generator.generate_batch(
                batch,
                num_samples=1,
                show_progress=True,
            )

            for traj in trajectories:
                import re
                steps_fmt = []
                for step in traj.steps:
                    sd = {
                        "step_id": step.step_id,
                        "step_type": step.metadata.get("action", "unknown"),
                    }
                    content = step.text or ""
                    for tag in ["think", "search", "answer", "documents"]:
                        m = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
                        if m:
                            sd[tag] = m.group(1).strip()
                    steps_fmt.append(sd)

                record = {
                    "trajectory_id": traj.trajectory_id,
                    "question": traj.question,
                    "gold_answer": traj.gold_answer,
                    "final_answer": traj.final_answer,
                    "is_correct": traj.is_correct,
                    "metadata": traj.metadata,
                    "steps": steps_fmt,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if traj.is_correct:
                    total_correct += 1
                total_generated += 1

            f.flush()
            os.fsync(f.fileno())
            acc = total_correct / total_generated * 100 if total_generated else 0
            print(f"  Saved {len(trajectories)} | Total: {total_generated} | Acc: {acc:.2f}%")

    # ---------------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    acc = total_correct / total_generated * 100 if total_generated else 0
    print(f"Questions:  {total_generated}")
    print(f"Correct:    {total_correct}")
    print(f"Accuracy:   {acc:.2f}%")
    print(f"Output:     {output_path}")

    # Save summary JSON
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "data": args.data,
            "policy_model": args.policy_model,
            "lora_adapter": args.lora_adapter,
            "retriever": args.retriever,
            "limit": args.limit,
            "total": total_generated,
            "correct": total_correct,
            "accuracy": acc,
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Summary:    {summary_path}")


if __name__ == "__main__":
    main()
