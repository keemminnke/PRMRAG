#!/usr/bin/env python3
"""Evaluate a model on ALL datasets with a single model load.

Results are saved incrementally to a JSON file after each dataset.

Usage:
    # PRO-Step (ours)
    python scripts/eval_all_datasets.py \
        --model outputs/dpo_policy_mcts_v1/merged_model \
        --tag v1 --mode searchr1

    # Search-R1 (baseline)
    python scripts/eval_all_datasets.py \
        --model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo \
        --tag searchr1 --mode searchr1 --use-default-prompt \
        --doc-begin "<information>" --doc-end "</information>"

    # Standard RAG (baseline)
    python scripts/eval_all_datasets.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tag stdrag --mode standard
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE

from flashrag.config import Config
from flashrag.utils import get_dataset, get_retriever, get_generator
from flashrag.pipeline import SequentialPipeline
from flashrag.pipeline import SearchR1Pipeline
from flashrag.prompt import PromptTemplate


SYSTEM_PROMPT = """You are a helpful assistant who is good at answering questions with multi-turn search engine calling. To answer questions, you must first reason through the available information using <think> and </think>. If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with relevant documents enclosed in <documents> and </documents>. You can search as many times as you want. Once you have sufficient information or if you find no further external knowledge is needed, directly provide your final answer. Ensure your answer is concise, using nouns or short phrases whenever possible. Conclude with: "So the answer is <answer>answer</answer>"."""

ALL_DATASETS = ["popqa", "hotpotqa", "2wikimultihopqa", "bamboogle", "musique"]

RESULTS_DIR = "outputs/eval_results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--tag", type=str, required=True, help="e.g., v1, searchr1, stdrag")
    p.add_argument("--mode", type=str, choices=["searchr1", "standard"], default="searchr1")

    # Datasets
    p.add_argument("--datasets", type=str, nargs="+", default=ALL_DATASETS)
    p.add_argument("--skip", type=str, nargs="*", default=[], help="Datasets to skip")

    # Retriever
    p.add_argument("--index-path", type=str, default="data/indexes/bge_Flat_IVF4096.index")
    p.add_argument("--corpus-path", type=str, default="data/kilt/kilt_corpus_flashrag.jsonl")
    p.add_argument("--top-k", type=int, default=3)

    # Generation
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-retrieval", type=int, default=10)

    # SearchR1 tokens
    p.add_argument("--doc-begin", type=str, default="<documents>")
    p.add_argument("--doc-end", type=str, default="</documents>")
    p.add_argument("--use-default-prompt", action="store_true")

    # GPU
    p.add_argument("--gpu-util", type=float, default=0.70)

    # Data
    p.add_argument("--data-dir", type=str, default="data/flashrag")
    return p.parse_args()


def resolve_model_path(model_name: str) -> str:
    if os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "config.json")):
        return model_name
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(model_name, cache_dir=HF_CACHE)
        print(f"Resolved {model_name} -> {local_path}")
        return local_path
    except Exception:
        return model_name


def load_results(results_file, tag):
    """Load existing results from JSON or scan FlashRAG output dirs."""
    results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

    # Also scan existing FlashRAG output dirs for completed results
    for ds in ALL_DATASETS:
        if ds in results:
            continue
        # Check outputs/flashrag_{tag}_{ds}/ or outputs/flashrag_{tag}/
        for pattern in [f"outputs/flashrag_{tag}_{ds}", f"outputs/flashrag_{tag}"]:
            if not os.path.isdir(pattern):
                continue
            for root, dirs, files in os.walk(pattern):
                if "metric_score.txt" in files and ds in root:
                    mpath = os.path.join(root, "metric_score.txt")
                    em, f1 = 0.0, 0.0
                    with open(mpath) as mf:
                        for line in mf:
                            if line.startswith("em:"):
                                em = float(line.split(":")[1].strip())
                            elif line.startswith("f1:"):
                                f1 = float(line.split(":")[1].strip())
                    results[ds] = {"em": em, "f1": f1, "time_sec": 0, "source": "resumed"}
                    print(f"[RESUME] Found existing result for {ds}: EM={em*100:.2f}%")
                    break
    return results


def save_results(results, results_file):
    """Save results incrementally."""
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


def print_results_table(results):
    """Print current results as a table."""
    print("\n" + "=" * 70)
    print(f"{'Dataset':<20} {'EM':>10} {'F1':>10} {'Time':>10}")
    print("-" * 70)
    for ds in ALL_DATASETS:
        if ds in results:
            r = results[ds]
            em = f"{r['em']*100:.2f}%"
            f1 = f"{r['f1']*100:.2f}%"
            t = f"{r.get('time_sec', 0):.0f}s"
        else:
            em = f1 = t = "-"
        print(f"{ds:<20} {em:>10} {f1:>10} {t:>10}")

    # Average
    done = [r for ds, r in results.items() if ds in ALL_DATASETS]
    if done:
        avg_em = sum(r["em"] for r in done) / len(done)
        avg_f1 = sum(r["f1"] for r in done) / len(done)
        print("-" * 70)
        print(f"{'Avg (' + str(len(done)) + '/' + str(len(ALL_DATASETS)) + ')':<20} {avg_em*100:.2f}%{' ':>4} {avg_f1*100:.2f}%")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)

    results_file = os.path.join(RESULTS_DIR, f"{args.tag}.json")
    results = load_results(results_file, args.tag)

    datasets_to_run = [ds for ds in args.datasets if ds not in args.skip]

    print("=" * 70)
    print(f"EVAL ALL DATASETS — {args.tag}")
    print("=" * 70)
    print(f"Model:      {model_path}")
    print(f"Mode:       {args.mode}")
    print(f"Datasets:   {datasets_to_run}")
    print(f"Results:    {results_file}")
    print(f"Temp:       {args.temperature}")
    print("=" * 70)

    # Print already-completed results
    if results:
        print("\nAlready completed:")
        print_results_table(results)

    # Shared config (first dataset — will be overridden per dataset)
    first_ds = datasets_to_run[0] if datasets_to_run else "hotpotqa"
    config_dict = {
        "data_dir": args.data_dir,
        "dataset_name": first_ds,
        "split": ["test"],
        "test_sample_num": None,
        "random_sample": False,

        "save_dir": f"outputs/flashrag_{args.tag}",
        "save_intermediate_data": True,
        "save_metric_score": True,

        "retrieval_method": "bge",
        "retrieval_model_path": "BAAI/bge-base-en-v1.5",
        "index_path": args.index_path,
        "corpus_path": args.corpus_path,
        "retrieval_topk": args.top_k,
        "retrieval_batch_size": 256,
        "retrieval_use_fp16": True,
        "retrieval_query_max_length": 256,
        "retrieval_pooling_method": "mean",
        "faiss_gpu": True,

        "framework": "vllm",
        "generator_model": model_path,
        "generator_model_path": model_path,
        "generator_max_input_len": 16384,
        "gpu_memory_utilization": args.gpu_util,
        "gpu_num": 2,
        "generation_params": {
            "max_tokens": args.max_tokens if args.mode == "searchr1" else 256,
            "temperature": args.temperature,
            "top_p": 0.95,
        },

        "metrics": ["em", "f1"],
        "metric_setting": {},
        "seed": 42,
        "gpu_id": "0,1",
    }

    config = Config(config_dict=config_dict)

    # Build generator & retriever ONCE
    print("\nLoading generator...")
    generator = get_generator(config)
    print("Loading retriever...")
    retriever = get_retriever(config)

    for ds in datasets_to_run:
        # Skip if already done
        if ds in results:
            print(f"\n[SKIP] {ds} — already done (EM={results[ds]['em']*100:.2f}%)")
            continue

        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating: {ds}")
        print(f"{'='*70}")

        # Update config for this dataset
        config_dict["dataset_name"] = ds
        config_dict["save_dir"] = f"outputs/flashrag_{args.tag}_{ds}"
        ds_config = Config(config_dict=config_dict)

        # Load dataset
        all_split = get_dataset(ds_config)
        test_data = all_split["test"]
        print(f"Loaded {len(test_data)} questions")

        # Build pipeline
        if args.mode == "standard":
            prompt_template = PromptTemplate(config=ds_config)
            pipeline = SequentialPipeline(
                config=ds_config,
                prompt_template=prompt_template,
                retriever=retriever,
                generator=generator,
            )
        else:  # searchr1
            if args.use_default_prompt:
                prompt_template = None
            else:
                prompt_template = PromptTemplate(
                    config=ds_config,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt="Question: {question}\n",
                )
            pipeline = SearchR1Pipeline(
                config=ds_config,
                prompt_template=prompt_template,
                max_retrieval_num=args.max_retrieval,
                begin_of_query_token="<search>",
                end_of_query_token="</search>",
                begin_of_documents_token=args.doc_begin,
                end_of_documents_token=args.doc_end,
                begin_of_answer_token="<answer>",
                end_of_answer_token="</answer>",
                retriever=retriever,
                generator=generator,
            )

        # Run
        t0 = time.time()
        result_dataset = pipeline.run(test_data, do_eval=True)
        elapsed = time.time() - t0

        # Extract metrics
        metric_file = None
        ds_save_dir = f"outputs/flashrag_{args.tag}_{ds}"
        for root, dirs, files in os.walk(ds_save_dir):
            for f in files:
                if f == "metric_score.txt":
                    metric_file = os.path.join(root, f)
                    break

        em, f1 = 0.0, 0.0
        if metric_file:
            with open(metric_file) as mf:
                for line in mf:
                    if line.startswith("em:"):
                        em = float(line.split(":")[1].strip())
                    elif line.startswith("f1:"):
                        f1 = float(line.split(":")[1].strip())

        # Save result
        results[ds] = {
            "em": em,
            "f1": f1,
            "time_sec": elapsed,
            "n_questions": len(test_data),
            "timestamp": datetime.now().isoformat(),
        }
        save_results(results, results_file)

        print(f"\n[RESULT] {ds}: EM={em*100:.2f}%, F1={f1*100:.2f}% ({elapsed:.0f}s)")
        print_results_table(results)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 70)
    print_results_table(results)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
