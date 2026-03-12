#!/usr/bin/env python3
"""Evaluate with FlashRAG SequentialPipeline (Standard RAG: single retrieve-and-read).

Usage:
    python scripts/eval_flashrag_standard.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tag standard_base_t0 \
        --temperature 0
"""

import argparse
import os

os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tag", type=str, default="standard")

    # Retriever
    p.add_argument("--index-path", type=str,
                   default="data/indexes/bge_Flat_IVF4096.index")
    p.add_argument("--corpus-path", type=str,
                   default="data/kilt/kilt_corpus_flashrag.jsonl")
    p.add_argument("--top-k", type=int, default=3)

    # Data
    p.add_argument("--data-dir", type=str, default="data/flashrag")
    p.add_argument("--dataset", type=str, default="hotpotqa")
    p.add_argument("--limit", type=int, default=None)

    # Generation
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)

    # GPU
    p.add_argument("--gpu-util", type=float, default=0.75)
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


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    output_dir = f"outputs/flashrag_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

    config_dict = {
        "data_dir": args.data_dir,
        "dataset_name": args.dataset,
        "split": ["test"],
        "test_sample_num": args.limit,
        "random_sample": False,

        "save_dir": output_dir,
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
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 0.95,
        },

        "metrics": ["em", "f1"],
        "metric_setting": {},

        "seed": 42,
        "gpu_id": "0,1",
    }

    config = Config(config_dict=config_dict)

    print("=" * 70)
    print("FlashRAG Standard RAG Evaluation")
    print("=" * 70)
    print(f"Model:      {model_path}")
    print(f"Retriever:  bge-base-en-v1.5 (top_k={args.top_k})")
    print(f"Dataset:    {args.dataset}")
    print(f"Temp:       {args.temperature}")
    print(f"Output:     {output_dir}")
    print("=" * 70)

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test questions")

    prompt_template = PromptTemplate(config=config)

    pipeline = SequentialPipeline(config=config, prompt_template=prompt_template)

    result_dataset = pipeline.run(test_data, do_eval=True)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
