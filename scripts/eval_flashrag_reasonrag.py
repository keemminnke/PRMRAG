#!/usr/bin/env python3
"""Evaluate ReasonRAG model using FlashRAG SearchR1Pipeline.

Uses ReasonRAG's tokens: <query>, <reference>, <answer>
and their BEGIN_REASONING system prompt.

Usage:
    python scripts/eval_flashrag_reasonrag.py \
        --model outputs/reasonrag_merged_model \
        --tag reasonrag_t0 \
        --temperature 0
"""

import argparse
import json
import os
import sys

os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE

from flashrag.config import Config
from flashrag.utils import get_dataset, get_retriever, get_generator
from flashrag.pipeline import SearchR1Pipeline
from flashrag.prompt import PromptTemplate


# ReasonRAG BEGIN_REASONING system prompt (from their paper/code)
SYSTEM_PROMPT = """You are an assistant for question answering with access to a retrieval tool. Upon receiving a question, your task is to:

* Analyze and Decompose the Question: Break the question into smaller, manageable sub-questions to ensure all aspects are addressed.

* Evaluate Your Knowledge: Assess each sub-question or component:
  - Identify parts you can confidently answer based on your existing knowledge.
  - Pinpoint parts that require additional information or verification through retrieval tools.

* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.

* Respond Format:
  If your knowledge is sufficient to answer the question, conclude with:
  "So the answer is <answer>answer</answer>"
  If retrieval is necessary to provide a complete answer, conclude with:
  "So the next query is <query>query</query>\""""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="outputs/reasonrag_merged_model")
    p.add_argument("--lora", type=str, default=None)
    p.add_argument("--tag", type=str, default="reasonrag")

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
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-retrieval", type=int, default=10)

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
        # Data
        "data_dir": args.data_dir,
        "dataset_name": args.dataset,
        "split": ["test"],
        "test_sample_num": args.limit,
        "random_sample": False,

        # Save
        "save_dir": output_dir,
        "save_intermediate_data": True,
        "save_metric_score": True,

        # Retriever
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

        # Generator
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

        # Metrics
        "metrics": ["em", "f1"],
        "metric_setting": {},

        # Misc
        "seed": 42,
        "gpu_id": "0,1",
    }

    if args.lora:
        config_dict["generator_lora_path"] = args.lora

    config = Config(config_dict=config_dict)

    print("=" * 70)
    print("FlashRAG Evaluation — ReasonRAG Model")
    print("=" * 70)
    print(f"Model:      {model_path}")
    print(f"Retriever:  bge-base-en-v1.5 (top_k={args.top_k})")
    print(f"Dataset:    {args.dataset} ({args.limit} questions)")
    print(f"Temp:       {args.temperature}")
    print(f"Output:     {output_dir}")
    print("=" * 70)

    # Load dataset
    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test questions")

    # ReasonRAG uses <query>/<reference>/<answer> tokens
    prompt_template = PromptTemplate(
        config=config,
        system_prompt=SYSTEM_PROMPT,
        user_prompt="Question: {question}\n",
    )

    pipeline = SearchR1Pipeline(
        config=config,
        prompt_template=prompt_template,
        max_retrieval_num=args.max_retrieval,
        begin_of_query_token="<query>",
        end_of_query_token="</query>",
        begin_of_documents_token="<reference>",
        end_of_documents_token="</reference>",
        begin_of_answer_token="<answer>",
        end_of_answer_token="</answer>",
    )

    # Run
    result_dataset = pipeline.run(test_data, do_eval=True)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
