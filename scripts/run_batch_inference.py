#!/usr/bin/env python3
"""Batch inference script for PolicyModelVLLM on GH200.

Maximizes throughput by:
1. Batching questions together for vLLM
2. Incremental saving to prevent data loss
3. Progress tracking with tqdm

Usage:
    python scripts/run_batch_inference.py \
        --input data/questions.jsonl \
        --output outputs/results.jsonl \
        --batch-size 64 \
        --model Qwen/Qwen2.5-7B-Instruct
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model


@dataclass
class BatchConfig:
    """Configuration for batch inference."""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 800
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    seed: int = 42

    # Batch settings
    batch_size: int = 64

    # Stop sequences (optional)
    stop_sequences: Optional[List[str]] = None


def load_questions(input_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSON or JSONL file.

    Supports:
    - JSONL: Each line is a JSON object with 'question' field
    - JSON: List of objects or single object
    - TXT: Each line is a question

    Args:
        input_path: Path to input file

    Returns:
        List of question dicts with at least 'question' field
    """
    path = Path(input_path)
    questions = []

    if path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Ensure 'question' field exists
                    if isinstance(data, str):
                        questions.append({'question': data})
                    elif 'question' in data:
                        questions.append(data)
                    elif 'text' in data:
                        data['question'] = data['text']
                        questions.append(data)
                    else:
                        # Use first string value as question
                        for v in data.values():
                            if isinstance(v, str):
                                data['question'] = v
                                break
                        questions.append(data)

    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        questions.append({'question': item})
                    else:
                        questions.append(item)
            else:
                questions.append(data)

    elif path.suffix == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append({'question': line.strip()})
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"Loaded {len(questions)} questions from {input_path}")
    return questions


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def run_batch_inference(
    questions: List[Dict[str, Any]],
    output_path: str,
    config: BatchConfig,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run batch inference on questions.

    Args:
        questions: List of question dicts
        output_path: Path to output JSONL file
        config: Batch inference configuration
        resume: If True, skip already processed questions

    Returns:
        Summary statistics
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load already processed question IDs if resuming
    processed_ids = set()
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        qid = data.get('question_id') or data.get('_id') or data.get('id')
                        if qid:
                            processed_ids.add(qid)
                    except json.JSONDecodeError:
                        continue
        print(f"Resume mode: {len(processed_ids)} questions already processed")

    # Filter out already processed questions
    if resume:
        questions = [
            q for q in questions
            if (q.get('_id') or q.get('id') or q.get('question_id')) not in processed_ids
        ]
        print(f"Remaining questions to process: {len(questions)}")

    if not questions:
        print("No questions to process!")
        return {'total': 0, 'processed': 0}

    # Initialize model
    print(f"\nLoading model: {config.model_name}")
    model_config = {
        'model_name': config.model_name,
        'temperature': config.temperature,
        'top_p': config.top_p,
        'max_tokens': config.max_tokens,
        'gpu_memory_utilization': config.gpu_memory_utilization,
        'max_model_len': config.max_model_len,
        'tensor_parallel_size': config.tensor_parallel_size,
        'seed': config.seed,
    }
    model = load_policy_model(model_config)
    print("Model loaded successfully!")

    # Process in batches
    batches = chunk_list(questions, config.batch_size)
    total_processed = 0

    print(f"\nProcessing {len(questions)} questions in {len(batches)} batches (batch_size={config.batch_size})")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Open output file in append mode
    with open(output_file, 'a', encoding='utf-8') as f:
        for batch_idx, batch in enumerate(tqdm(batches, desc="Batches")):
            # Extract questions for this batch
            batch_questions = [q['question'] for q in batch]

            # Batch generate with vLLM
            try:
                responses = model.batch_generate_with_chat_template(
                    user_messages=batch_questions,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stop_sequences=config.stop_sequences,
                )
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                # Fall back to individual processing for this batch
                responses = []
                for q in batch_questions:
                    try:
                        resp = model.generate_with_chat_template(
                            user_message=q,
                            max_tokens=config.max_tokens,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            stop_sequences=config.stop_sequences,
                        )
                        responses.append(resp)
                    except Exception as e2:
                        print(f"  Error for question: {e2}")
                        responses.append(f"ERROR: {e2}")

            # Save results immediately (incremental)
            for q_data, response in zip(batch, responses):
                result = {
                    'question_id': q_data.get('_id') or q_data.get('id') or q_data.get('question_id'),
                    'question': q_data['question'],
                    'response': response,
                    'gold_answer': q_data.get('answer') or q_data.get('gold_answer'),
                    'batch_idx': batch_idx,
                    'timestamp': datetime.now().isoformat(),
                }

                # Add any additional fields from original data
                for key in ['level', 'type', 'supporting_facts']:
                    if key in q_data:
                        result[key] = q_data[key]

                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            # Flush to disk after each batch (crash safety)
            f.flush()
            os.fsync(f.fileno())

            total_processed += len(batch)

            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                tqdm.write(f"  Processed {total_processed}/{len(questions)} questions")

    print(f"\nDone! Processed {total_processed} questions")
    print(f"Results saved to: {output_path}")

    return {
        'total': len(questions),
        'processed': total_processed,
        'output_file': str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference with PolicyModelVLLM on GH200",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input file (JSONL, JSON, or TXT)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output JSONL file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')

    # Model settings
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p sampling')
    parser.add_argument('--max-tokens', type=int, default=800,
                        help='Maximum tokens to generate')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='GPU memory utilization for vLLM')
    parser.add_argument('--max-model-len', type=int, default=8192,
                        help='Maximum model context length')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Batch settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference')

    # Stop sequences
    parser.add_argument('--stop', type=str, nargs='*', default=None,
                        help='Stop sequences (space-separated)')

    args = parser.parse_args()

    # Create config
    config = BatchConfig(
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        batch_size=args.batch_size,
        stop_sequences=args.stop,
    )

    # Print configuration
    print("=" * 60)
    print("BATCH INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"GPU memory utilization: {config.gpu_memory_utilization}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Resume: {args.resume}")
    print("=" * 60)

    # Load questions
    questions = load_questions(args.input)

    # Run inference
    stats = run_batch_inference(
        questions=questions,
        output_path=args.output,
        config=config,
        resume=args.resume,
    )

    print(f"\nSummary: {stats}")


if __name__ == '__main__':
    main()
