#!/usr/bin/env python3
"""Train critic model for step-level evaluation.

Usage:
    python scripts/train_critic_model.py \
        --train-data outputs/test_judge_5q_newdata_merged_clean.jsonl \
        --output-dir outputs/critic_model_test \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --num-epochs 1

This script trains a critic model that can:
1. Generate rationale for step evaluation
2. Predict verdict (GOOD/BAD)

Following the loss function:
    L_critic = -E[log P(v_j | x, s_t, v_<j) + λ log P(r | x, s_t, v)]

Where:
    - v: Verdict (GOOD/BAD)
    - r: Rationale (reasoning)
    - x: Question
    - s_t: Current step
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prmrag.training.critic_trainer import (
    create_critic_trainer,
    CriticTrainingConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train critic model for step evaluation"
    )

    # Data
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training data (merged JSONL with judge labels)"
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Path to eval data (optional)"
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/critic_model",
        help="Output directory for checkpoints"
    )

    # Training
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )

    # LoRA
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    args = parser.parse_args()

    # Validate
    if not args.train_data.exists():
        print(f"Error: Training data not found: {args.train_data}")
        sys.exit(1)

    print("=" * 70)
    print("CRITIC MODEL TRAINING")
    print("=" * 70)
    print()
    print(f"Training data:  {args.train_data}")
    if args.eval_data:
        print(f"Eval data:      {args.eval_data}")
    print(f"Model:          {args.model_name}")
    print(f"Output dir:     {args.output_dir}")
    print()

    # Create config (unused, using create_critic_trainer instead)
    # config = CriticTrainingConfig(...)

    # Create trainer
    trainer = create_critic_trainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Prepare data
    train_dataset, eval_dataset = trainer.prepare_data(
        train_file=args.train_data,
        eval_file=args.eval_data,
    )

    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    print()

    # Count labels
    label_counts = {'GOOD': 0, 'BAD': 0, 'OTHER': 0}
    for sample in train_dataset:
        label = sample.get('label', 'OTHER')
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts['OTHER'] += 1

    print(f"Total samples: {len(train_dataset)}")
    print(f"Label distribution:")
    print(f"  - GOOD: {label_counts['GOOD']} ({100*label_counts['GOOD']/len(train_dataset):.1f}%)")
    print(f"  - BAD: {label_counts['BAD']} ({100*label_counts['BAD']/len(train_dataset):.1f}%)")
    if label_counts['OTHER'] > 0:
        print(f"  - OTHER: {label_counts['OTHER']}")
    print()

    # Show detailed sample (full content, not truncated)
    print("=" * 70)
    print("SAMPLE TRAINING EXAMPLE (FULL)")
    print("=" * 70)
    sample_messages = train_dataset[0]['messages']
    for msg in sample_messages:
        print(f"\n[{msg['role'].upper()}]:")
        print("-" * 50)
        print(msg['content'])
        print("-" * 50)

    # Show second sample if available (different label if possible)
    if len(train_dataset) > 1:
        # Find a sample with different label
        first_label = train_dataset[0].get('label')
        second_idx = 1
        for i, sample in enumerate(train_dataset):
            if sample.get('label') != first_label:
                second_idx = i
                break

        print("\n" + "=" * 70)
        print(f"SECOND SAMPLE (label={train_dataset[second_idx].get('label')})")
        print("=" * 70)
        sample_messages = train_dataset[second_idx]['messages']
        for msg in sample_messages:
            print(f"\n[{msg['role'].upper()}]:")
            print("-" * 50)
            print(msg['content'])
            print("-" * 50)

    # Tokenize and show token counts for first sample
    print("\n" + "=" * 70)
    print("TOKEN ANALYSIS")
    print("=" * 70)
    sample_text = trainer.tokenizer.apply_chat_template(
        train_dataset[0]['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    tokens = trainer.tokenizer.encode(sample_text)
    print(f"Sample 1 token count: {len(tokens)}")
    print(f"Max sequence length: {args.max_seq_length}")
    if len(tokens) > args.max_seq_length:
        print(f"WARNING: Sample exceeds max_seq_length by {len(tokens) - args.max_seq_length} tokens!")

    # Check a few more samples for token length distribution
    token_lengths = []
    for i in range(min(100, len(train_dataset))):
        text = trainer.tokenizer.apply_chat_template(
            train_dataset[i]['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        token_lengths.append(len(trainer.tokenizer.encode(text)))

    print(f"\nToken length stats (first {len(token_lengths)} samples):")
    print(f"  Min: {min(token_lengths)}")
    print(f"  Max: {max(token_lengths)}")
    print(f"  Avg: {sum(token_lengths)/len(token_lengths):.0f}")
    exceeding = sum(1 for l in token_lengths if l > args.max_seq_length)
    if exceeding > 0:
        print(f"  Exceeding max_seq_length: {exceeding} ({100*exceeding/len(token_lengths):.1f}%)")
    print()

    # Train
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print()
    print(f"Model saved to: {args.output_dir}/final_model")
    print()
    print("To use the trained model:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{args.output_dir}/final_model')")
    print()


if __name__ == "__main__":
    main()
