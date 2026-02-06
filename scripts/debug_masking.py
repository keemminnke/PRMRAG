#!/usr/bin/env python3
"""Debug script to check masking and label leakage issues.

Usage:
    python scripts/debug_masking.py --data outputs/judge_labels_clean.jsonl --num-samples 5
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from prmrag.training.critic_trainer import CriticDataFormatter, DataCollatorForCompletionOnlyLM


def check_label_leakage(messages):
    """Check if user message contains label information (leakage)."""
    leakage_patterns = [
        "judge_label", "judge_reasoning",
        "Label: GOOD", "Label: BAD", "Label: 1", "Label: 0",
        "GOOD", "BAD",  # These might be too broad, check context
        "verdict", "violates", "correct step", "incorrect step"
    ]

    issues = []
    for msg in messages:
        if msg['role'] in ['system', 'user']:
            content = msg['content']
            for pattern in leakage_patterns:
                if pattern.lower() in content.lower():
                    issues.append(f"Found '{pattern}' in {msg['role']} message")

    return issues


def debug_token_sequence(tokenizer, messages, max_length=8192):
    """Debug the actual token sequence to understand the structure."""
    print("\n" + "=" * 70)
    print("DETAILED TOKEN ANALYSIS")
    print("=" * 70)

    # Format with chat template
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    print(f"\n[1] Full formatted text (first 500 chars):")
    print("-" * 50)
    print(full_text[:500])
    print("-" * 50)

    # Tokenize
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    print(f"\n[2] Total tokens: {len(tokens)}")

    # Find special tokens
    print(f"\n[3] Looking for special tokens:")

    special_tokens_to_find = {
        '<｜Assistant｜>': None,
        '<｜User｜>': None,
        '<｜begin▁of▁sentence｜>': None,
        '<｜end▁of▁sentence｜>': None,
        '<think>': None,
        '</think>': None,
    }

    vocab = tokenizer.get_vocab()
    for token_str in special_tokens_to_find:
        if token_str in vocab:
            special_tokens_to_find[token_str] = vocab[token_str]
            print(f"    {token_str}: ID = {vocab[token_str]}")
        else:
            # Try encoding
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            print(f"    {token_str}: encodes to {encoded}")
            if len(encoded) == 1:
                special_tokens_to_find[token_str] = encoded[0]

    # Find all occurrences of <｜Assistant｜>
    assistant_token_id = special_tokens_to_find.get('<｜Assistant｜>')
    if assistant_token_id:
        positions = [i for i, t in enumerate(tokens) if t == assistant_token_id]
        print(f"\n[4] <｜Assistant｜> (ID={assistant_token_id}) found at positions: {positions}")

        if positions:
            last_pos = positions[-1]
            print(f"\n[5] Tokens AFTER last <｜Assistant｜> (position {last_pos}):")
            after_tokens = tokens[last_pos+1:last_pos+50]  # Next 50 tokens
            after_text = tokenizer.decode(after_tokens)
            print(f"    Token IDs: {after_tokens[:20]}...")
            print(f"    Decoded: {after_text[:200]}...")

    # Check for <think> token
    think_token_id = special_tokens_to_find.get('<think>')
    if think_token_id:
        positions = [i for i, t in enumerate(tokens) if t == think_token_id]
        print(f"\n[6] <think> (ID={think_token_id}) found at positions: {positions}")

    # Show the boundary region
    print(f"\n[7] Looking for assistant content boundary...")

    # Find where the assistant message content should start
    # by tokenizing without assistant message
    messages_no_assistant = [m for m in messages if m['role'] != 'assistant']
    text_no_assistant = tokenizer.apply_chat_template(
        messages_no_assistant, tokenize=False, add_generation_prompt=True
    )
    tokens_no_assistant = tokenizer.encode(text_no_assistant, add_special_tokens=False)

    print(f"    Tokens without assistant response: {len(tokens_no_assistant)}")
    print(f"    Tokens with assistant response: {len(tokens)}")
    print(f"    Difference (should be assistant content): {len(tokens) - len(tokens_no_assistant)}")

    # The supervised region should start at len(tokens_no_assistant)
    boundary = len(tokens_no_assistant)
    supervised_tokens = tokens[boundary:]
    supervised_text = tokenizer.decode(supervised_tokens)

    print(f"\n[8] Supervised region (starting at token {boundary}):")
    print(f"    Num tokens: {len(supervised_tokens)}")
    print(f"    Text: {supervised_text[:300]}...")

    return boundary, len(supervised_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--detailed", action="store_true", help="Show detailed token analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("MASKING & LABEL LEAKAGE DIAGNOSTIC")
    print("=" * 70)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    # Load formatter
    formatter = CriticDataFormatter(tokenizer)

    # Prepare samples
    print(f"\nLoading data from: {args.data}")
    samples = formatter.prepare_training_data(Path(args.data))
    print(f"Total samples: {len(samples)}")

    # Detect response template
    print("\n" + "=" * 70)
    print("RESPONSE TEMPLATE DETECTION")
    print("=" * 70)

    test_messages = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "MARKER_START_HERE"}
    ]
    formatted = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)

    print(f"\nFull formatted text (for template detection):")
    print("-" * 50)
    print(formatted)
    print("-" * 50)

    # Find marker position
    marker_pos = formatted.find("MARKER_START_HERE")
    if marker_pos != -1:
        template_region = formatted[max(0, marker_pos-50):marker_pos]
        print(f"\nText just before assistant content:")
        print(f"  {repr(template_region)}")

    # Create collator with detected template
    # Try different templates
    possible_templates = [
        "<|im_start|>assistant\n",
        "assistant\n",
        "<｜Assistant｜>",
        "<|Assistant|>",
    ]

    working_template = None
    for template in possible_templates:
        if template in formatted:
            working_template = template
            print(f"\n✓ Found working template: {repr(template)}")
            break

    if not working_template:
        print("\n⚠ No standard template found! Using fallback detection...")
        working_template = template_region[-30:] if len(template_region) >= 30 else template_region
        print(f"  Fallback template: {repr(working_template)}")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=working_template,
        tokenizer=tokenizer,
        debug=False  # Disable internal debug for cleaner output
    )

    # Detailed token analysis for first sample
    if args.detailed or True:  # Always do detailed analysis
        sample = samples[0]
        boundary, num_supervised = debug_token_sequence(tokenizer, sample['messages'], args.max_length)
        print(f"\n✓ Correct boundary should be at token {boundary}, supervising {num_supervised} tokens")

    # Analyze samples
    print("\n" + "=" * 70)
    print(f"ANALYZING {args.num_samples} SAMPLES")
    print("=" * 70)

    total_tokens = 0
    total_supervised = 0
    leakage_count = 0

    for i in range(min(args.num_samples, len(samples))):
        sample = samples[i]
        messages = sample['messages']
        label = sample.get('label', 'UNKNOWN')

        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1} (label={label})")
        print("=" * 70)

        # 1. Check label leakage
        leakage_issues = check_label_leakage(messages)
        if leakage_issues:
            print("\n⚠️  LABEL LEAKAGE DETECTED:")
            for issue in leakage_issues:
                print(f"    - {issue}")
            leakage_count += 1
        else:
            print("\n✓ No obvious label leakage in input")

        # 2. Check masking
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )

        # Apply collator
        batch = collator([{
            'input_ids': tokenized['input_ids'][0].tolist(),
            'attention_mask': tokenized['attention_mask'][0].tolist(),
        }])

        labels = batch['labels'][0]
        num_tokens = len(labels)
        supervised_mask = labels != -100
        num_supervised = supervised_mask.sum().item()

        total_tokens += num_tokens
        total_supervised += num_supervised

        print(f"\n📊 TOKEN STATISTICS:")
        print(f"    Total tokens: {num_tokens}")
        print(f"    Supervised tokens: {num_supervised} ({100*num_supervised/num_tokens:.1f}%)")

        if num_supervised == 0:
            print("    ❌ ALL TOKENS MASKED - No learning happening!")
        elif num_supervised < 20:
            print("    ⚠️  Very few supervised tokens - likely only Label is being learned!")
        else:
            print("    ✓ Reasonable number of supervised tokens")

        # 3. Show what's being supervised
        if num_supervised > 0:
            # Get the supervised text
            supervised_ids = tokenized['input_ids'][0][supervised_mask]
            supervised_text = tokenizer.decode(supervised_ids)

            print(f"\n📝 SUPERVISED TEXT (what model learns):")
            print("-" * 50)
            print(supervised_text[:500])
            if len(supervised_text) > 500:
                print("... [truncated]")
            print("-" * 50)

        # 4. Show assistant message for comparison
        assistant_msg = [m for m in messages if m['role'] == 'assistant']
        if assistant_msg:
            print(f"\n📝 EXPECTED ASSISTANT CONTENT:")
            print("-" * 50)
            print(assistant_msg[0]['content'][:500])
            print("-" * 50)

        # 5. Show user message (to check for leakage)
        user_msg = [m for m in messages if m['role'] == 'user']
        if user_msg and i < 2:  # Only show for first 2 samples
            print(f"\n📝 USER MESSAGE (check for leakage):")
            print("-" * 50)
            print(user_msg[0]['content'][:1000])
            if len(user_msg[0]['content']) > 1000:
                print("... [truncated]")
            print("-" * 50)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal tokens analyzed: {total_tokens}")
    print(f"Total supervised tokens: {total_supervised}")
    print(f"Average supervised tokens per sample: {total_supervised / args.num_samples:.1f}")
    print(f"Samples with label leakage: {leakage_count}/{args.num_samples}")

    if total_supervised / args.num_samples < 20:
        print("\n❌ PROBLEM: Too few supervised tokens!")
        print("   This explains why loss drops so fast - model is only learning Label tokens.")
        print("   FIX: Check response_template matching.")
    elif leakage_count > 0:
        print("\n⚠️  WARNING: Label leakage detected!")
        print("   This can cause artificially low loss.")
        print("   FIX: Remove judge_label/judge_reasoning from Previous Steps.")
    else:
        print("\n✓ Masking looks correct.")
        print("   If loss is still too low, consider:")
        print("   - More aggressive regularization")
        print("   - Data augmentation")
        print("   - Check if reasoning patterns are too repetitive")


if __name__ == "__main__":
    main()
