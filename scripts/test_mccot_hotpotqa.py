#!/usr/bin/env python3
"""Test MC-CoT with real HotpotQA data."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config


def load_hotpotqa_questions(data_path, num_questions=10):
    """Load questions from HotpotQA dataset.

    Args:
        data_path: Path to hotpotqa_train.jsonl
        num_questions: Number of questions to load

    Returns:
        List of question dicts with 'id', 'question', 'answer'
    """
    questions = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_questions:
                break

            data = json.loads(line)
            questions.append({
                'id': data.get('_id', f'q{i+1}'),
                'question': data['question'],
                'answer': data['answer'],
            })

    return questions


def generate_step_by_step(policy_model, question, max_steps=3):
    """Generate step-by-step reasoning for a question.

    Args:
        policy_model: Policy model for generation
        question: Question to solve
        max_steps: Maximum number of steps

    Returns:
        dict with 'steps' and 'final_answer'
    """
    steps = []

    for step_num in range(1, max_steps + 1):
        # Build prompt
        if step_num == 1:
            user_prompt = f"""Question: {question}

Solve this step by step. You MUST respond with EXACTLY ONE step in this format:
"Step 1: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step 1.

Step 1:"""
        else:
            # Build continuation with previous steps
            prev_steps_text = "\n".join([f"Step {s['num']}: {s['content']}" for s in steps])

            user_prompt = f"""Question: {question}

{prev_steps_text}

Continue solving. You MUST respond with EXACTLY ONE step in this format:
"Step {step_num}: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step {step_num}.

Step {step_num}:"""

        # Generate
        response = policy_model.generate_with_chat_template(
            user_message=user_prompt,
            max_tokens=200,
            temperature=0.7,
            stop_sequences=None
        )

        # Post-process: Remove future steps
        import re
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        # Parse to extract content (remove "Step N:" prefix if present)
        content = response.strip()
        if content.startswith(f"Step {step_num}:"):
            content = content[len(f"Step {step_num}:"):].strip()

        steps.append({
            'num': step_num,
            'content': content,
        })

        # Check if we have an answer
        content_lower = content.lower()
        if any(marker in content_lower for marker in ['the answer is', 'therefore', 'final answer']):
            break

    # Extract final answer from last step
    last_content = steps[-1]['content'] if steps else ""
    final_answer = extract_answer(last_content)

    return {
        'steps': steps,
        'final_answer': final_answer,
    }


def extract_answer(text):
    """Extract answer from text."""
    text_lower = text.lower()

    # Pattern 1: "the answer is X"
    if "the answer is" in text_lower:
        idx = text_lower.rfind("the answer is")
        answer_part = text[idx + len("the answer is"):].strip()
        sentences = answer_part.split('.')
        return sentences[0].strip()

    # Pattern 2: "therefore, X"
    if "therefore" in text_lower:
        idx = text_lower.rfind("therefore")
        answer_part = text[idx + len("therefore"):].strip()
        if answer_part.startswith(','):
            answer_part = answer_part[1:].strip()
        sentences = answer_part.split('.')
        return sentences[0].strip()

    # Pattern 3: "final answer: X"
    if "final answer" in text_lower:
        idx = text_lower.rfind("final answer")
        answer_part = text[idx + len("final answer"):].strip()
        if answer_part.startswith(':') or answer_part.startswith('is'):
            answer_part = answer_part.lstrip(':is').strip()
        sentences = answer_part.split('.')
        return sentences[0].strip()

    # Fallback: return full text
    return text.strip()


def check_answer(predicted, gold):
    """Check if answer is correct (simple matching)."""
    pred_norm = predicted.lower().strip()
    gold_norm = gold.lower().strip()

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Contains
    if gold_norm in pred_norm:
        return True

    return False


def main():
    print("=" * 70)
    print("MC-CoT Test with Real HotpotQA Data")
    print("=" * 70)

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load model
    print("\n[1] Loading Qwen2.5-7B model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Load real HotpotQA data
    data_path = Path("/root/.local/PRMRAG/data/raw/hotpotqa_train.jsonl")
    if not data_path.exists():
        # Fallback to local path
        data_path = Path("data/raw/hotpotqa_train.jsonl")

    print(f"\n[2] Loading questions from {data_path}")
    questions = load_hotpotqa_questions(data_path, num_questions=10)
    print(f"✓ Loaded {len(questions)} questions")

    # Generate for each question
    results = []

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}/{len(questions)}: {q['id']}")
        print(f"{'='*70}")
        print(f"Q: {q['question']}")
        print(f"Gold Answer: {q['answer']}")
        print()

        # Generate step-by-step
        result = generate_step_by_step(policy_model, q['question'], max_steps=3)

        # Display steps
        print("Generated Steps:")
        for step in result['steps']:
            print(f"  Step {step['num']}: {step['content']}")

        print(f"\nFinal Answer: {result['final_answer']}")

        # Check correctness
        is_correct = check_answer(result['final_answer'], q['answer'])
        print(f"Correct: {'✓' if is_correct else '✗'}")

        results.append({
            'question_id': q['id'],
            'question': q['question'],
            'gold_answer': q['answer'],
            'predicted_answer': result['final_answer'],
            'steps': result['steps'],
            'is_correct': is_correct,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    num_correct = sum(1 for r in results if r['is_correct'])
    accuracy = num_correct / len(results) * 100

    print(f"Total Questions: {len(results)}")
    print(f"Correct: {num_correct}")
    print(f"Accuracy: {accuracy:.1f}%")

    print("\nResults by question:")
    for r in results:
        status = "✓" if r['is_correct'] else "✗"
        print(f"  {status} {r['question_id']}: {r['question'][:50]}...")


if __name__ == "__main__":
    main()
