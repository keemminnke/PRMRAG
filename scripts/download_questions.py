#!/usr/bin/env python3
"""Download HotpotQA questions only (train, validation, test)."""

import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def save_questions(dataset, output_file: Path):
    """Save HotpotQA questions to JSONL format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc=f"Saving {output_file.name}"):
            question_data = {
                '_id': example['id'],
                'question': example['question'],
                'answer': example['answer'],
                'type': example['type'],
                'level': example['level'],
                'supporting_facts': example['supporting_facts']
            }
            f.write(json.dumps(question_data, ensure_ascii=False) + '\n')


def main():
    questions_dir = Path("data/raw/questions")
    questions_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("DOWNLOADING HOTPOTQA QUESTIONS")
    print("="*70)

    # Download all splits
    for split_name in ['train', 'validation', 'test']:
        print(f"\n[{split_name.upper()}] Downloading...")
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split_name)

        output_file = questions_dir / f"hotpotqa_{split_name}.jsonl"
        save_questions(dataset, output_file)

        print(f"  ✅ Saved {len(dataset):,} questions to {output_file}")

    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"\nQuestions saved to: {questions_dir}/")
    print(f"  - hotpotqa_train.jsonl")
    print(f"  - hotpotqa_validation.jsonl")
    print(f"  - hotpotqa_test.jsonl")
    print()


if __name__ == "__main__":
    main()
