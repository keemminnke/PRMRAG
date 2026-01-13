#!/usr/bin/env python3
"""Prepare HotpotQA question JSONL files under data/raw/questions.

Downloads HotpotQA via Hugging Face datasets and writes files compatible with PRMRAG:

  data/raw/questions/hotpotqa_train.jsonl
  data/raw/questions/hotpotqa_validation.jsonl
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset

def write_split(dataset, split: str, output_path: Path):
    """Write dataset split to JSONL."""
    print(f"Writing {split} to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Map HF split names to PRMRAG filenames
    # HF: train, validation
    
    with open(output_path, 'w') as f:
        for item in dataset[split]:
            # Keep original structure as seen in existing files
            record = {
                "_id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"✓ Wrote {len(dataset[split])} examples.")

def main():
    parser = argparse.ArgumentParser(description="Download/convert HotpotQA to PRMRAG JSONL format")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/questions"),
        help="Output directory (default: data/raw/questions)",
    )
    args = parser.parse_args()

    print("Loading hotpot_qa (distractor) from Hugging Face...")
    dataset = load_dataset("hotpot_qa", "distractor")

    # Train
    write_split(dataset, "train", args.output_dir / "hotpotqa_train.jsonl")
    
    # Validation
    write_split(dataset, "validation", args.output_dir / "hotpotqa_validation.jsonl")

    # Test (HotpotQA test set is hidden/not in public HF, usually people use validation as dev)
    # If needed, we could make a dummy test set or copy validation
    
if __name__ == "__main__":
    main()
