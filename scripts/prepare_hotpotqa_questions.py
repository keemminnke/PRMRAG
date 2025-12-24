#!/usr/bin/env python3
"""Prepare HotpotQA question JSONL files under data/raw/questions.

Downloads HotpotQA from official URLs and writes files compatible with PRMRAG:

  data/raw/questions/hotpotqa_train.jsonl
  data/raw/questions/hotpotqa_validation.jsonl
"""

import argparse
import json
import requests
from pathlib import Path
from tqdm import tqdm

HOTPOT_URLS = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "validation": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
}

def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def convert_to_jsonl(input_path: Path, output_path: Path):
    """Convert HotpotQA JSON to JSONL."""
    print(f"Converting {input_path} to {output_path}...")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    with open(output_path, 'w') as f:
        for item in tqdm(data, desc="Writing JSONL"):
            # Ensure consist keys with PRMRAG expectation
            record = {
                "_id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "level": item.get("level"),
                "type": item.get("type"),
                "supporting_facts": item.get("supporting_facts", []),
                "context": item.get("context", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"✓ Converted {len(data)} examples.")

def main():
    parser = argparse.ArgumentParser(description="Download/convert HotpotQA to PRMRAG JSONL format")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/questions"),
        help="Output directory (default: data/raw/questions)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("data/raw/tmp"),
        help="Temporary directory for downloads",
    )
    args = parser.parse_args()

    for split, url in HOTPOT_URLS.items():
        json_path = args.tmp_dir / f"hotpot_{split}.json"
        jsonl_path = args.output_dir / f"hotpotqa_{split}.jsonl"
        
        if not jsonl_path.exists():
            print(f"Processing {split} split...")
            if not json_path.exists():
                download_file(url, json_path)
            convert_to_jsonl(json_path, jsonl_path)
            # Optional: remove tmp file
            # json_path.unlink()
        else:
            print(f"Skipping {split} (already exists at {jsonl_path})")

if __name__ == "__main__":
    main()
