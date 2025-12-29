import json
import numpy as np
from pathlib import Path

def analyze_kilt_lengths(file_path):
    lengths = []
    print(f"Reading {file_path}...")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100000: break  # Sample first 100k docs for speed
            doc = json.loads(line)
            
            # KILT text can be string or list of strings
            text = doc['text']
            if isinstance(text, list):
                text = ' '.join(text)
            
            lengths.append(len(text))

    print(f"\nStats for {len(lengths):,} documents:")
    print(f"Mean length: {np.mean(lengths):.1f} chars")
    print(f"Median length: {np.median(lengths):.1f} chars")
    print(f"Min length: {np.min(lengths)} chars")
    print(f"Max length: {np.max(lengths)} chars")
    print(f"90th percentile: {np.percentile(lengths, 90):.1f} chars")
    print(f"99th percentile: {np.percentile(lengths, 99):.1f} chars")

if __name__ == "__main__":
    analyze_kilt_lengths("data/kilt/kilt_knowledgesource.json")
