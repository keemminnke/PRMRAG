#!/usr/bin/env python3
"""Re-extract DPO pairs from V1 tree with α=0 (pure F1)."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_mcts_dpo import load_trees, extract_dpo_pairs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", default="outputs/mcts_dpo_5000q_tree.jsonl")
    p.add_argument("--output", default="outputs/mcts_dpo_v1tree_alpha0.jsonl")
    p.add_argument("--min-reward-diff", type=float, default=0.01)
    p.add_argument("--max-children", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=7)
    args = p.parse_args()

    print(f"Loading tree: {args.tree}")
    trees = load_trees(args.tree, args.max_children, args.max_depth)
    print(f"Loaded {len(trees)} trees")

    print(f"\nExtracting pairs with α=0, δ={args.min_reward_diff}...")
    pairs = extract_dpo_pairs(trees, args.min_reward_diff, critic_alpha=0.0)

    with open(args.output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(pairs):,} pairs → {args.output}")

if __name__ == "__main__":
    main()
