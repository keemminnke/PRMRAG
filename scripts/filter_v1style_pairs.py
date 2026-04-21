#!/usr/bin/env python3
"""Filter K=3 pairs to V1-style distribution: hard + critic-only + no easy wins.

V1 empirical distribution:
  - 27.6% ΔF1=0 (critic-only signal)
  - 49.1% ΔF1 < 0.05 (hard)
  - 2.9% easy wins (cho_f1 > 0.9)

Target filter:
  1. Keep pairs where ΔF1 < 0.1 (hard) OR (ΔF1 = 0 AND Δcritic > τ) (critic-only useful)
  2. Drop easy wins (cho_f1 > 0.9)
  3. Critic agrees with chosen order (cho_critic >= rej_critic)
  4. Deduplicate by prompt (keep 1 per prompt)
"""
import json, argparse, random
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-delta-f1", type=float, default=0.1)
    p.add_argument("--min-delta-critic", type=float, default=0.3,
                   help="For ΔF1=0 pairs, require at least this Δcritic")
    p.add_argument("--max-chosen-f1", type=float, default=0.9)
    p.add_argument("--dedup-by-prompt", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    kept = []
    stats = {"total":0, "easy_win":0, "critic_disagree":0, "delta_too_big":0, "critic_only_weak":0, "accepted":0}

    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            stats["total"] += 1
            cf, rf = r.get('chosen_f1',0), r.get('rejected_f1',0)
            cc, rc = r.get('chosen_critic',0), r.get('rejected_critic',0)
            diff = cf - rf
            if cf > args.max_chosen_f1:
                stats["easy_win"] += 1; continue
            if cc < rc:
                stats["critic_disagree"] += 1; continue
            if abs(diff) < 0.01:
                if (cc - rc) < args.min_delta_critic:
                    stats["critic_only_weak"] += 1; continue
            elif abs(diff) > args.max_delta_f1:
                stats["delta_too_big"] += 1; continue
            kept.append(r)
            stats["accepted"] += 1

    print(f"Input: {stats['total']:,} pairs")
    print(f"  Filtered (easy win cho>0.9):       {stats['easy_win']:,}")
    print(f"  Filtered (critic disagree):        {stats['critic_disagree']:,}")
    print(f"  Filtered (ΔF1 > {args.max_delta_f1}):           {stats['delta_too_big']:,}")
    print(f"  Filtered (critic-only weak):       {stats['critic_only_weak']:,}")
    print(f"  Accepted:                          {stats['accepted']:,}")

    if args.dedup_by_prompt:
        by_prompt = defaultdict(list)
        for r in kept:
            key = json.dumps(r['prompt'], sort_keys=True)
            by_prompt[key].append(r)
        deduped = []
        for key, items in by_prompt.items():
            items.sort(key=lambda x: -(x['chosen_f1'] - x['rejected_f1'] + 0.3*(x.get('chosen_critic',0) - x.get('rejected_critic',0))))
            deduped.append(items[0])
        random.shuffle(deduped)
        print(f"\nAfter dedup by prompt: {len(deduped):,} pairs ({100*len(deduped)/len(kept):.1f}% of accepted)")
        kept = deduped

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved to: {args.output}")

    # Distribution summary
    import numpy as np
    cfs = [r['chosen_f1'] for r in kept]
    rfs = [r['rejected_f1'] for r in kept]
    diffs = [a-b for a,b in zip(cfs,rfs)]
    print(f"\nDistribution:")
    print(f"  cho_f1 mean/median: {np.mean(cfs):.3f} / {np.median(cfs):.3f}")
    print(f"  rej_f1 mean/median: {np.mean(rfs):.3f} / {np.median(rfs):.3f}")
    print(f"  ΔF1 mean/median:    {np.mean(diffs):.3f} / {np.median(diffs):.3f}")
    delta_zero = sum(1 for d in diffs if abs(d) < 0.01)
    hard = sum(1 for d in diffs if 0 < abs(d) < 0.05)
    print(f"  ΔF1 = 0 (critic-only): {delta_zero:,} ({100*delta_zero/len(kept):.1f}%)")
    print(f"  ΔF1 < 0.05 (hard):     {hard:,} ({100*hard/len(kept):.1f}%)")

if __name__ == "__main__":
    main()
