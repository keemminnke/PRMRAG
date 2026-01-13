#!/usr/bin/env python3
"""
Category-based analyzer for PRMRAG results JSONL.

Categories:
  A: False-positive suspects
     - is_correct == True and final MC == 0.0

  B: Bad trajectories (exclude from training)
     - B1: num_steps >= 10
     - B2: any step rpe > 10.0

  C: Step-1 unfair bad label
     - first step action == 'Reason' and label == 'bad'

Features:
  - Overall statistics summary
  - Search by question id
  - Per-category listing with optional detailed dump
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_results(path: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return results


def final_mc(result: Dict[str, Any]) -> float:
    steps = result.get("steps") or []
    if not steps:
        return 0.0
    return float(steps[-1].get("mc_after", 0.0))


def steps_count(result: Dict[str, Any]) -> int:
    if "num_steps" in result and result["num_steps"] is not None:
        try:
            return int(result["num_steps"])
        except Exception:
            pass
    return len(result.get("steps") or [])


def max_rpe(result: Dict[str, Any]) -> float:
    steps = result.get("steps") or []
    rpes = []
    for s in steps:
        try:
            rpes.append(float(s.get("rpe", 0.0)))
        except Exception:
            rpes.append(0.0)
    return max(rpes) if rpes else 0.0


def category_flags(result: Dict[str, Any], mc_eps: float = 1e-9) -> Dict[str, bool]:
    is_corr = bool(result.get("is_correct"))
    fm = final_mc(result)
    n_steps = steps_count(result)
    mrpe = max_rpe(result)

    # A: correct but MC stuck at zero
    cat_a = is_corr and abs(fm - 0.0) <= mc_eps

    # B1/B2
    cat_b1 = n_steps >= 10
    cat_b2 = mrpe > 10.0
    cat_b = cat_b1 or cat_b2

    # C: step1 Reason but labeled bad
    steps = result.get("steps") or []
    cat_c = False
    if steps:
        s1 = steps[0]
        action = str(s1.get("action", "")).lower()
        label = str(s1.get("label", "")).lower()
        cat_c = action == "reason" and label == "bad"

    return {
        "A": cat_a,
        "B1": cat_b1,
        "B2": cat_b2,
        "B": cat_b,
        "C": cat_c,
    }


def print_overall_stats(results: List[Dict[str, Any]]):
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    with_rag = sum(1 for r in results if r.get("has_rag"))

    flags_list = [category_flags(r) for r in results]
    counts = {k: sum(1 for f in flags_list if f[k]) for k in ["A", "B1", "B2", "B", "C"]}

    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total trajectories: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)" if total else "Correct: 0")
    print(f"With RAG: {with_rag} ({with_rag/total*100:.1f}%)" if total else "With RAG: 0")
    print("-" * 70)
    for k in ["A", "B1", "B2", "B", "C"]:
        c = counts[k]
        pct = (c / total * 100) if total else 0.0
        print(f"Category {k}: {c} ({pct:.1f}%)")
    print("=" * 70)
    print()


def short_summary(result: Dict[str, Any]) -> str:
    qid = result.get("question_id", "<no-id>")
    fm = final_mc(result)
    n_steps = steps_count(result)
    mrpe = max_rpe(result)
    corr = "✓" if result.get("is_correct") else "✗"
    q_short = str(result.get("question", "")).replace("\n", " ")
    if len(q_short) > 80:
        q_short = q_short[:77] + "..."
    return f"[{corr}] MC={fm:.3f} steps={n_steps} maxRPE={mrpe:.2f} | {qid} | {q_short}"


def print_detailed(result: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(f"QUESTION ID: {result.get('question_id')}")
    print("=" * 70)
    print(f"Question: {result.get('question')}")
    print(f"Gold answer: {result.get('gold_answer')}")
    print(f"Predicted: {result.get('predicted_answer')}")
    print(f"Correct: {result.get('is_correct')}")
    print(f"Final MC: {final_mc(result):.3f}")
    print(f"Steps: {steps_count(result)} (CoT: {result.get('num_cot_steps')}, RAG: {result.get('num_rag_steps')})")
    print(f"Max RPE: {max_rpe(result):.3f}")

    print("\nSTEPS:")
    steps = result.get("steps") or []
    for s in steps:
        step_num = s.get("step_num")
        action = s.get("action")
        label = s.get("label")
        rpe = s.get("rpe")
        mc_b = s.get("mc_before")
        mc_a = s.get("mc_after")
        print(f"\nStep {step_num}: action={action} label={label} RPE={rpe} MC={mc_b}→{mc_a}")
        content = str(s.get("content", "")).rstrip()
        if content:
            for line in content.splitlines():
                if line.strip():
                    print(f"  {line}")


def filter_by_category(
    results: Iterable[Dict[str, Any]], category: str
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results:
        flags = category_flags(r)
        if flags.get(category, False):
            out.append(r)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze results JSONL by categories A/B/C")
    parser.add_argument("results_file", type=str, help="Path to results JSONL file")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print overall statistics summary",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["A", "B1", "B2", "B", "C"],
        help="List trajectories in a category",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show full details for listed trajectories",
    )
    parser.add_argument(
        "--qid",
        type=str,
        help="Show detailed view for a specific question id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of listed trajectories (default: 50)",
    )
    args = parser.parse_args(argv)

    path = Path(args.results_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    results = load_results(path)

    if args.summary or (not args.category and not args.qid):
        print_overall_stats(results)

    if args.qid:
        matches = [r for r in results if r.get("question_id") == args.qid]
        if not matches:
            print(f"Question ID not found: {args.qid}", file=sys.stderr)
            return 1
        for r in matches:
            print_detailed(r)

    if args.category:
        cats = filter_by_category(results, args.category)
        print("=" * 70)
        print(f"CATEGORY {args.category} ({len(cats)} trajectories)")
        print("=" * 70)

        if not cats:
            print("No trajectories found.\n")
            return 0

        if args.limit and len(cats) > args.limit:
            cats = cats[: args.limit]
            print(f"Showing first {args.limit} trajectories.\n")

        for r in cats:
            if args.detail:
                print_detailed(r)
            else:
                print(short_summary(r))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

