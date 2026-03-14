#!/usr/bin/env python3
"""Analyze PRM test-time scaling: K=1,2,...,128 performance curves.

For each K, takes first K trajectories per question and recomputes:
- Majority Voting (MV)
- Best-of-N (BoN) with avg/min aggregation
- Weighted Majority Voting (WMV) with avg/min aggregation

Reads per-trajectory JSONL files produced by compare_voting_methods.py.

Usage:
    python scripts/analyze_prm_scaling.py \
        --results-dir outputs/prm_scaling/ \
        --datasets popqa hotpotqa 2wikimultihopqa musique \
        --k-values 1 2 4 8 16 32 64 128 \
        --output outputs/prm_scaling/scaling_analysis.json
"""

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple


# ============================================================
# Methods configuration
# ============================================================

METHODS = {
    'critic_v9': {
        'score_key': 'critic_step_scores',
        'file_pattern': 'scores_{ds}_critic_v9_per_trajectory.jsonl',
        'label': 'Critic v9 (3000q)',
    },
    'critic_v8': {
        'score_key': 'critic_step_scores',
        'file_pattern': 'scores_{ds}_critic_v8_per_trajectory.jsonl',
        'label': 'Critic v8 (2000q)',
    },
    'versaprm': {
        'score_key': 'versaprm_step_scores',
        'file_pattern': 'scores_{ds}_versaprm_per_trajectory.jsonl',
        'label': 'VersaPRM',
    },
    'mathprm': {
        'score_key': 'mathprm_step_scores',
        'file_pattern': 'scores_{ds}_mathprm_per_trajectory.jsonl',
        'label': 'MathPRM',
    },
}


# ============================================================
# Metrics (same as compare_voting_methods.py)
# ============================================================

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


def compute_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold answer."""
    if not predicted or not gold:
        return 0.0
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold (EM after normalization)."""
    if not predicted or not gold:
        return False
    return normalize_answer(predicted) == normalize_answer(gold)


# ============================================================
# Data loading and grouping
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def group_by_question(trajectories: List[Dict]) -> Dict[str, List[Dict]]:
    """Group trajectories by question ID, preserving order."""
    groups = {}
    for traj in trajectories:
        tid = traj.get('trajectory_id', '')
        # Format: {qid}_sample_{n}
        idx = tid.rfind('_sample_')
        if idx > 0:
            qid = tid[:idx]
        else:
            qid = tid

        if qid not in groups:
            groups[qid] = []
        groups[qid].append(traj)
    return groups


def get_answer(traj: Dict) -> str:
    """Get predicted answer from trajectory (handles both field names)."""
    return traj.get('predicted_answer', traj.get('final_answer', ''))


# ============================================================
# Scaling computation
# ============================================================

def compute_scaling_for_k(
    groups: Dict[str, List[Dict]],
    k: int,
    score_key: str = None,
) -> Dict:
    """Compute MV, BoN, WMV for first K trajectories per question."""

    aggregations = ['avg', 'min']
    results = {
        'mv': {'em_correct': 0, 'f1_sum': 0.0, 'total': 0},
    }
    if score_key:
        for agg in aggregations:
            results[f'bon_{agg}'] = {'em_correct': 0, 'f1_sum': 0.0, 'total': 0}
            results[f'wmv_{agg}'] = {'em_correct': 0, 'f1_sum': 0.0, 'total': 0}

    for qid, trajs in groups.items():
        subset = trajs[:k]
        if not subset:
            continue

        gold = subset[0].get('gold_answer', '')

        # --- Majority Voting ---
        answers = [get_answer(t) for t in subset]
        answer_counts = Counter()
        original_map = {}
        for a in answers:
            if a:
                key = a.strip().lower()
                answer_counts[key] += 1
                if key not in original_map:
                    original_map[key] = a

        if answer_counts:
            best_key = answer_counts.most_common(1)[0][0]
            mv_answer = original_map.get(best_key, best_key)
        else:
            mv_answer = ''

        results['mv']['total'] += 1
        results['mv']['f1_sum'] += compute_f1(mv_answer, gold)
        if check_answer(mv_answer, gold):
            results['mv']['em_correct'] += 1

        # --- BoN and WMV (only if scores available) ---
        if score_key:
            for agg in aggregations:
                # Compute trajectory-level scores
                traj_scores = []
                for t in subset:
                    step_scores = t.get(score_key, [])
                    if step_scores:
                        if agg == 'min':
                            traj_scores.append(min(step_scores))
                        else:  # avg
                            traj_scores.append(sum(step_scores) / len(step_scores))
                    else:
                        traj_scores.append(0.0)

                # BoN: best trajectory by score
                best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
                bon_answer = get_answer(subset[best_idx])
                results[f'bon_{agg}']['total'] += 1
                results[f'bon_{agg}']['f1_sum'] += compute_f1(bon_answer, gold)
                if check_answer(bon_answer, gold):
                    results[f'bon_{agg}']['em_correct'] += 1

                # WMV: weighted majority voting
                weighted_votes = {}
                orig_answers = {}
                for t, sc in zip(subset, traj_scores):
                    ans = get_answer(t)
                    if ans:
                        key = ans.strip().lower()
                        weighted_votes[key] = weighted_votes.get(key, 0.0) + sc
                        if key not in orig_answers:
                            orig_answers[key] = ans

                if weighted_votes:
                    best_key = max(weighted_votes, key=weighted_votes.get)
                    wmv_answer = orig_answers.get(best_key, best_key)
                else:
                    wmv_answer = ''

                results[f'wmv_{agg}']['total'] += 1
                results[f'wmv_{agg}']['f1_sum'] += compute_f1(wmv_answer, gold)
                if check_answer(wmv_answer, gold):
                    results[f'wmv_{agg}']['em_correct'] += 1

    return results


def to_pct(results: Dict, metric_key: str) -> float:
    """Convert raw results to percentage."""
    total = results['total']
    if total == 0:
        return 0.0
    if metric_key == 'em':
        return 100.0 * results['em_correct'] / total
    else:  # f1
        return 100.0 * results['f1_sum'] / total


# ============================================================
# Printing
# ============================================================

def print_scaling_table(all_results: Dict, k_values: List[int]):
    """Print scaling results as formatted tables."""

    for ds, ds_data in all_results.items():
        print(f"\n{'='*100}")
        print(f"  Dataset: {ds.upper()}")
        print(f"{'='*100}")

        # Collect all method+strategy combos
        columns = ['MV']
        method_keys = []  # (method_name, strategy) tuples for data access

        methods_data = ds_data.get('methods', {})
        for method_name in ['critic_v9', 'critic_v8', 'versaprm', 'mathprm']:
            if method_name not in methods_data:
                continue
            label = METHODS[method_name]['label']
            for strategy in ['bon_avg', 'bon_min', 'wmv_avg', 'wmv_min']:
                short_strategy = strategy.replace('bon_', 'BoN-').replace('wmv_', 'WMV-')
                columns.append(f"{label} {short_strategy}")
                method_keys.append((method_name, strategy))

        # Header
        col_w = 12
        header = f"{'K':>4}"
        for col in columns:
            header += f" | {col:>{col_w}}"
        print(header)
        print("-" * len(header))

        # Rows (F1)
        for ki, k in enumerate(k_values):
            row = f"{k:>4}"

            # MV
            mv_data = methods_data.get('majority_voting', [])
            if ki < len(mv_data):
                row += f" | {mv_data[ki]['f1']:>{col_w}.1f}"
            else:
                row += f" | {'—':>{col_w}}"

            # Other methods
            for method_name, strategy in method_keys:
                m_data = methods_data.get(method_name, {})
                if isinstance(m_data, dict) and strategy in m_data and ki < len(m_data[strategy]):
                    row += f" | {m_data[strategy][ki]['f1']:>{col_w}.1f}"
                else:
                    row += f" | {'—':>{col_w}}"

            print(row)

        print(f"\n  (Values are F1 scores in %)")


# ============================================================
# Plotting
# ============================================================

def plot_scaling_curves(all_results: Dict, k_values: List[int], plot_dir: str):
    """Generate scaling curve plots per dataset."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # Color and style config
    STYLES = {
        'majority_voting': {'color': '#888888', 'linestyle': '--', 'marker': 'o', 'label': 'Majority Voting'},
        'critic_v9_bon_min': {'color': '#e74c3c', 'linestyle': '-', 'marker': 's', 'label': 'Critic v9 BoN(min)'},
        'critic_v9_wmv_min': {'color': '#c0392b', 'linestyle': '-.', 'marker': '^', 'label': 'Critic v9 WMV(min)'},
        'critic_v8_bon_min': {'color': '#3498db', 'linestyle': '-', 'marker': 's', 'label': 'Critic v8 BoN(min)'},
        'critic_v8_wmv_min': {'color': '#2980b9', 'linestyle': '-.', 'marker': '^', 'label': 'Critic v8 WMV(min)'},
        'versaprm_bon_avg': {'color': '#2ecc71', 'linestyle': '-', 'marker': 'D', 'label': 'VersaPRM BoN(avg)'},
        'versaprm_wmv_avg': {'color': '#27ae60', 'linestyle': '-.', 'marker': 'v', 'label': 'VersaPRM WMV(avg)'},
        'mathprm_bon_avg': {'color': '#9b59b6', 'linestyle': '-', 'marker': 'p', 'label': 'MathPRM BoN(avg)'},
        'mathprm_wmv_avg': {'color': '#8e44ad', 'linestyle': '-.', 'marker': 'h', 'label': 'MathPRM WMV(avg)'},
    }

    for ds, ds_data in all_results.items():
        methods_data = ds_data.get('methods', {})

        fig, (ax_f1, ax_em) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'PRM Test-Time Scaling — {ds.upper()}', fontsize=14, fontweight='bold')

        for metric, ax, metric_label in [('f1', ax_f1, 'F1 (%)'), ('em', ax_em, 'EM (%)')]:
            # MV
            mv_data = methods_data.get('majority_voting', [])
            if mv_data:
                vals = [d[metric] for d in mv_data]
                style = STYLES['majority_voting']
                ax.plot(k_values[:len(vals)], vals,
                        color=style['color'], linestyle=style['linestyle'],
                        marker=style['marker'], label=style['label'], markersize=5)

            # Scoring methods — plot best strategies
            for method_name in ['critic_v9', 'critic_v8', 'versaprm', 'mathprm']:
                if method_name not in methods_data:
                    continue
                m_data = methods_data[method_name]
                if not isinstance(m_data, dict):
                    continue

                for strategy in ['bon_avg', 'bon_min', 'wmv_avg', 'wmv_min']:
                    style_key = f'{method_name}_{strategy}'
                    if style_key not in STYLES or strategy not in m_data:
                        continue
                    vals = [d[metric] for d in m_data[strategy]]
                    style = STYLES[style_key]
                    ax.plot(k_values[:len(vals)], vals,
                            color=style['color'], linestyle=style['linestyle'],
                            marker=style['marker'], label=style['label'], markersize=5)

            ax.set_xlabel('K (number of trajectories)', fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_xscale('log', base=2)
            ax.set_xticks(k_values)
            ax.set_xticklabels([str(k) for k in k_values])
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(plot_dir, f'scaling_{ds}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {out_path}")

    # Combined plot (avg across datasets)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('PRM Test-Time Scaling — Average across Datasets (F1)', fontsize=14, fontweight='bold')

    # Collect per-method averages
    all_methods_avg = {}
    datasets_with_data = list(all_results.keys())

    # MV average
    mv_avgs = []
    for ki in range(len(k_values)):
        vals = []
        for ds in datasets_with_data:
            mv_data = all_results[ds].get('methods', {}).get('majority_voting', [])
            if ki < len(mv_data):
                vals.append(mv_data[ki]['f1'])
        mv_avgs.append(sum(vals) / len(vals) if vals else 0)
    all_methods_avg['majority_voting'] = mv_avgs

    for method_name in ['critic_v9', 'critic_v8', 'versaprm', 'mathprm']:
        for strategy in ['bon_avg', 'bon_min', 'wmv_avg', 'wmv_min']:
            style_key = f'{method_name}_{strategy}'
            if style_key not in STYLES:
                continue
            avgs = []
            for ki in range(len(k_values)):
                vals = []
                for ds in datasets_with_data:
                    m_data = all_results[ds].get('methods', {}).get(method_name, {})
                    if isinstance(m_data, dict) and strategy in m_data and ki < len(m_data[strategy]):
                        vals.append(m_data[strategy][ki]['f1'])
                avgs.append(sum(vals) / len(vals) if vals else 0)
            if any(v > 0 for v in avgs):
                all_methods_avg[style_key] = avgs

    for key, vals in all_methods_avg.items():
        style = STYLES.get(key, {'color': 'gray', 'linestyle': '-', 'marker': 'o', 'label': key})
        ax.plot(k_values[:len(vals)], vals,
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], label=style['label'], markersize=5)

    ax.set_xlabel('K (number of trajectories)', fontsize=11)
    ax.set_ylabel('F1 (%)', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(plot_dir, 'scaling_average.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze PRM test-time scaling curves")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with per-trajectory JSONL files and raw trajectories")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset names to analyze")
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help="K values for scaling analysis")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for scaling results")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for scaling curve plots (default: same as output dir)")
    args = parser.parse_args()

    if args.plot_dir is None:
        args.plot_dir = str(Path(args.output).parent / "plots")

    all_results = {}

    for ds in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Analyzing: {ds}")
        print(f"{'='*60}")

        # Load raw trajectories for MV baseline
        raw_path = os.path.join(args.results_dir, f"trajectories_{ds}_128.jsonl")
        if not os.path.exists(raw_path):
            print(f"  [SKIP] Raw trajectories not found: {raw_path}")
            continue

        print(f"  Loading raw trajectories: {raw_path}")
        raw_trajs = load_jsonl(raw_path)
        raw_groups = group_by_question(raw_trajs)
        print(f"  {len(raw_groups)} questions, {len(raw_trajs)} total trajectories")

        ds_results = {'k_values': args.k_values, 'num_questions': len(raw_groups), 'methods': {}}

        # --- Majority Voting scaling ---
        print(f"  Computing Majority Voting scaling...")
        mv_scaling = []
        for k in args.k_values:
            res = compute_scaling_for_k(raw_groups, k)
            total = res['mv']['total']
            mv_scaling.append({
                'k': k,
                'em': to_pct(res['mv'], 'em'),
                'f1': to_pct(res['mv'], 'f1'),
            })
        ds_results['methods']['majority_voting'] = mv_scaling

        # --- Scoring methods ---
        for method_name, method_config in METHODS.items():
            per_traj_path = os.path.join(
                args.results_dir,
                method_config['file_pattern'].format(ds=ds)
            )
            if not os.path.exists(per_traj_path):
                print(f"  [SKIP] {method_config['label']} — file not found: {per_traj_path}")
                continue

            print(f"  Loading {method_config['label']}...")
            scored_trajs = load_jsonl(per_traj_path)
            scored_groups = group_by_question(scored_trajs)
            score_key = method_config['score_key']

            method_scaling = {}
            for strategy in ['bon_avg', 'bon_min', 'wmv_avg', 'wmv_min']:
                method_scaling[strategy] = []

            for k in args.k_values:
                res = compute_scaling_for_k(scored_groups, k, score_key=score_key)
                for strategy in ['bon_avg', 'bon_min', 'wmv_avg', 'wmv_min']:
                    method_scaling[strategy].append({
                        'k': k,
                        'em': to_pct(res[strategy], 'em'),
                        'f1': to_pct(res[strategy], 'f1'),
                    })

            ds_results['methods'][method_name] = method_scaling
            # Print K=128 result
            best_f1 = max(
                method_scaling[s][-1]['f1'] for s in ['bon_avg', 'bon_min', 'wmv_avg', 'wmv_min']
            )
            print(f"    Best F1 at K=128: {best_f1:.1f}%")

        all_results[ds] = ds_results

    # Save JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {args.output}")

    # Print table
    print_scaling_table(all_results, args.k_values)

    # Generate plots
    plot_scaling_curves(all_results, args.k_values, args.plot_dir)


if __name__ == "__main__":
    main()
