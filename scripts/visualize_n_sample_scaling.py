#!/usr/bin/env python3
import json, os, random, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from matplotlib.ticker import ScalarFormatter

# ─── helpers ──────────────────────────────────────────────────────────────
def check_answer(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    return gold.strip().lower() in pred.strip().lower()

def load_data(filepath: str):
    """Load jsonl and group by question."""
    by_q = defaultdict(list)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return {}
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            # 이미지처럼 도메인이 나뉘어 있다면 rec["domain"] 사용, 
            # 없다면 기본값 "HotpotQA" 등으로 지정
            domain = rec.get("domain", "HotpotQA") 
            by_q[domain].append(rec)
    
    # 도메인별로 질문 그룹화
    domain_data = {}
    for domain, trajs in by_q.items():
        q_groups = defaultdict(list)
        for t in trajs:
            q_groups[t["question"]].append(t)
        
        q_list = []
        for q, ts in q_groups.items():
            gold = ts[0]["gold_answer"]
            q_list.append({
                "gold": gold,
                "pred": [t["predicted_answer"] for t in ts],
                "is_correct": [check_answer(t["predicted_answer"], gold) for t in ts],
                "critic_min": [t.get("critic_min", 0.0) for t in ts],
                "versaprm_min": [t.get("versaprm_min", 0.0) for t in ts],
                "n": len(ts),
            })
        domain_data[domain] = q_list
    return domain_data

def simulate(q_data, sample_powers, n_seeds=5):
    methods = ["majority", "critic_wmv", "versa_wmv", "critic_bon", "versa_bon"]
    results = {m: {n: [] for n in sample_powers} for m in methods}

    for n in sample_powers:
        for seed in range(n_seeds):
            random.seed(seed)
            counts = {m: 0 for m in methods}
            for qd in q_data:
                idxs = random.sample(range(qd["n"]), min(n, qd["n"]))
                gold_norm = qd["gold"].strip().lower()

                # Majority Voting
                preds = [qd["pred"][i].strip().lower() for i in idxs]
                if preds:
                    voted = Counter(preds).most_common(1)[0][0]
                    counts["majority"] += (gold_norm in voted)

                # WMV (Weighted Majority Voting)
                for prefix, key in [("critic", "critic_min"), ("versa", "versaprm_min")]:
                    wmv_scores = {}
                    for i in idxs:
                        ans = qd["pred"][i].strip().lower()
                        wmv_scores[ans] = wmv_scores.get(ans, 0.0) + qd[key][i]
                    if wmv_scores:
                        best_ans = max(wmv_scores, key=wmv_scores.get)
                        counts[f"{prefix}_wmv"] += (gold_norm in best_ans)

                # BoN (Best-of-N)
                for prefix, key in [("critic", "critic_min"), ("versa", "versaprm_min")]:
                    best = max(idxs, key=lambda i: qd[key][i])
                    counts[f"{prefix}_bon"] += qd["is_correct"][best]

            for m in methods:
                results[m][n].append(counts[m] / len(q_data) * 100)
    
    return {m: {n: np.mean(results[m][n]) for n in sample_powers} for m in methods}

# ─── Plotting ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/versaprm_per_trajectory.jsonl")
    parser.add_argument("--output", default="outputs/full_scaling_results.png")
    args = parser.parse_args()

    sample_powers = [1, 2, 4, 8, 16, 32, 64, 128]
    domain_data = load_data(args.input)
    
    # 도메인 리스트 (이미지와 동일한 순서, 데이터가 없으면 HotpotQA 하나만 그림)
    target_domains = list(domain_data.keys())[:3] 
    if not target_domains: return

    fig, axes = plt.subplots(2, len(target_domains), figsize=(6 * len(target_domains), 10))
    if len(target_domains) == 1: axes = axes.reshape(2, 1)

    # 스타일 설정 (이미지 범례 기준)
    style = {
        "majority":   {"color": "black",   "marker": "*", "label": "Majority Voting", "ls": "--"},
        "critic_wmv": {"color": "#ff7f0e", "marker": "o", "label": "Critic PRM WMV"},
        "versa_wmv":  {"color": "#d62728", "marker": "o", "label": "VersaPRM WMV (Ours)"},
        "critic_bon": {"color": "#ff7f0e", "marker": "s", "label": "Critic PRM BoN"},
        "versa_bon":  {"color": "#d62728", "marker": "s", "label": "VersaPRM BoN (Ours)"}
    }

    for col, domain in enumerate(target_domains):
        print(f"Simulating {domain}...")
        res = simulate(domain_data[domain], sample_powers)

        # Row 0: WMV Comparison
        ax0 = axes[0, col]
        ax0.plot(sample_powers, [res["majority"][n] for n in sample_powers], **style["majority"])
        ax0.plot(sample_powers, [res["critic_wmv"][n] for n in sample_powers], **style["critic_wmv"])
        ax0.plot(sample_powers, [res["versa_wmv"][n] for n in sample_powers], **style["versa_wmv"])
        ax0.set_title(f"{domain} (WMV)")

        # Row 1: BoN Comparison
        ax1 = axes[1, col]
        ax1.plot(sample_powers, [res["majority"][n] for n in sample_powers], **style["majority"])
        ax1.plot(sample_powers, [res["critic_bon"][n] for n in sample_powers], **style["critic_bon"])
        ax1.plot(sample_powers, [res["versa_bon"][n] for n in sample_powers], **style["versa_bon"])
        ax1.set_title(f"{domain} (BoN)")

        for ax in [ax0, ax1]:
            ax.set_xscale('log', base=2)
            ax.set_xticks(sample_powers)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, which='both', linestyle='-', alpha=0.3)
            # Y축을 선형(Linear)으로 유지 (이미지와 동일)
            ax.set_yscale('linear') 

    axes[1, 1 if len(target_domains)>1 else 0].set_xlabel("Number of generated CoT solutions (log scale)")
    
    # 범례 통합 표시
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(args.output, dpi=300)
    plt.savefig(args.output.replace(".png", ".pdf"))
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
