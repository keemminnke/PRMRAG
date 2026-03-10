#!/usr/bin/env python3
"""DPO 학습 실시간 모니터링.

Usage:
    python scripts/monitor_dpo.py
    python scripts/monitor_dpo.py --log logs/step_dpo_experiment.log --interval 30
"""

import re
import time
import argparse
import os

def parse_metrics(log_path):
    """로그에서 모든 학습 metric을 추출."""
    metrics = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\{'loss': (.+?)\}", line)
            if m:
                try:
                    d = eval("{" + m.group(0)[1:])
                    metrics.append(d)
                except:
                    pass
    return metrics


def display(metrics, show_all=False):
    """학습 상태를 요약 출력."""
    if not metrics:
        print("No metrics found yet.")
        return

    latest = metrics[-1]
    total_steps = None
    # Try to find total steps from progress bar
    step = len(metrics) * 10  # rough estimate

    # Key metrics
    loss = latest.get('loss', 0)
    acc = latest.get('rewards/accuracies', 0)
    margin = latest.get('rewards/margins', 0)
    chosen_r = latest.get('rewards/chosen', 0)
    rejected_r = latest.get('rewards/rejected', 0)
    epoch = latest.get('epoch', 0)
    lr = latest.get('learning_rate', 0)
    grad_norm = latest.get('grad_norm', 0)

    os.system('clear')
    print("=" * 65)
    print("  Step-level DPO Training Monitor")
    print("=" * 65)
    print(f"  Epoch:  {epoch:.3f}  |  Log entries: {len(metrics)}")
    print(f"  LR:     {lr:.2e}")
    print("-" * 65)
    print(f"  {'Metric':<25} {'Current':>10} {'First':>10} {'Delta':>10}")
    print("-" * 65)

    first = metrics[0]
    rows = [
        ("Loss",              'loss'),
        ("Rewards Accuracy",  'rewards/accuracies'),
        ("Rewards Margin",    'rewards/margins'),
        ("Rewards Chosen",    'rewards/chosen'),
        ("Rewards Rejected",  'rewards/rejected'),
        ("Grad Norm",         'grad_norm'),
        ("Logps Chosen",      'logps/chosen'),
        ("Logps Rejected",    'logps/rejected'),
    ]

    for label, key in rows:
        cur = latest.get(key, 0)
        fst = first.get(key, 0)
        delta = cur - fst
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<25} {cur:>10.4f} {fst:>10.4f} {sign}{delta:>9.4f}")

    print("-" * 65)

    # Trend (last 5 entries)
    if len(metrics) >= 3:
        recent = metrics[-5:]
        print(f"\n  Recent trend (last {len(recent)} logs):")
        print(f"  {'Entry':<8} {'Loss':>8} {'Acc':>8} {'Margin':>10} {'Chosen':>10} {'Rejected':>10}")
        print(f"  {'-'*56}")
        for i, m in enumerate(recent):
            idx = len(metrics) - len(recent) + i + 1
            print(f"  {idx:<8} {m.get('loss',0):>8.4f} {m.get('rewards/accuracies',0):>8.3f} "
                  f"{m.get('rewards/margins',0):>10.4f} {m.get('rewards/chosen',0):>10.4f} "
                  f"{m.get('rewards/rejected',0):>10.4f}")

    # Health check
    print(f"\n  --- Health Check ---")
    if loss > 0.69:
        print(f"  [!] Loss {loss:.4f} > 0.69 — 아직 학습 초기, 관찰 필요")
    elif loss > 0.68:
        print(f"  [~] Loss {loss:.4f} — 서서히 감소 중")
    elif loss > 0.65:
        print(f"  [OK] Loss {loss:.4f} — 정상 학습 진행")
    else:
        print(f"  [OK] Loss {loss:.4f} — 잘 내려가는 중")

    if acc > 0.7:
        print(f"  [!!] Accuracy {acc:.3f} > 0.7 — 과적합 주의")
    elif acc > 0.6:
        print(f"  [OK] Accuracy {acc:.3f} — 양호")
    elif acc > 0.55:
        print(f"  [~] Accuracy {acc:.3f} — 서서히 개선 중")
    else:
        print(f"  [!] Accuracy {acc:.3f} — 아직 랜덤 수준")

    if margin > 0:
        print(f"  [OK] Margin {margin:.4f} > 0 — chosen이 rejected보다 선호됨")
    else:
        print(f"  [!] Margin {margin:.4f} <= 0 — chosen/rejected 구분 못함")

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/step_dpo_experiment.log")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    if args.once:
        metrics = parse_metrics(args.log)
        display(metrics)
        return

    print(f"Monitoring {args.log} (refresh every {args.interval}s, Ctrl+C to stop)\n")
    try:
        while True:
            metrics = parse_metrics(args.log)
            display(metrics)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
