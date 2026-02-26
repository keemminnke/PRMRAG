#!/usr/bin/env python3
"""KTO hyperparameter sweep — auto-finds stable (beta, lambda0) combos.

Runs short probe runs (--max-steps) for each config, detects collapse,
then reports results and optionally launches the best config for full training.

Usage:
    # Single GPU
    python scripts/sweep_kto.py

    # 2 GPUs (DDP) — each probe + full training uses torchrun
    python scripts/sweep_kto.py --num-gpus 2
    python scripts/sweep_kto.py --num-gpus 2 --run-winner
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from itertools import product
from pathlib import Path

# ============================================================
# Search grid — edit freely
# ============================================================
BETA_VALUES    = [0.05, 0.1, 0.2]
LAMBDA0_VALUES = [1.0, 1.5, 2.0, 3.0]
LR_VALUES      = [5e-6]          # usually fine to fix LR during beta/lambda sweep

# Default data / model settings (override with CLI flags)
DEFAULT_INPUT      = "outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/kto_sweep_tmp"
DEFAULT_MAX_STEPS  = 50
DEFAULT_LIMIT      = 500          # trajectories to use per probe (keep small for speed)
DEFAULT_LOGGING    = 10


def parse_args():
    p = argparse.ArgumentParser(description="KTO hyperparameter sweep")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Training data JSONL")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Number of GPUs for DDP (1=single GPU, 2+=torchrun)")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                   help="Optimizer steps per probe run")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="Max trajectories per probe (keep small)")
    p.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=None,
                   help="Gradient accum steps (default: 16/num_gpus to keep eff batch=32)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--truncate-after-bad", action="store_true", default=True)
    p.add_argument("--run-winner", action="store_true",
                   help="After sweep, automatically launch full training with best config")
    p.add_argument("--full-output-dir", default="outputs/kto_policy_sweep_winner",
                   help="Output dir for full training run (used with --run-winner)")
    p.add_argument("--full-epochs", type=int, default=1,
                   help="Epochs for full training run")
    args = p.parse_args()
    # Auto-adjust gradient accumulation to keep effective batch size = 32
    if args.gradient_accumulation is None:
        args.gradient_accumulation = max(1, 16 // args.num_gpus)
    return args


def _build_launcher(num_gpus: int) -> list:
    """Return the command prefix: torchrun for multi-GPU, python for single."""
    if num_gpus > 1:
        return ["torchrun", f"--nproc_per_node={num_gpus}"]
    return [sys.executable]


def run_probe(config: dict, args, tmp_dir: Path) -> dict:
    """Run a single probe and return the result dict."""
    beta     = config["beta"]
    lambda0  = config["lambda0"]
    lr       = config["lr"]

    result_file = tmp_dir / f"result_b{beta}_l{lambda0}_lr{lr}.json"
    out_dir     = tmp_dir / f"run_b{beta}_l{lambda0}_lr{lr}"

    cmd = _build_launcher(args.num_gpus) + [
        "scripts/train_kto_policy.py",
        "--input", args.input,
        "--output-dir", str(out_dir),
        "--beta", str(beta),
        "--lambda0", str(lambda0),
        "--learning-rate", str(lr),
        "--max-steps", str(args.max_steps),
        "--limit", str(args.limit),
        "--logging-steps", str(args.logging_steps),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--sweep-result-file", str(result_file),
        "--no-wandb",
    ]
    if args.truncate_after_bad:
        cmd.append("--truncate-after-bad")

    label = f"beta={beta}, lambda0={lambda0}, lr={lr}"
    print(f"\n{'─'*60}")
    print(f"[Probe] {label}")
    print(f"{'─'*60}")

    proc = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

    if not result_file.exists():
        print(f"  [ERROR] No result file produced (exit code {proc.returncode})")
        return {"config": config, "collapsed": True, "collapse_reason": "process_error",
                "final_bad_loss": None, "final_kl": None, "final_good_loss": None,
                "final_step": None}

    with open(result_file) as f:
        result = json.load(f)

    result["config"] = config
    return result


def score_result(r: dict) -> float:
    """Higher = better config. Returns -inf if collapsed."""
    if r.get("collapsed"):
        return float("-inf")
    bad_loss  = r.get("final_bad_loss") or 0.0
    good_loss = r.get("final_good_loss") or 1.0
    kl        = r.get("final_kl") or 0.0
    # Want: bad_loss high (not collapsed), good_loss decreasing, KL ≥ 0
    return bad_loss - 0.5 * good_loss + 0.1 * max(kl, 0.0)


def print_table(results: list):
    header = f"{'beta':>6} {'lambda0':>8} {'lr':>8} {'step':>6} {'bad_loss':>9} {'good_loss':>10} {'KL':>7}  status"
    print(f"\n{'='*75}")
    print("SWEEP RESULTS")
    print(f"{'='*75}")
    print(header)
    print("─" * 75)

    for r in sorted(results, key=score_result, reverse=True):
        cfg = r.get("config", {})
        b  = cfg.get("beta", "?")
        l0 = cfg.get("lambda0", "?")
        lr = cfg.get("lr", "?")
        step      = r.get("final_step", "?")
        bad_loss  = r.get("final_bad_loss")
        good_loss = r.get("final_good_loss")
        kl        = r.get("final_kl")
        status    = "COLLAPSED" if r.get("collapsed") else "OK"
        reason    = f" ({r.get('collapse_reason', '')})" if r.get("collapsed") else ""

        bl_str  = f"{bad_loss:.4f}"  if bad_loss  is not None else "  N/A"
        gl_str  = f"{good_loss:.4f}" if good_loss is not None else "  N/A"
        kl_str  = f"{kl:.3f}"        if kl        is not None else "  N/A"

        print(f"{b:>6} {l0:>8} {lr:>8} {step!s:>6} {bl_str:>9} {gl_str:>10} {kl_str:>7}  {status}{reason}")

    print(f"{'='*75}")


def main():
    args = parse_args()

    configs = [
        {"beta": b, "lambda0": l0, "lr": lr}
        for b, l0, lr in product(BETA_VALUES, LAMBDA0_VALUES, LR_VALUES)
    ]

    eff_batch = args.batch_size * args.gradient_accumulation * args.num_gpus
    print(f"{'='*60}")
    print(f"KTO Hyperparameter Sweep")
    print(f"{'='*60}")
    print(f"  Configs   : {len(configs)}")
    print(f"  Steps/run : {args.max_steps}")
    print(f"  Data limit: {args.limit} trajectories")
    print(f"  Grid      : beta={BETA_VALUES}, lambda0={LAMBDA0_VALUES}")
    print(f"  GPUs      : {args.num_gpus} ({'torchrun' if args.num_gpus > 1 else 'python'})")
    print(f"  Eff batch : {args.batch_size} x {args.gradient_accumulation} x {args.num_gpus} = {eff_batch}")
    print()

    tmp_dir = Path(DEFAULT_OUTPUT_DIR) / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Temp dir  : {tmp_dir}\n")

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running config: {config}")
        r = run_probe(config, args, tmp_dir)
        results.append(r)

        # Save intermediate results
        with open(tmp_dir / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print_table(results)

    # Find best
    valid = [r for r in results if not r.get("collapsed")]
    if not valid:
        print("\n[!] All configs collapsed. Try smaller beta or lambda0.")
        return

    best = max(valid, key=score_result)
    bc = best["config"]
    print(f"\nBest config: beta={bc['beta']}, lambda0={bc['lambda0']}, lr={bc['lr']}")
    print(f"  bad_loss={best.get('final_bad_loss'):.4f}, "
          f"good_loss={best.get('final_good_loss'):.4f}, "
          f"KL={best.get('final_kl'):.3f}")

    # Print full training command
    launcher = f"torchrun --nproc_per_node={args.num_gpus}" if args.num_gpus > 1 else "python"
    full_cmd = (
        f"{launcher} scripts/train_kto_policy.py \\\n"
        f"    --input {args.input} \\\n"
        f"    --output-dir {args.full_output_dir} \\\n"
        f"    --beta {bc['beta']} \\\n"
        f"    --lambda0 {bc['lambda0']} \\\n"
        f"    --learning-rate {bc['lr']} \\\n"
        f"    --batch-size {args.batch_size} \\\n"
        f"    --gradient-accumulation {args.gradient_accumulation} \\\n"
        f"    --num-epochs {args.full_epochs} \\\n"
        f"    --lora-r {args.lora_r} \\\n"
        f"    --lora-alpha {args.lora_alpha} \\\n"
        f"    --save-steps 100 \\\n"
        f"    --truncate-after-bad \\\n"
        f"    --wandb-run-name kto_sweep_winner"
    )
    print(f"\n{'─'*60}")
    print("Full training command:")
    print(f"{'─'*60}")
    print(full_cmd)

    if args.run_winner:
        print(f"\n[--run-winner] Launching full training...")
        winner_cmd = _build_launcher(args.num_gpus) + [
            "scripts/train_kto_policy.py",
            "--input", args.input,
            "--output-dir", args.full_output_dir,
            "--beta", str(bc["beta"]),
            "--lambda0", str(bc["lambda0"]),
            "--learning-rate", str(bc["lr"]),
            "--batch-size", str(args.batch_size),
            "--gradient-accumulation", str(args.gradient_accumulation),
            "--num-epochs", str(args.full_epochs),
            "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
            "--save-steps", "100",
            "--truncate-after-bad",
            "--wandb-run-name", "kto_sweep_winner",
        ]
        subprocess.run(winner_cmd, cwd=str(Path(__file__).parent.parent))


if __name__ == "__main__":
    main()
