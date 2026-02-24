#!/usr/bin/env python3
"""Train policy model with step-level DPO loss (document masking).

Usage:
    # Small test
    python scripts/train_dpo_policy.py \
        --trajectories outputs/judge_labels_merged_hotpotqa_musique.jsonl \
        --critic-scores outputs/critic_scores_v8_binary_per_trajectory.jsonl \
        --output-dir outputs/dpo_policy_test \
        --limit 50 --no-wandb

    # Full training
    python scripts/train_dpo_policy.py \
        --trajectories outputs/judge_labels_merged_hotpotqa_musique.jsonl \
        --critic-scores outputs/critic_scores_v8_binary_per_trajectory.jsonl \
        --output-dir outputs/dpo_policy_v1 \
        --pairing best-vs-all \
        --wandb-run-name dpo_policy_v1

Key ideas:
  - Input: trajectory-level pairs (chosen=correct, rejected=incorrect)
  - Loss: DPO with document masking (<documents>...</documents> tokens excluded)
  - Custom StepLevelDPOTrainer with separate ref model
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from prmrag.training.dpo_trainer import (
    DPODataPreparer,
    StepLevelDPOTrainer,
    DPODebugCallback,
)

HF_CACHE_DIR = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train policy model with step-level DPO loss"
    )

    # Data
    parser.add_argument(
        "--trajectories", type=Path, required=True,
        help="Path to judge_labels JSONL (step content)",
    )
    parser.add_argument(
        "--critic-scores", type=Path, required=True,
        help="Path to critic_scores per_trajectory JSONL",
    )

    # Model
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/dpo_policy_v1",
        help="Output directory",
    )

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument(
        "--pairing", type=str, default="best-vs-all",
        choices=["best-vs-all", "best-vs-worst", "all-vs-all"],
        help="Pairing strategy for chosen/rejected",
    )

    # Training
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="prmrag-dpo", help="Wandb project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")

    # Debug
    parser.add_argument("--limit", type=int, default=None, help="Limit data for debugging")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not args.trajectories.exists():
        print(f"Error: Trajectories not found: {args.trajectories}")
        sys.exit(1)
    if not args.critic_scores.exists():
        print(f"Error: Critic scores not found: {args.critic_scores}")
        sys.exit(1)

    print("=" * 70)
    print("DPO POLICY TRAINING (Document-masked Step-level)")
    print("=" * 70)
    print()
    print(f"Trajectories:    {args.trajectories}")
    print(f"Critic scores:   {args.critic_scores}")
    print(f"Model:           {args.model_name}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Pairing:         {args.pairing}")
    print(f"Beta:            {args.beta}")
    print(f"Max length:      {args.max_length}")
    print(f"Batch size:      {args.batch_size} x {args.gradient_accumulation} accum")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"LoRA:            r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Wandb:           {'Disabled' if args.no_wandb else args.wandb_project}")
    if args.limit:
        print(f"Limit:           {args.limit}")
    print()

    # =========================================================================
    # 1. Load tokenizer
    # =========================================================================
    print("Loading tokenizer...")
    os.environ["HF_HOME"] = HF_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # =========================================================================
    # 2. Prepare dataset
    # =========================================================================
    print("\nPreparing DPO dataset...")
    preparer = DPODataPreparer(tokenizer)
    dataset = preparer.prepare_dataset(
        trajectories_path=str(args.trajectories),
        critic_scores_path=str(args.critic_scores),
        pairing=args.pairing,
        limit=args.limit,
    )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: No training pairs produced!")
        sys.exit(1)

    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE TRAINING DATA")
    print("=" * 70)
    sample = dataset[0]
    print(f"Prompt (last 200 chars): ...{sample['prompt'][-200:]}")
    print(f"Chosen (first 300 chars): {sample['chosen'][:300]}...")
    print(f"Rejected (first 300 chars): {sample['rejected'][:300]}...")
    print()

    # Token length distribution
    print("Analyzing token lengths...")
    lengths_chosen = []
    lengths_rejected = []
    for i in range(min(100, len(dataset))):
        s = dataset[i]
        cho_toks = tokenizer.encode(s["prompt"] + s["chosen"], add_special_tokens=False)
        rej_toks = tokenizer.encode(s["prompt"] + s["rejected"], add_special_tokens=False)
        lengths_chosen.append(len(cho_toks))
        lengths_rejected.append(len(rej_toks))

    print(f"Token lengths (first {len(lengths_chosen)} samples):")
    print(f"  Chosen   - Min: {min(lengths_chosen)}, Max: {max(lengths_chosen)}, "
          f"Avg: {sum(lengths_chosen)/len(lengths_chosen):.0f}")
    print(f"  Rejected - Min: {min(lengths_rejected)}, Max: {max(lengths_rejected)}, "
          f"Avg: {sum(lengths_rejected)/len(lengths_rejected):.0f}")
    exceeding = sum(1 for l in lengths_chosen + lengths_rejected if l > args.max_length)
    if exceeding:
        total = len(lengths_chosen) + len(lengths_rejected)
        print(f"  Exceeding max_length: {exceeding}/{total} ({100*exceeding/total:.1f}%)")
    print()

    # =========================================================================
    # 3. Load models
    # =========================================================================
    print("Loading policy model (with LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE_DIR,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE_DIR,
    )
    ref_model.eval()

    # =========================================================================
    # 4. Setup training
    # =========================================================================
    report_to = []
    if not args.no_wandb and WANDB_AVAILABLE:
        report_to.append("wandb")
        run_name = args.wandb_run_name or f"dpo_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "method": "dpo_doc_masked",
                "beta": args.beta,
                "pairing": args.pairing,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": args.batch_size * args.gradient_accumulation,
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "num_epochs": args.num_epochs,
                "num_pairs": len(dataset),
            },
            tags=["dpo", "prmrag", "doc-masked"],
        )
        print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    else:
        report_to.append("tensorboard")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        save_total_limit=2,
        report_to=report_to,
        remove_unused_columns=False,  # We need custom columns
        dataloader_pin_memory=True,
    )

    # =========================================================================
    # 5. Train
    # =========================================================================
    debug_callback = DPODebugCallback()

    trainer = StepLevelDPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=args.beta,
        max_length=args.max_length,
        args=training_args,
        train_dataset=dataset,
        callbacks=[debug_callback],
    )

    # Check for checkpoint resume
    checkpoint_dir = Path(args.output_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    ) if checkpoint_dir.exists() else []

    resume_checkpoint = None
    if checkpoints:
        resume_checkpoint = str(checkpoints[-1])
        print(f"\nResuming from checkpoint: {resume_checkpoint}")

    print(f"\n{'='*70}")
    print("STARTING DPO TRAINING")
    print(f"{'='*70}\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # =========================================================================
    # 6. Save
    # =========================================================================
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save training info
    info = {
        "model_name": args.model_name,
        "method": "dpo_doc_masked",
        "beta": args.beta,
        "pairing": args.pairing,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "num_pairs": len(dataset),
        "completed_at": datetime.now().isoformat(),
    }
    with open(Path(args.output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*70}")
    print("DPO TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Model saved to: {final_path}")
    print()
    print("To use the trained model:")
    print(f"  python scripts/generate_trajectories.py \\")
    print(f"      --lora_adapter {final_path} ...")
    print()

    # Finish wandb
    if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
