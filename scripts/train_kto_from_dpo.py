#!/usr/bin/env python3
"""Train KTO on chosen/rejected from DPO dataset.

Uses TRL KTOTrainer. chosen → desirable, rejected → undesirable.
Document masking applied via custom trainer.

Usage:
    torchrun --nproc_per_node=2 scripts/train_kto_from_dpo.py \
        --dpo-dataset outputs/mcts_dpo_5000q_v4b.jsonl \
        --output-dir outputs/kto_from_dpo_v1 \
        --wandb-run-name kto_from_dpo_v1
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOTrainer, KTOConfig
from peft import LoraConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

HF_CACHE_DIR = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

# Qwen2.5 token IDs for document masking
_OPEN_LT = 27
_CLOSE_LT = 522
_DOC_TOKEN = 50778


class DocMaskKTOTrainer(KTOTrainer):
    """KTOTrainer that masks <documents>...</documents> in completion."""

    _mask_log_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "completion_mask" in inputs and "input_ids" in inputs:
            masked_count = self._mask_document_spans(inputs)
            DocMaskKTOTrainer._mask_log_counter += 1
            if DocMaskKTOTrainer._mask_log_counter <= 3 or DocMaskKTOTrainer._mask_log_counter % 200 == 0:
                active = inputs["completion_mask"].sum().item()
                print(f"[DocMask-KTO] step={DocMaskKTOTrainer._mask_log_counter} "
                      f"active_tokens={active} doc_masked={masked_count}")
        return super().compute_loss(
            model, inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    @staticmethod
    def _mask_document_spans(inputs):
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        batch_size, seq_len = input_ids.shape
        masked_total = 0

        for i in range(batch_size):
            ids = input_ids[i].tolist()
            in_doc = False
            j = 0
            while j < seq_len:
                if (not in_doc and j + 1 < seq_len
                        and ids[j] == _OPEN_LT and ids[j + 1] == _DOC_TOKEN):
                    in_doc = True
                if in_doc:
                    completion_mask[i, j] = 0
                    masked_total += 1
                    if (j + 1 < seq_len
                            and ids[j] == _CLOSE_LT and ids[j + 1] == _DOC_TOKEN):
                        completion_mask[i, j] = 0
                        if j + 1 < seq_len:
                            completion_mask[i, j + 1] = 0
                            masked_total += 1
                        if j + 2 < seq_len:
                            completion_mask[i, j + 2] = 0
                            masked_total += 1
                        j += 3
                        in_doc = False
                        continue
                j += 1

        return masked_total


def parse_args():
    p = argparse.ArgumentParser(description="KTO from DPO chosen/rejected")
    p.add_argument("--dpo-dataset", type=Path, required=True)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", type=str, default="outputs/kto_from_dpo_v1")

    p.add_argument("--beta", type=float, default=0.1, help="KTO beta")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)

    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--wandb-project", type=str, default="prmrag-kto")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.model_name):
        args.model_name = os.path.abspath(args.model_name)

    if not args.dpo_dataset.exists():
        print(f"Error: Dataset not found: {args.dpo_dataset}")
        sys.exit(1)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["HF_HOME"] = HF_CACHE_DIR

    # Load DPO data → KTO format: each pair becomes 2 samples
    # chosen → desirable=True, rejected → desirable=False
    records = []
    with open(args.dpo_dataset) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                prompt = d["prompt"]
                # Desirable (chosen)
                records.append({
                    "prompt": prompt,
                    "completion": d["chosen"],
                    "label": True,
                })
                # Undesirable (rejected)
                records.append({
                    "prompt": prompt,
                    "completion": d["rejected"],
                    "label": False,
                })

    if args.limit:
        records = records[:args.limit * 2]

    dataset = HFDataset.from_list(records)

    n_desirable = sum(1 for r in records if r["label"])
    n_undesirable = sum(1 for r in records if not r["label"])

    if local_rank == 0:
        print("=" * 70)
        print("KTO FROM DPO (Document Masking)")
        print("=" * 70)
        print(f"Dataset:      {args.dpo_dataset}")
        print(f"Total:        {len(dataset)} ({n_desirable} desirable, {n_undesirable} undesirable)")
        print(f"Model:        {args.model_name}")
        print(f"Output:       {args.output_dir}")
        print(f"Beta:         {args.beta}")
        print(f"Max length:   {args.max_length}")
        num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        eff = args.batch_size * args.gradient_accumulation * num_gpus
        print(f"Batch:        {args.batch_size} x {args.gradient_accumulation} x {num_gpus} = {eff} eff")
        print(f"LR:           {args.learning_rate}")
        print(f"LoRA:         r={args.lora_r}, alpha={args.lora_alpha}")
        print()

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE_DIR,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA
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

    # Wandb
    report_to = []
    if not args.no_wandb and WANDB_AVAILABLE:
        report_to.append("wandb")
        if local_rank == 0:
            run_name = args.wandb_run_name or f"kto_dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_name": args.model_name,
                    "method": "kto_from_dpo",
                    "beta": args.beta,
                    "dataset": str(args.dpo_dataset),
                    "num_samples": len(dataset),
                    "n_desirable": n_desirable,
                    "n_undesirable": n_undesirable,
                    "learning_rate": args.learning_rate,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "num_epochs": args.num_epochs,
                },
                tags=["kto", "dpo-data", "prmrag"],
            )
    else:
        report_to.append("tensorboard")

    training_args = KTOConfig(
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=2,
        report_to=report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        beta=args.beta,
        max_length=args.max_length,
    )

    trainer = DocMaskKTOTrainer(
        model=model,
        ref_model=None,  # TRL uses adapter-disabled model as reference
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Resume
    checkpoint_dir = Path(args.output_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    ) if checkpoint_dir.exists() else []
    resume_checkpoint = str(checkpoints[-1]) if checkpoints else None
    if resume_checkpoint and local_rank == 0:
        print(f"Resuming from: {resume_checkpoint}")

    if local_rank == 0:
        print(f"\n{'='*70}")
        print("STARTING KTO TRAINING")
        print(f"{'='*70}\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    if local_rank == 0:
        tokenizer.save_pretrained(str(final_path))
        info = {
            "model_name": args.model_name,
            "method": "kto_from_dpo",
            "beta": args.beta,
            "dataset": str(args.dpo_dataset),
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_length": args.max_length,
            "num_samples": len(dataset),
            "n_desirable": n_desirable,
            "n_undesirable": n_undesirable,
            "num_epochs": args.num_epochs,
            "document_masking": True,
            "completed_at": datetime.now().isoformat(),
        }
        with open(Path(args.output_dir) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n{'='*70}")
        print("KTO TRAINING COMPLETED")
        print(f"Model saved to: {final_path}")
        print(f"{'='*70}")

    if not args.no_wandb and WANDB_AVAILABLE and local_rank == 0 and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
