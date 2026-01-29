"""Critic model trainer for step-level evaluation.

This module provides:
- Fine-tuning critic model with consensus-filtered data
- Rationale generation + Verdict prediction (Equation 6, 7)
- LoRA-based efficient fine-tuning
- Automatic masking for instruction-only loss
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


@dataclass
class CriticTrainingConfig:
    """Configuration for critic model training."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])

    # Training settings
    output_dir: str = "outputs/critic_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096  # Increased for RAG (long observations)

    # Data settings
    consensus_only: bool = True  # Only use consensus steps

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Optimization
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True


class CriticDataFormatter:
    """Format trajectory data for critic training (Chat Template Support)."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        # Response template for masking - will be set based on tokenizer
        self._response_template = None

    @property
    def response_template(self) -> str:
        """Get response template based on tokenizer type."""
        if self._response_template is None:
            # Try to detect from tokenizer
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                if 'im_start' in self.tokenizer.chat_template:
                    # Qwen/DeepSeek style
                    self._response_template = "<|im_start|>assistant\n"
                else:
                    # Generic fallback
                    self._response_template = "assistant\n"
            else:
                self._response_template = "<|im_start|>assistant\n"
        return self._response_template

    def format_step_for_training(
        self,
        question: str,
        previous_steps: List[Dict[str, Any]],
        current_step: Dict[str, Any],
    ) -> str:
        """Format a single step into training format using Chat Template.

        NOTE: Gold Answer is intentionally excluded to prevent data leakage.
        The critic must learn to evaluate step quality based on reasoning logic,
        not by comparing with the answer.

        Args:
            question: Original question
            previous_steps: List of previous steps (context)
            current_step: Step to evaluate

        Returns:
            Formatted text for training with chat template applied
        """
        # Build user input context (NO gold answer - prevents data leakage)
        input_parts = []
        input_parts.append(f"Question: {question}")
        input_parts.append("")

        # Add previous steps for context
        if previous_steps:
            input_parts.append("Previous Steps:")
            for i, prev_step in enumerate(previous_steps, 1):
                input_parts.append(f"Step {i}:")

                # Add thought if exists
                thought = prev_step.get('thought')
                if thought:
                    input_parts.append(f"Thought: {thought}")

                # Add action
                action = prev_step.get('action', 'Unknown')
                action_input = prev_step.get('action_input', '')
                if action_input:
                    input_parts.append(f"Action: {action}[{action_input}]")
                else:
                    input_parts.append(f"Action: {action}")

                # Add observation if exists
                obs = prev_step.get('observation')
                if obs:
                    input_parts.append(f"Observation: {obs}")

                input_parts.append("")

        # Add current step to evaluate
        input_parts.append("Current Step to Evaluate:")

        thought = current_step.get('thought')
        if thought:
            input_parts.append(f"Thought: {thought}")

        action = current_step.get('action', 'Unknown')
        action_input = current_step.get('action_input', '')
        if action_input:
            input_parts.append(f"Action: {action}[{action_input}]")
        else:
            input_parts.append(f"Action: {action}")

        obs = current_step.get('observation')
        if obs:
            input_parts.append(f"Observation: {obs}")

        input_parts.append("")
        input_parts.append("Task: Evaluate the quality of the Current Step. Analyze whether the reasoning is logically sound and grounded in evidence, then provide a label (GOOD/BAD).")

        user_content = "\n".join(input_parts)

        # Build assistant response (what model should learn)
        label = current_step.get('judge_label', 'UNKNOWN').upper()
        reasoning = current_step.get('judge_reasoning', '')
        assistant_content = f"Reasoning: {reasoning}\nLabel: {label}"

        # System prompt for critic role
        system_content = "You are a step-level critic model for evaluating reasoning quality in multi-hop question answering. Analyze each step's logical soundness and evidence grounding, then provide a label (GOOD/BAD)."

        # Return messages format for trl conversational training
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        return messages

    def prepare_training_data(
        self,
        data_file: Path,
        consensus_only: bool = True,
    ) -> List[Dict[str, str]]:
        """Prepare training data from merged judge results.

        Args:
            data_file: Path to merged JSONL file with judge labels
            consensus_only: Only use steps with consensus (if True) or include
                           all steps with judge labels (both GOOD and BAD)

        Returns:
            List of formatted training samples
        """
        training_samples = []
        label_counts = {'GOOD': 0, 'BAD': 0}

        with open(data_file) as f:
            for line in f:
                trajectory = json.loads(line)

                question = trajectory['question']
                # gold_answer removed - prevents data leakage
                steps = trajectory['steps']

                # Process each step
                for idx, step in enumerate(steps):
                    label = step.get('judge_label', 'UNKNOWN').upper()

                    # Skip if no judge label
                    if label not in ['GOOD', 'BAD']:
                        continue

                    # Filtering logic:
                    # - If consensus_only=True: use consensus steps (typically GOOD)
                    # - Always include BAD steps (important for learning to reject)
                    is_consensus = step.get('consensus', False)
                    is_bad = (label == 'BAD')

                    if consensus_only and not (is_consensus or is_bad):
                        continue

                    # Get previous steps for context
                    previous_steps = steps[:idx]

                    # Format step (no gold_answer) - returns messages format
                    messages = self.format_step_for_training(
                        question=question,
                        previous_steps=previous_steps,
                        current_step=step,
                    )

                    training_samples.append({
                        'messages': messages,  # trl conversational format
                        'question_id': trajectory['question_id'],
                        'step_num': step.get('step_num', idx + 1),
                        'label': label,
                    })
                    label_counts[label] += 1

        print(f"  Label distribution: GOOD={label_counts['GOOD']}, BAD={label_counts['BAD']}")
        return training_samples


class CriticTrainer:
    """Trainer for critic model."""

    def __init__(
        self,
        config: CriticTrainingConfig,
    ):
        self.config = config

        # Load tokenizer
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Data formatter
        self.formatter = CriticDataFormatter(self.tokenizer)

        # Model will be loaded later
        self.model = None

    def prepare_data(
        self,
        train_file: Path,
        eval_file: Optional[Path] = None,
    ) -> tuple:
        """Prepare training and evaluation datasets.

        Args:
            train_file: Path to training data (merged JSONL)
            eval_file: Path to eval data (optional)

        Returns:
            (train_dataset, eval_dataset) tuple
        """
        print(f"\nPreparing training data from: {train_file}")
        train_samples = self.formatter.prepare_training_data(
            train_file,
            consensus_only=self.config.consensus_only,
        )
        print(f"✓ Prepared {len(train_samples)} training samples")

        # Create HuggingFace Dataset
        train_dataset = Dataset.from_list(train_samples)

        eval_dataset = None
        if eval_file and eval_file.exists():
            print(f"\nPreparing eval data from: {eval_file}")
            eval_samples = self.formatter.prepare_training_data(
                eval_file,
                consensus_only=self.config.consensus_only,
            )
            print(f"✓ Prepared {len(eval_samples)} eval samples")
            eval_dataset = Dataset.from_list(eval_samples)

        return train_dataset, eval_dataset

    def load_model(self):
        """Load model with LoRA."""
        print(f"\nLoading model: {self.config.model_name}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            trust_remote_code=True,
        )

        # Prepare for LoRA
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # LoRA config
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora_target_modules,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        print("✓ Model loaded with LoRA")

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """Train the critic model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        if self.model is None:
            self.load_model()

        # SFTConfig with completion_only_loss (replaces DataCollatorForCompletionOnlyLM)
        # assistant_only_loss=True: only compute loss on assistant responses
        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="tensorboard",
            remove_unused_columns=False,
            # Note: assistant_only_loss requires chat template with {% generation %}
            # Using full sequence loss for now (trl 0.27+ compatibility)
            assistant_only_loss=False,
            max_length=self.config.max_seq_length,
        )

        # SFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        # Train
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print()

        trainer.train()

        # Save final model
        final_path = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        print(f"\n✓ Final model saved to: {final_path}")

        # Save training info
        info = {
            'config': {
                'model_name': self.config.model_name,
                'lora_r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_train_epochs,
                'batch_size': self.config.per_device_train_batch_size,
                'gradient_accumulation': self.config.gradient_accumulation_steps,
            },
            'data': {
                'num_train_samples': len(train_dataset),
                'num_eval_samples': len(eval_dataset) if eval_dataset else 0,
                'consensus_only': self.config.consensus_only,
            },
            'training': {
                'completed_at': datetime.now().isoformat(),
            }
        }

        info_path = Path(self.config.output_dir) / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"✓ Training info saved to: {info_path}")


def create_critic_trainer(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "outputs/critic_model",
    **kwargs
) -> CriticTrainer:
    """Create a critic trainer with custom config.

    Args:
        model_name: Model to fine-tune
        output_dir: Output directory
        **kwargs: Additional config parameters

    Returns:
        CriticTrainer instance
    """
    config = CriticTrainingConfig(
        model_name=model_name,
        output_dir=output_dir,
        **kwargs
    )

    return CriticTrainer(config)
