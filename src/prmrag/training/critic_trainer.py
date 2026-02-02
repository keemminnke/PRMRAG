"""
DeepSeek-Critic Trainer (Final Optimized Version)

Key Features:
1. DeepSeek-R1 Style: Uses <think> tags to encourage CoT reasoning.
2. Loss Masking: Uses DataCollatorForCompletionOnlyLM to train ONLY on the output (Assistant), ignoring the input (User).
3. Dynamic Template: Automatically detects if the model uses <|im_start|> or <｜Assistant｜>.
4. LoRA/QLoRA: Efficient fine-tuning on GH200.
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


@dataclass
class CriticTrainingConfig:
    """Configuration for critic model training."""

    # Model settings
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # or 8B depending on exact repo

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # Training settings
    output_dir: str = "outputs/critic_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    # DeepSeek-R1 generates long chains of thought, so we need a long context.
    max_seq_length: int = 4096 

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Optimization
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True


class CriticDataFormatter:
    """Format trajectory data for DeepSeek-R1 critic training."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def format_step_for_training(
        self,
        question: str,
        previous_steps: List[Dict[str, Any]],
        current_step: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Constructs the chat messages.
        Structure:
        System: Define role.
        User: Context (Question + History + Current Step).
        Assistant: <think> Reasoning </think> Label.
        """
        
        # 1. Build User Context
        input_parts = [f"Question: {question}", ""]

        if previous_steps:
            input_parts.append("Previous Steps:")
            for i, prev_step in enumerate(previous_steps, 1):
                input_parts.append(f"Step {i}:")
                if prev_step.get('thought'):
                    input_parts.append(f"Thought: {prev_step.get('thought')}")
                
                action = prev_step.get('action', 'Unknown')
                action_input = prev_step.get('action_input', '')
                input_parts.append(f"Action: {action}[{action_input}]" if action_input else f"Action: {action}")

                if prev_step.get('observation'):
                    input_parts.append(f"Observation: {prev_step.get('observation')}")
                input_parts.append("")

        input_parts.append("Current Step to Evaluate:")
        if current_step.get('thought'):
            input_parts.append(f"Thought: {current_step.get('thought')}")

        action = current_step.get('action', 'Unknown')
        action_input = current_step.get('action_input', '')
        input_parts.append(f"Action: {action}[{action_input}]" if action_input else f"Action: {action}")

        if current_step.get('observation'):
            input_parts.append(f"Observation: {current_step.get('observation')}")

        input_parts.append("")
        input_parts.append("Task: Evaluate the quality of the Current Step. First, analyze whether the reasoning is logically sound and grounded in evidence using the <think> tag, then provide a label (GOOD/BAD).")

        user_content = "\n".join(input_parts)
        
        # 2. Build Assistant Response (DeepSeek Style)
        # We inject the reasoning into <think> tags to train the model's internal monologue.
        label = current_step.get('judge_label', 'UNKNOWN').upper()
        reasoning = current_step.get('judge_reasoning', '')
        
        # The Core Logic: Reasoning First, Label Last.
        assistant_content = f"<think>\n{reasoning}\n</think>\nLabel: {label}"

        system_content = (
            "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
            "Analyze each step's logical soundness and evidence grounding. "
            "First, provide a reasoning explanation inside <think> tags, and then output the final label (GOOD/BAD)."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

    def prepare_training_data(self, data_file: Path) -> List[Dict[str, Any]]:
        training_samples = []
        label_counts = {'GOOD': 0, 'BAD': 0}

        print(f"Reading data from {data_file}...")
        with open(data_file) as f:
            for line in f:
                try:
                    trajectory = json.loads(line)
                    question = trajectory['question']
                    steps = trajectory['steps']

                    for idx, step in enumerate(steps):
                        label = step.get('judge_label', 'UNKNOWN').upper()
                        # Only train on valid labels
                        if label not in ['GOOD', 'BAD']:
                            continue

                        messages = self.format_step_for_training(
                            question=question,
                            previous_steps=steps[:idx],
                            current_step=step,
                        )

                        training_samples.append({
                            'messages': messages,  # SFTTrainer expects 'messages' column
                            'question_id': trajectory.get('question_id', trajectory.get('trajectory_id')),
                            'step_num': step.get('step_num', idx + 1),
                            'label': label,
                        })
                        label_counts[label] += 1
                except json.JSONDecodeError:
                    continue

        print(f"  Data Stats: GOOD={label_counts['GOOD']}, BAD={label_counts['BAD']}")
        return training_samples


class CriticTrainer:
    """Trainer for critic model with auto-masking and LoRA."""

    def __init__(self, config: CriticTrainingConfig):
        self.config = config
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.formatter = CriticDataFormatter(self.tokenizer)
        self.model = None

    def _get_response_template(self) -> str:
        """
        Dynamically detects the start token for the assistant's response.
        Crucial for DataCollatorForCompletionOnlyLM to work correctly with different models.
        """
        # 1. Probe the tokenizer with a dummy message
        dummy_messages = [
            {"role": "user", "content": "TEST"},
            {"role": "assistant", "content": ""} # Empty content to find the boundary
        ]
        
        # Apply template but do not tokenize yet
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                dummy_messages, 
                tokenize=False, 
                add_generation_prompt=True 
            )
        except Exception as e:
            print(f"Warning: Could not apply chat template automatically. Using default Qwen. Error: {e}")
            return "<|im_start|>assistant\n"

        # 2. Check for known patterns
        if "<|im_start|>assistant" in prompt_str:
            print("Detected Template: Qwen/DeepSeek-Distill (<|im_start|>assistant)")
            return "<|im_start|>assistant\n"
        elif "<｜Assistant｜>" in prompt_str:
            print("Detected Template: DeepSeek-R1 Original (<｜Assistant｜>)")
            return "<｜Assistant｜>"
        elif "Assistant:" in prompt_str:
             print("Detected Template: Generic (Assistant:)")
             return "Assistant:"
        
        # Default fallback
        print(f"Warning: Unknown template structure. Ends with: {prompt_str[-20:]}")
        return "<|im_start|>assistant\n"

    def prepare_data(self, train_file: Path, eval_file: Optional[Path] = None) -> tuple:
        print(f"\nPreparing training data from: {train_file}")
        train_samples = self.formatter.prepare_training_data(train_file)
        print(f"✓ Prepared {len(train_samples)} training samples")
        train_dataset = Dataset.from_list(train_samples)

        eval_dataset = None
        if eval_file and eval_file.exists():
            print(f"\nPreparing eval data from: {eval_file}")
            eval_samples = self.formatter.prepare_training_data(eval_file)
            print(f"✓ Prepared {len(eval_samples)} eval samples")
            eval_dataset = Dataset.from_list(eval_samples)

        return train_dataset, eval_dataset

    def load_model(self):
        print(f"\nLoading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" # Highly recommended for GH200
        )

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora_target_modules,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        print("✓ Model loaded with LoRA")

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        if self.model is None:
            self.load_model()

        # [CRITICAL] Setup Data Collator for Loss Masking
        # This ensures we only calculate loss on the Assistant's response (Reasoning + Label),
        # not on the User's prompt.
        response_template = self._get_response_template()
        
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

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
            save_total_limit=2,
            max_seq_length=self.config.max_seq_length,
            report_to="tensorboard",
            remove_unused_columns=True,
        )

        print("\n" + "=" * 70)
        print(f"STARTING TRAINING with template: {repr(response_template)}")
        print("=" * 70)
        
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=collator, # Apply masking
        )

        trainer.train()

        # Save Final Model
        final_path = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        print(f"\n✓ Final model saved to: {final_path}")
        
        # Save info
        info = {
            'config': {
                'model_name': self.config.model_name,
                'lora_r': self.config.lora_r,
                'max_seq_length': self.config.max_seq_length,
            },
            'training': {
                'completed_at': datetime.now().isoformat(),
                'samples': len(train_dataset)
            }
        }
        with open(Path(self.config.output_dir) / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)


def create_critic_trainer(model_name: str, output_dir: str, **kwargs) -> CriticTrainer:
    config = CriticTrainingConfig(model_name=model_name, output_dir=output_dir, **kwargs)
    return CriticTrainer(config)

if __name__ == "__main__":
    # Example usage
    trainer = create_critic_trainer(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        output_dir="outputs/critic"
    )
    
    # Assuming data exists
    train_file = Path("data/merged_consensus_data.jsonl")
    if train_file.exists():
        train_ds, _ = trainer.prepare_data(train_file)
        trainer.train(train_ds)
