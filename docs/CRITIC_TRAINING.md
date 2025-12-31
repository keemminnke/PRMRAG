# Critic Model Training Guide

## Overview

This guide explains how to train a Critic model for step-level evaluation using consensus-filtered Judge data.

The Critic model learns to:
1. **Generate Rationale** (`v`): Step-by-step reasoning for evaluation
2. **Predict Verdict** (`r`): Final judgment (GOOD/BAD)

Following the loss function from the paper:

```
L_PRM = Σ log P(v_j | x, s_t, v_<j)           (Equation 6)
L_LM  = log P(r | x, s_t, v)                   (Equation 7)
L_critic = -E[L_PRM + λ × L_LM]                (Equation 8)
```

Where:
- `x`: Question
- `s_t`: Current step (thought + action + observation)
- `v`: Verdict tokens (GOOD/BAD)
- `r`: Rationale tokens (reasoning)
- `λ`: Weight (default: 1.0)

---

## Implementation Details

### Data Format

**Input** (User instruction):
```
### User:
Question: Which magazine was started first?
Gold Answer: Arthur's Magazine

Previous Steps:
Step 1:
Thought: I need to find founding years...
Action: Search[query="Arthur's Magazine founding"]
Observation: [1] Founded in 1844...

Current Step to Evaluate:
Thought: Based on document [1]...
Action: Finish[answer="Arthur's Magazine"]
Observation: [Result]
```

**Output** (Critic response):
```
### Critic:
Reasoning: The step correctly uses information from document [1] (founded 1844) without hallucination. The conclusion is well-grounded.
Label: GOOD
```

**Note**: We use **Reasoning → Label** order (CoT style):
- Model generates thoughtful reasoning first
- Then makes final judgment based on reasoning
- This improves consistency and rationale quality

### Loss Masking

We use `DataCollatorForCompletionOnlyLM` to compute loss **only** on the Critic response:

```python
collator = DataCollatorForCompletionOnlyLM(
    response_template="### Critic:\n",
    tokenizer=tokenizer,
)
```

This ensures:
- ✅ User instruction tokens: **NO loss** (labels = -100)
- ✅ Critic response tokens: **Full loss** (learns to generate)

### Why This Implements Equation 8

1. **Autoregressive Generation** (CoT Style):
   ```
   P(Reasoning, Label | x, s_t) = P(Reasoning | x, s_t) × P(Label | x, s_t, Reasoning)
   ```

2. **Token-level Factorization**:
   - `Reasoning` tokens: `P(r_1 | x, s_t) × P(r_2 | x, s_t, r_1) × ...`
   - `Label` tokens: `P(v_1 | x, s_t, r) × P(v_2 | x, s_t, r, v_1) × ...`

3. **Single Loss**:
   ```python
   loss = -Σ log P(token_i | prefix)  # Standard causal LM loss
   ```
   Naturally combines Equations 6 & 7 because the model generates `Reasoning` then `Label` sequentially.

   **Key Advantage**: Label is conditioned on Reasoning, improving consistency.

4. **Lambda Weighting**:
   - Default λ=1: Equal weight to both tasks
   - If needed, can modify loss weights (requires custom loss function)

---

## Usage

### 1. Basic Training (Test with 5 questions)

```bash
python scripts/train_critic_model.py \
  --train-data outputs/test_judge_5q_newdata_merged_clean.jsonl \
  --output-dir outputs/critic_model_test \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --num-epochs 3 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --consensus-only
```

**Expected Output**:
- Training samples: ~11 (73.3% consensus rate)
- Training time: ~5-10 minutes (with LoRA)
- Model size: ~200MB (LoRA adapters only)

### 2. Full Training (After Judge Labeling 695 Trajectories)

```bash
python scripts/train_critic_model.py \
  --train-data outputs/hybrid_1000q_medium_fulltext/merged_all_judge_results_clean.jsonl \
  --eval-data outputs/hybrid_1000q_medium_fulltext/eval_judge_results_clean.jsonl \
  --output-dir outputs/critic_model_full \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --num-epochs 3 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --max-seq-length 4096
```

**Estimated**:
- Training samples: ~1,500 (assuming 70% consensus, ~2 steps per trajectory)
- Training time: ~2-4 hours
- GPU memory: ~40GB (with bf16 + gradient checkpointing)

### 3. Custom LoRA Configuration

```bash
python scripts/train_critic_model.py \
  --train-data outputs/data.jsonl \
  --output-dir outputs/critic_custom \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.1 \
  --learning-rate 1e-4
```

---

## Configuration Options

### Model Settings
```python
model_name = "Qwen/Qwen2.5-7B-Instruct"  # Base model
```

### LoRA Settings
```python
lora_r = 16                # Rank (higher = more parameters)
lora_alpha = 32            # Scaling factor
lora_dropout = 0.05        # Dropout rate
lora_target_modules = [    # Which layers to adapt
    "q_proj", "v_proj",
    "k_proj", "o_proj"
]
```

### Training Settings
```python
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4     # Effective batch = 4 × 4 = 16
learning_rate = 2e-4
warmup_ratio = 0.1
max_seq_length = 2048               # Increase to 4096 for long contexts
```

### Data Settings
```python
consensus_only = True               # Only use consensus steps
                                    # (RPE and Judge agree)
```

---

## Output Structure

```
outputs/critic_model/
├── checkpoint-500/              # Intermediate checkpoints
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── final_model/                 # Final trained model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── training_info.json           # Training metadata
├── runs/                        # TensorBoard logs
└── ...
```

---

## Using Trained Model

### 1. Load Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "outputs/critic_model/final_model"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

### 2. Evaluate Step

```python
from prmrag.training.critic_trainer import CriticDataFormatter

# Format input
formatter = CriticDataFormatter(tokenizer)
prompt = formatter.format_step_for_training(
    question="Which magazine was started first?",
    gold_answer="Arthur's Magazine",
    previous_steps=[...],
    current_step={...}
)

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Parse response
# Expected format:
# ### Critic:
# Reasoning: The step correctly uses...
# Label: GOOD
```

### 3. Merge LoRA into Base Model (Optional)

```python
# Merge for faster inference
merged_model = model.merge_and_unload()
merged_model.save_pretrained("outputs/critic_model_merged")
```

---

## Training Tips

### 1. Start Small
- Test with 5-10 questions first
- Verify data format is correct
- Check loss decreases

### 2. Monitor Consensus Rate
- High consensus (>70%): Good quality data
- Low consensus (<50%): May need Judge calibration

### 3. Adjust Sequence Length
- Average step observation: ~6,000 chars
- Recommended `max_seq_length`:
  - Test: 2048
  - Full: 4096
  - Long context: 8192

### 4. Memory Management
```bash
# If OOM:
--batch-size 2 \
--gradient-accumulation 8 \
--gradient-checkpointing
```

### 5. Learning Rate Scheduling
- Default: Linear warmup (10%) + decay
- For large datasets: Consider cosine schedule
- For small datasets: Constant LR may work better

---

## Evaluation

### 1. Check Training Loss

```bash
tensorboard --logdir outputs/critic_model/runs
```

**Expected**:
- Initial loss: ~2.5-3.0
- Final loss: ~0.5-1.0
- Smooth decrease (no overfitting)

### 2. Qualitative Evaluation

Sample predictions:
```python
# Generate on test set
for sample in test_dataset:
    prediction = model.generate(sample['text'])
    print(f"Ground truth: {sample['label']}")
    print(f"Predicted: {prediction}")
```

### 3. Quantitative Metrics

```python
# Label accuracy
correct = sum(pred == gt for pred, gt in zip(predictions, ground_truth))
accuracy = correct / len(predictions)

# F1 score (for GOOD/BAD classification)
from sklearn.metrics import f1_score
f1 = f1_score(ground_truth, predictions, pos_label="GOOD")
```

---

## Troubleshooting

### Issue: Loss not decreasing

**Possible causes**:
1. Learning rate too high/low
   - Solution: Try 1e-4 or 5e-5
2. Data format incorrect
   - Solution: Check response template matches
3. Sequence length too short
   - Solution: Increase `max_seq_length`

### Issue: OOM (Out of Memory)

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use fp16 instead of bf16
4. Reduce `max_seq_length`
5. Reduce LoRA rank

### Issue: Model predicts same label always

**Possible causes**:
1. Class imbalance
   - Solution: Check GOOD/BAD distribution
2. Overfitting
   - Solution: Reduce epochs, add dropout
3. Learning rate too high
   - Solution: Reduce to 1e-4

---

## Advanced: Custom Loss Weighting

If you need different weights for Label vs Reasoning (λ ≠ 1):

```python
from transformers import Trainer

class WeightedCriticTrainer(Trainer):
    def __init__(self, *args, lambda_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_weight = lambda_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']

        # Find "Reasoning:" position
        reasoning_token = self.tokenizer.encode("Reasoning:", add_special_tokens=False)[0]
        reasoning_pos = (inputs['input_ids'] == reasoning_token).nonzero(as_tuple=True)

        if len(reasoning_pos[0]) > 0:
            split_idx = reasoning_pos[1][0].item()

            # Verdict loss (before "Reasoning:")
            verdict_loss = F.cross_entropy(
                logits[:, :split_idx, :].reshape(-1, logits.size(-1)),
                labels[:, :split_idx].reshape(-1),
                ignore_index=-100
            )

            # Reasoning loss (after "Reasoning:")
            reasoning_loss = F.cross_entropy(
                logits[:, split_idx:, :].reshape(-1, logits.size(-1)),
                labels[:, split_idx:].reshape(-1),
                ignore_index=-100
            )

            # Weighted combination
            total_loss = verdict_loss + self.lambda_weight * reasoning_loss
        else:
            # Fallback
            total_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )

        return (total_loss, outputs) if return_outputs else total_loss
```

---

## Next Steps

After training:

1. **Evaluate on Test Set**
   - Compare with Judge (QwQ-32B) labels
   - Check consensus rate

2. **Replace Judge in Pipeline**
   ```python
   # Use trained critic instead of QwQ-32B
   critic_labeler = CriticLabeler(model_path="outputs/critic_model/final_model")
   labels = critic_labeler.label_trajectory(trajectory)
   ```

3. **Iterative Improvement**
   - Collect more data
   - Retrain with better labels
   - Bootstrap quality

4. **Deploy for Inference**
   - Merge LoRA for speed
   - Quantize to int8 for efficiency
   - Serve via vLLM

---

## References

- Paper: VersaPRM - Step-level Process Reward Models
- Library: [Hugging Face TRL](https://github.com/huggingface/trl)
- Model: [Qwen2.5](https://huggingface.co/Qwen)
- LoRA: [PEFT](https://github.com/huggingface/peft)
