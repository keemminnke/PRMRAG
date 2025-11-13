# Usage Guide

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Create Example Data

```bash
make example
# or
python scripts/create_example_data.py
```

### 3. Run Full Pipeline

```bash
make run-pipeline
# or
python scripts/label_dataset.py \
    --config configs/default.yaml \
    --input data/raw/example_trajectories.jsonl \
    --output data/labeled/example_labeled.jsonl
```

### 4. Analyze Results

```bash
make analysis
# or
python scripts/consensus_analysis.py \
    --input data/labeled/example_labeled.jsonl \
    --output outputs/metrics/consensus_analysis.json
```

## Input Format

Your input data should be in JSONL format with the following structure:

```json
{
  "trajectory_id": "example_001",
  "question": "What is the capital of France?",
  "gold_answer": "Paris",
  "supporting_facts": [
    "Paris is the capital of France."
  ],
  "steps": [
    {
      "step_id": 0,
      "action_type": "retrieve",
      "action": "Search for: capital of France",
      "observation": "Found information...",
      "passages": ["Paris is the capital..."]
    }
  ],
  "final_answer": "Paris",
  "is_correct": true
}
```

## Configuration

Edit `configs/default.yaml` to customize:

### RPE Labeling (Small Model)

```yaml
rpe:
  model_name: "meta-llama/Llama-2-7b-hf"
  num_rollouts: 5
  threshold_good: 0.8  # P_t >= 0.8 → GOOD
  threshold_bad: 0.3   # P_t <= 0.3 → BAD
```

### Judge Labeling (Large Model)

```yaml
judge:
  model_name: "meta-llama/Llama-2-70b-hf"
  temperature: 0.3
  prompt_style: "versaprm"
  use_gold_answer: true
```

### Consensus Strategy

```yaml
consensus:
  strategy: "strict"  # strict, lenient, or balanced
  trajectory_level: true  # Filter entire trajectory if any step conflicts
  min_agreement: 0.8
```

#### Strategies:

**Strict** (default):
- Positive: RPE=GOOD + Judge=GOOD → label=1
- Negative: RPE=BAD + Judge=BAD → label=0
- Everything else → filtered

**Lenient**:
- Positive: RPE∈{GOOD, BORDERLINE} + Judge=GOOD → label=1
- Negative: RPE∈{BAD, BORDERLINE} + Judge=BAD → label=0

**Balanced**:
- Positive: RPE∈{GOOD, BORDERLINE} + Judge=GOOD → label=1
- Negative: RPE=BAD + Judge=BAD → label=0

## Output Format

### Labeled Trajectories

`data/labeled/example_labeled.jsonl` contains:

```json
{
  "trajectory": { ... },
  "rpe_labels": [
    {
      "step_id": 0,
      "rpe": 1.8,
      "label": "GOOD"
    }
  ],
  "judge_labels": [
    {
      "step_id": 0,
      "label": "GOOD",
      "reasoning": "..."
    }
  ],
  "consensus_labels": [
    {
      "step_id": 0,
      "consensus_label": 1,
      "is_consensus": true
    }
  ],
  "is_filtered": false
}
```

### Training Samples

`data/labeled/example_training_samples.jsonl` contains step-level samples ready for training:

```json
{
  "trajectory_id": "example_001",
  "step_id": 0,
  "question": "...",
  "prefix": [...],
  "action": "...",
  "observation": "...",
  "passages": [...],
  "label": 1,  # 0 or 1
  "confidence": 0.9
}
```

## Advanced Usage

### Custom Consensus Rules

Edit `configs/default.yaml`:

```yaml
consensus:
  rules:
    positive_consensus:
      rpe: ["GOOD", "BORDERLINE"]
      judge: ["GOOD"]
      label: 1
    negative_consensus:
      rpe: ["BAD"]
      judge: ["BAD"]
      label: 0
```

### Caching Labels

Save intermediate results to avoid re-running expensive labeling:

```bash
# First run - saves RPE and Judge labels
python scripts/label_dataset.py \
    --config configs/default.yaml \
    --input data/raw/trajectories.jsonl \
    --output data/labeled/labeled.jsonl

# Later - load from cache and only run consensus
python scripts/label_dataset.py \
    --config configs/default.yaml \
    --input data/raw/trajectories.jsonl \
    --output data/labeled/labeled_v2.jsonl \
    --skip-rpe --rpe-cache data/labeled/labeled_rpe.jsonl \
    --skip-judge --judge-cache data/labeled/labeled_judge.jsonl
```

### Custom Models

For RPE and Judge labeling, you can use:

1. **OpenAI/Anthropic APIs**: Set `model_name: "gpt-4"` or `model_name: "claude-3-opus"`
2. **Local models via vLLM**: Load with transformers and pass to labelers
3. **Custom endpoints**: Implement model client and pass to labelers

Example with custom model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from prmrag.labeling import RPELabeler

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

labeler = RPELabeler(config, model=model, tokenizer=tokenizer)
```

## Tips

1. **Start small**: Use `--limit 10` to test on a small subset first
2. **Monitor agreement**: High agreement (>80%) indicates good labeling quality
3. **Adjust thresholds**: If too much data is filtered, try "lenient" strategy
4. **Use trajectory-level filtering**: More conservative but higher quality
5. **Check disagreement patterns**: Use `consensus_analysis.py` to identify issues

## Troubleshooting

**Issue**: Too much data filtered

**Solution**: Try lenient strategy or lower `min_agreement`

**Issue**: Low agreement between RPE and Judge

**Solution**: Check RPE thresholds, judge prompt, or model quality

**Issue**: Out of memory

**Solution**: Reduce `batch_size` in config or use smaller model
