# Step Forcing for Structured Solution Generation

## Overview

This document explains the **step forcing** approach implemented in PRMRAG for generating structured, semantically meaningful reasoning steps.

## Motivation

### Problem with "\n\n" Splitting

Traditional approaches split reasoning into steps using arbitrary delimiters like "\n\n". This has several issues:

1. **No semantic boundaries**: Splits occur at arbitrary points, not logical reasoning boundaries
2. **Overly fine-grained**: A single reasoning step might be split into multiple fragments
3. **Inconsistent structure**: Different models produce different splitting patterns

### Solution: Step Forcing

Instead of post-hoc splitting, we **force** the model to generate structured steps during generation:

```
Step 1: [Model generates first reasoning step]
Step 2: [Model generates second reasoning step]
...
Step T: [Model generates final reasoning step]
```

This approach:
- ✅ Ensures semantic step boundaries
- ✅ Produces consistent structure across models
- ✅ Makes step-by-step evaluation more meaningful
- ✅ Enables better MC estimation per step

## Implementation

### 1. Step Forcing Prompts

The `StepForcingPrompt` class builds prompts that guide the model:

```python
from prmrag.generation import StepForcingPrompt

prompt_builder = StepForcingPrompt()

# Initial prompt (Step 1)
prompt = prompt_builder.build_initial_prompt(
    question="What is the capital of France?",
    context=""
)
# Output:
# Question: What is the capital of France?
#
# Please solve this problem step by step. Start with:
#
# Step 1:

# Continuation prompt (Step 2, 3, ...)
prompt = prompt_builder.build_continuation_prompt(
    question="What is the capital of France?",
    previous_steps=[step1, step2],
    context=""
)
# Output:
# Question: What is the capital of France?
#
# Step 1: France is a country in Western Europe...
# Step 2: The capital city is determined by...
#
# Step 3:
```

### 2. Step Parser

The `StepParser` class extracts structured steps from model responses:

```python
from prmrag.generation import StepParser

parser = StepParser()

# Parse multiple steps
text = """
Step 1: First, let's identify the problem.
Step 2: Next, we gather relevant information.
Step 3: Finally, we reach a conclusion.
"""
steps = parser.parse_steps(text)
# Returns: [ForcedStep(1, "First, let's..."), ForcedStep(2, "Next, we..."), ...]

# Parse single step response
response = "First, let's identify the problem."
content = parser.parse_single_step_response(response, expected_step_num=1)
# Returns: "First, let's identify the problem."
```

### 3. Integration with Adaptive Generator

The `AdaptiveTrajectoryGenerator` automatically uses step forcing when enabled:

```python
config = {
    'use_step_forcing': True,  # Enable step forcing
    'num_rollouts': 5,
    'delta': 0.1,
    'epsilon': 0.1,
    # ...
}

generator = AdaptiveTrajectoryGenerator(
    policy_model=model,
    retriever=retriever,
    config=config
)

# Generate trajectory with forced steps
trajectory = generator.generate_trajectory(
    question="What is the capital of France?",
    gold_answer="Paris"
)

# Each step has structured format
for step in trajectory.steps:
    print(step.text)  # "Step 1: France is a country..."
    print(step.content)  # "France is a country..." (without prefix)
```

## Multi-Path Sampling

Following the approach from the paper, we sample multiple solution paths per problem:

```python
from prmrag.generation import MultiPathSampler

sampler = MultiPathSampler(
    max_paths=2048,           # Maximum paths to sample
    min_correct_paths=1,      # Minimum correct paths required
    min_incorrect_paths=1,    # Minimum incorrect paths required
)

# Sampling loop
num_sampled = 0
num_correct = 0
num_incorrect = 0

while sampler.should_continue_sampling(num_sampled, num_correct, num_incorrect):
    # Generate trajectory
    trajectory = generator.generate_trajectory(question, gold_answer)

    num_sampled += 1
    if trajectory.is_correct:
        num_correct += 1
    else:
        num_incorrect += 1

# Check if problem should be discarded
if sampler.should_discard_problem(num_sampled, num_correct, num_incorrect):
    print("Discarding problem - insufficient correct/incorrect paths")
```

## Configuration

### Enable Step Forcing

In `configs/adaptive_generation.yaml`:

```yaml
adaptive:
  # Step forcing (structured generation)
  use_step_forcing: true  # Force "Step N:" format for semantic boundaries

  # Multi-path sampling
  max_paths: 2048      # Maximum paths to sample per problem
  min_correct_paths: 1  # Minimum correct paths required
  min_incorrect_paths: 1  # Minimum incorrect paths required
```

### Disable Step Forcing (Legacy Mode)

If you want to use the old "\n\n" splitting approach:

```yaml
adaptive:
  use_step_forcing: false  # Use legacy splitting
```

## Benefits

### 1. Better MC Estimation

With semantic step boundaries, MC estimation is more accurate:

```python
# Step 1: "France is a country in Western Europe"
mc_before = 0.6
mc_after = 0.85  # Clear improvement
rpe = 0.85 / 0.6 = 1.42  # GOOD step

# Step 2: "Paris is the capital"
mc_before = 0.85
mc_after = 0.95  # Continued improvement
rpe = 0.95 / 0.85 = 1.12  # GOOD step
```

### 2. Better Consensus Labeling

Judges can evaluate semantic reasoning steps instead of arbitrary fragments:

```python
# Good: Semantic step
"Step 1: France is a country in Western Europe with a long history."

# Bad: Arbitrary fragment (from "\n\n" split)
"France is a"  # Incomplete thought
```

### 3. Better Training Data

PRM training benefits from semantically meaningful steps:

```python
# Training sample
state = "Question: What is the capital of France?\nStep 1: France is a country..."
label = 1  # GOOD - semantically complete reasoning
```

## Models

### Policy Model (MC Estimation)

**Qwen2.5-7B-Instruct** is used for:
- Generating CoT steps with step forcing
- MC rollouts for RPE estimation
- Small, efficient, good at structured generation

```yaml
policy_model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  temperature: 0.8
  top_p: 0.95
```

### Judge Model (Label Verification)

**GPT-20B** is used for:
- Verifying step quality
- VersaPRM-style judgments
- High-quality label signals

```yaml
judge:
  model_name: "gpt-20b"
  api_provider: "openai"
  temperature: 0.3
```

## Example Output

### With Step Forcing (Structured)

```json
{
  "question": "What is the capital of France?",
  "steps": [
    {
      "step_id": 0,
      "text": "Step 1: France is a country in Western Europe with Paris as its capital.",
      "content": "France is a country in Western Europe with Paris as its capital.",
      "step_type": "cot",
      "mc_before": 0.6,
      "mc_after": 0.92,
      "rpe": 1.53
    },
    {
      "step_id": 1,
      "text": "Step 2: Therefore, the answer is Paris.",
      "content": "Therefore, the answer is Paris.",
      "step_type": "cot",
      "mc_before": 0.92,
      "mc_after": 0.98,
      "rpe": 1.07
    }
  ],
  "final_answer": "Paris"
}
```

### Without Step Forcing (Legacy)

```json
{
  "question": "What is the capital of France?",
  "steps": [
    {
      "text": "France is a country in Western Europe",
      "mc_before": 0.6,
      "mc_after": 0.75
    },
    {
      "text": "with Paris",
      "mc_before": 0.75,
      "mc_after": 0.82
    },
    {
      "text": "as its capital.",
      "mc_before": 0.82,
      "mc_after": 0.95
    }
  ]
}
```

Notice how step forcing produces semantically complete reasoning units, while legacy splitting creates fragments.

## References

This implementation is based on the step forcing approach described in:

> "Solution Generation with Step Forcing. We utilize the 7.5K problems from the training set of the MATH dataset as the problem set. Since using "\n\n" for step division does not consider the semantics of each step and may result in overly fine-grained division, we apply a step forcing approach to generate solutions."

## Usage Example

```python
from prmrag.generation import (
    AdaptiveTrajectoryGenerator,
    StepForcingPrompt,
    StepParser,
    MultiPathSampler
)

# Initialize components
config = {
    'use_step_forcing': True,
    'max_paths': 2048,
    'num_rollouts': 5,
    'delta': 0.1,
    'epsilon': 0.1,
}

generator = AdaptiveTrajectoryGenerator(
    policy_model=qwen_7b,
    retriever=wikipedia_retriever,
    config=config
)

# Generate structured trajectory
trajectory = generator.generate_trajectory(
    question="What is the capital of France?",
    gold_answer="Paris"
)

# Each step has semantic boundaries
for step in trajectory.steps:
    print(f"{step.text}")
    print(f"  RPE: {step.rpe:.3f}")
    print(f"  Type: {step.step_type}")
```

## Conclusion

Step forcing is a simple but powerful technique that:
1. Produces semantically meaningful reasoning steps
2. Improves MC estimation accuracy
3. Enables better consensus labeling
4. Results in higher-quality training data for PRMs

By forcing the model to generate structured steps during inference, we ensure that each step represents a complete reasoning unit rather than an arbitrary text fragment.
