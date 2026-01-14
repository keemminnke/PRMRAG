"""LLM Judge labeler (VersaPRM style) using vLLM."""

import json
import re
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .base_labeler import BaseLabeler
from ..data.schemas import Trajectory, JudgeLabel
from ..data.processors import TrajectoryProcessor


class JudgeLabeler(BaseLabeler):
    """LLM Judge labeler using large language model evaluation.

    Following VersaPRM style:
    - Uses large LLM (e.g., 70B+) to judge step quality
    - Provides gold answer and supporting facts as context
    - Evaluates each step's contribution to reaching correct answer
    - Returns GOOD/BAD label with reasoning

    Now uses vLLM backend for fast inference.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_client=None,
    ):
        """Initialize Judge labeler.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Name of judge model
                - temperature: Sampling temperature
                - max_tokens: Max tokens for generation
                - max_retries: Max retry attempts
                - retry_delay: Delay between retries
                - prompt_style: Prompt template style
                - use_gold_answer: Whether to provide gold answer
                - use_supporting_facts: Whether to provide supporting facts
                - gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.7)
                - tensor_parallel_size: Number of GPUs for vLLM (default: 1)
            model_client: Pre-configured model client (optional, uses vLLM PolicyModel if None)
        """
        super().__init__(config)

        self.model_name = config.get("model_name", "Qwen/QwQ-32B")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2048)  # Increased for QwQ-32B long reasoning
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.prompt_style = config.get("prompt_style", "versaprm")
        self.use_gold_answer = config.get("use_gold_answer", True)
        self.use_supporting_facts = config.get("use_supporting_facts", True)

        # vLLM-specific settings
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.8)
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.max_model_len = config.get("max_model_len", 32768)

        self.model_client = model_client
        self.processor = TrajectoryProcessor()

        # Initialize vLLM model if no client provided
        if self.model_client is None:
            self._initialize_vllm_model()

    def _initialize_vllm_model(self):
        """Initialize vLLM model for judge labeling."""
        from ..models import load_policy_model

        model_config = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'tensor_parallel_size': self.tensor_parallel_size,
            'max_model_len': self.max_model_len,
        }

        print(f"Initializing Judge model with vLLM: {self.model_name}")
        self.model_client = load_policy_model(model_config)

    def label_trajectory(self, trajectory: Trajectory) -> List[JudgeLabel]:
        """Label a trajectory using LLM judge (batch mode - all steps at once).

        Args:
            trajectory: Trajectory to label

        Returns:
            List of JudgeLabel objects, one per step
        """
        # Build prompt for whole trajectory
        prompt = self._build_whole_trajectory_prompt(trajectory)

        # Call LLM once for all steps
        response = self._call_llm_with_retry(prompt)

        # Parse JSON response to get labels for all steps
        labels = self._parse_json_response(response, len(trajectory.steps))

        return labels

    def _build_whole_trajectory_prompt(self, trajectory: Trajectory) -> str:
        """Build prompt for evaluating all steps in one call.

        Strict Process Supervision: 검색 전략과 증거 기반 추론을 엄격히 평가.
        """
        # 1. 전체 Trajectory 구성
        interaction_history = []
        for i, step in enumerate(trajectory.steps, 1):
            step_text = f"## Step {i}\n"
            step_text += f"**Action:** {step.action}\n"
            if "Finish" in str(step.action) and trajectory.final_answer:
                step_text += f"**Model's Final Prediction:** {trajectory.final_answer}\n"
            if step.observation and step.observation.strip():
                step_text += f"**Observation:** {step.observation}\n"
            else:
                step_text += "**Observation:** (No information retrieved)\n"
            interaction_history.append(step_text)

        history_str = "\n\n".join(interaction_history)
        gold_answer = trajectory.gold_answer if self.use_gold_answer else "N/A"

        # 2. Strict Process Supervisor 프롬프트
        prompt = f"""You are a **Strict Process Supervisor** for an Active RAG (Retrieval-Augmented Generation) agent.
Your goal is NOT just to check if the answer is correct, but to judge whether the agent's **search strategy** and **reliance on evidence** are flawless.
You must penalize "lucky guesses" (correct answers without evidence) and reward "strategic resilience" (retrying after failure).

# Task
You will be given a full interaction trajectory consisting of multiple steps.
Your task is to **evaluate EACH step independently** based on the criteria below and assign a label (GOOD or BAD).

# Evaluation Criteria (Strict Process Supervision)

**Case 0: MISSING SEARCH (The 'Overconfidence' Error)**
If the Agent chooses to **Answer (Finish)** or **Reason** using internal knowledge WITHOUT any prior successful Search:
- **BAD if:** The question asks for specific factual details (e.g., obscure names, dates, statistics) that require verification, but the Agent skips 'Search' and relies on internal memory.
- **GOOD if:** The question is trivial or general knowledge where search is genuinely unnecessary.

**Case 1: Action is SEARCH**
- **GOOD if:** The query is specific, relevant, and logically derived. If previous search failed, the agent tries a DIFFERENT strategy.
- **BAD if:** Repeats the exact same failed query, or query is too vague to be useful.

**Case 2: Action is FINISH / REASON (The 'Answering' Phase)**
- **GOOD if:** The answer is strictly derived from the provided Observation with sufficient evidence.
- **BAD if:** The Observation is EMPTY or IRRELEVANT, but the agent answers anyway using internal knowledge (Parametric Shortcut). Even if factually correct, this is BAD in RAG context.

**Case 3: Handling Retrieval Failures**
If the previous Observation was IRRELEVANT or EMPTY:
- **GOOD if:** The agent admits failure and immediately performs another **Search** action.
- **BAD if:** The agent ignores the bad result and proceeds to answer (Lazy Finish).

# Input Data

## Question
{trajectory.question}

## Correct Answer (Ground Truth)
{gold_answer}

## Full Trajectory
{history_str}

# Output Instructions
1. Analyze the trajectory step by step.
2. For each step, check specifically for hallucinations against the Observation.
3. Output the result in a **JSON list** format.

Example Output Format:
```json
[
  {{"step": 1, "label": "GOOD", "reasoning": "The search query is specific and relevant."}},
  {{"step": 2, "label": "BAD", "reasoning": "The model claims facts not in the Observation. This is a hallucination."}}
]
```

Evaluate all {len(trajectory.steps)} steps now."""

        return prompt

    def _parse_json_response(self, response: str, num_steps: int) -> List[JudgeLabel]:
        """Parse JSON response from whole-trajectory evaluation.

        Args:
            response: Raw LLM response containing JSON
            num_steps: Expected number of steps

        Returns:
            List of JudgeLabel objects
        """
        labels = []

        # Extract JSON from response (handle ```json ... ``` blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON array
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = None

        parsed_results = []
        if json_str:
            try:
                parsed_results = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common issues
                try:
                    # Remove trailing commas
                    fixed = re.sub(r',\s*]', ']', json_str)
                    fixed = re.sub(r',\s*}', '}', fixed)
                    parsed_results = json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        # Build labels from parsed results
        for step_idx in range(num_steps):
            # Find matching result
            result = None
            for r in parsed_results:
                if r.get('step') == step_idx + 1:
                    result = r
                    break

            if result:
                label = result.get('label', 'BAD').upper()
                if 'GOOD' in label:
                    label = 'GOOD'
                else:
                    label = 'BAD'
                reasoning = result.get('reasoning', '')[:200]
            else:
                # Default if parsing failed
                label = 'BAD'
                reasoning = 'Failed to parse response'

            labels.append(JudgeLabel(
                step_id=step_idx,
                label=label,
                reasoning=reasoning,
                confidence=1.0,
                metadata={
                    "model_name": self.model_name,
                    "prompt_style": "whole_trajectory",
                },
            ))

        return labels

    def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with retry logic.

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt)
                return response
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"All retries failed: {e}")
                    # Return a default response on failure
                    return """Reasoning: Unable to evaluate due to model error.
Label: GOOD"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM using vLLM backend.

        Args:
            prompt: Input prompt

        Returns:
            LLM response
        """
        if self.model_client is None:
            raise RuntimeError("Model client not initialized")

        # Format Judge prompt with chat template
        # QwQ-32B is a chat model and needs proper formatting
        formatted_prompt = self._format_judge_prompt_for_chat(prompt)

        # Use raw generate() with formatted chat template
        response = self.model_client.generate(
            prompt=formatted_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response

    def _format_judge_prompt_for_chat(self, user_message: str) -> str:
        """Format Judge prompt for chat model (e.g., QwQ-32B).

        Args:
            user_message: VersaPRM-style judge prompt

        Returns:
            Formatted prompt with chat template
        """
        if hasattr(self.model_client, 'tokenizer') and hasattr(self.model_client.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": user_message}
            ]
            return self.model_client.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to manual Qwen format
            return f"""<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

    def label_batch(
        self,
        trajectories: List[Trajectory],
        show_progress: bool = True,
    ) -> List[List[JudgeLabel]]:
        """Label a batch of trajectories with progress bar.

        Args:
            trajectories: List of trajectories
            show_progress: Whether to show progress bar

        Returns:
            List of label lists
        """
        labels = []

        iterator = tqdm(trajectories, desc="Judge labeling") if show_progress else trajectories

        for traj in iterator:
            traj_labels = self.label_trajectory(traj)
            labels.append(traj_labels)

        return labels


def create_judge_labeler(config: Dict[str, Any]) -> JudgeLabeler:
    """Create JudgeLabeler with vLLM backend.

    Args:
        config: Configuration with model settings

    Returns:
        JudgeLabeler instance with vLLM model
    """
    return JudgeLabeler(config)
