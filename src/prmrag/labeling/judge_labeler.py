"""LLM Judge labeler (VersaPRM style) using vLLM."""

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

        self.model_name = config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 512)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.prompt_style = config.get("prompt_style", "versaprm")
        self.use_gold_answer = config.get("use_gold_answer", True)
        self.use_supporting_facts = config.get("use_supporting_facts", True)

        # vLLM-specific settings
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.7)
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)

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
        }

        print(f"Initializing Judge model with vLLM: {self.model_name}")
        self.model_client = load_policy_model(model_config)

    def label_trajectory(self, trajectory: Trajectory) -> List[JudgeLabel]:
        """Label a trajectory using LLM judge.

        Args:
            trajectory: Trajectory to label

        Returns:
            List of JudgeLabel objects, one per step
        """
        labels = []

        for step_idx in range(len(trajectory.steps)):
            judge_label = self._judge_step(trajectory, step_idx)
            labels.append(judge_label)

        return labels

    def _judge_step(
        self,
        trajectory: Trajectory,
        step_idx: int,
    ) -> JudgeLabel:
        """Judge a specific step using LLM.

        Args:
            trajectory: Trajectory containing the step
            step_idx: Index of the step to evaluate

        Returns:
            JudgeLabel for this step
        """
        # Build prompt
        prompt = self._build_judge_prompt(trajectory, step_idx)

        # Call LLM with retry logic
        response = self._call_llm_with_retry(prompt)

        # Parse response
        label, reasoning, confidence = self._parse_judge_response(response)

        return JudgeLabel(
            step_id=step_idx,
            label=label,
            reasoning=reasoning,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "prompt_style": self.prompt_style,
            },
        )

    def _build_judge_prompt(
        self,
        trajectory: Trajectory,
        step_idx: int,
    ) -> str:
        """Build judge prompt for a step.

        Args:
            trajectory: Trajectory containing the step
            step_idx: Index of the step

        Returns:
            Formatted prompt string
        """
        if self.prompt_style == "versaprm":
            return self._build_versaprm_prompt(trajectory, step_idx)
        else:
            return self._build_default_prompt(trajectory, step_idx)

    def _build_versaprm_prompt(
        self,
        trajectory: Trajectory,
        step_idx: int,
    ) -> str:
        """Build VersaPRM-style judge prompt.

        VersaPRM evaluates whether a step is helpful for reaching the correct answer.
        """
        step = trajectory.steps[step_idx]
        prefix_steps = trajectory.get_prefix(step_idx - 1) if step_idx > 0 else []

        prompt_parts = [
            "You are an expert evaluator for question-answering systems.",
            "",
            "# Task",
            "Evaluate whether a reasoning step is GOOD or BAD for answering the question correctly.",
            "",
            "# Evaluation Criteria",
            "A step is GOOD if:",
            "- It retrieves relevant information that helps answer the question",
            "- It makes correct logical inferences",
            "- It moves toward the correct answer",
            "- The retrieved passages are relevant and useful",
            "",
            "A step is BAD if:",
            "- It retrieves irrelevant or misleading information",
            "- It makes incorrect logical inferences",
            "- It leads away from the correct answer",
            "- The passages are off-topic or unhelpful",
            "",
            f"# Question\n{trajectory.question}",
        ]

        # Add gold answer if available
        if self.use_gold_answer and trajectory.gold_answer:
            prompt_parts.extend([
                "",
                f"# Correct Answer\n{trajectory.gold_answer}",
            ])

        # Add supporting facts if available
        if self.use_supporting_facts and trajectory.supporting_facts:
            prompt_parts.extend([
                "",
                "# Supporting Facts (for reference)",
            ])
            for i, fact in enumerate(trajectory.supporting_facts, 1):
                prompt_parts.append(f"{i}. {fact}")

        # Add previous steps for context
        if prefix_steps:
            prompt_parts.extend([
                "",
                "# Previous Steps",
            ])
            for i, prev_step in enumerate(prefix_steps, 1):
                prompt_parts.append(f"Step {i}: {prev_step.action}")
                prompt_parts.append(f"Result: {prev_step.observation[:200]}...")

        # Add current step to evaluate
        prompt_parts.extend([
            "",
            "# Step to Evaluate",
            f"Action: {step.action}",
        ])

        if step.passages:
            prompt_parts.append("\nRetrieved Passages:")
            for i, passage in enumerate(step.passages, 1):
                prompt_parts.append(f"{i}. {passage}")

        prompt_parts.extend([
            f"\nObservation: {step.observation}",
            "",
            "# Your Evaluation",
            "Provide your evaluation in the following format:",
            "",
            "Reasoning: [Explain why this step is good or bad]",
            "Label: [GOOD or BAD]",
            "Confidence: [0.0 to 1.0]",
        ])

        return "\n".join(prompt_parts)

    def _build_default_prompt(
        self,
        trajectory: Trajectory,
        step_idx: int,
    ) -> str:
        """Build default judge prompt."""
        step = trajectory.steps[step_idx]

        prompt = f"""Evaluate the following reasoning step:

Question: {trajectory.question}
Gold Answer: {trajectory.gold_answer or 'N/A'}

Step {step_idx + 1}:
Action: {step.action}
Observation: {step.observation}

Is this step GOOD (helpful) or BAD (unhelpful) for answering the question?

Response format:
Reasoning: [your explanation]
Label: [GOOD or BAD]
Confidence: [0.0 to 1.0]
"""
        return prompt

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
Label: GOOD
Confidence: 0.5"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM using vLLM backend.

        Args:
            prompt: Input prompt

        Returns:
            LLM response
        """
        if self.model_client is None:
            raise RuntimeError("Model client not initialized")

        # Use vLLM PolicyModel's generate_with_chat_template
        response = self.model_client.generate_with_chat_template(
            user_message=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response

    def _parse_judge_response(self, response: str) -> tuple[str, str, float]:
        """Parse LLM judge response.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (label, reasoning, confidence)
        """
        # Default values
        label = "GOOD"
        reasoning = ""
        confidence = 0.5

        # Parse response
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            if line.startswith("Reasoning:"):
                reasoning = line.split("Reasoning:", 1)[1].strip()
            elif line.startswith("Label:"):
                label_text = line.split("Label:", 1)[1].strip().upper()
                if "BAD" in label_text:
                    label = "BAD"
                elif "GOOD" in label_text:
                    label = "GOOD"
            elif line.startswith("Confidence:"):
                try:
                    conf_text = line.split("Confidence:", 1)[1].strip()
                    confidence = float(conf_text)
                except (ValueError, IndexError):
                    confidence = 0.5

        return label, reasoning, confidence

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
