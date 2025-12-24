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
            'max_model_len': 40960,
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
                "# Previous Steps (Context)",
            ])
            for i, prev_step in enumerate(prefix_steps, 1):
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"{prev_step.action}")
                if prev_step.observation:
                    # No truncation - Judge needs full context to detect hallucinations
                    prompt_parts.append(f"Observation: {prev_step.observation}")
                prompt_parts.append("")

        # Add current step to evaluate
        # This step's Observation is CRITICAL for evaluation
        prompt_parts.extend([
            "",
            "# Current Step to Evaluate",
            step.action,  # Contains "Thought: ...\nAction: ..."
        ])

        # Show retrieved passages (Observation) for current step only
        if step.observation and step.observation.strip():
            prompt_parts.append("\nObservation (Retrieved Information):")
            prompt_parts.append(step.observation)
        elif step.passages and step.passages[0]:
            # Fallback: if observation is empty but passages exist
            prompt_parts.append("\nObservation (Retrieved Information):")
            for i, passage in enumerate(step.passages, 1):
                if passage and passage.strip():
                    prompt_parts.append(f"{i}. {passage}")

        prompt_parts.extend([
            "",
            "# Your Evaluation Task",
            "Please think step by step to evaluate this interaction.",
            "",
            "**CRITICAL RULE: Even if the final answer matches the Correct Answer, you MUST label the step as BAD if:**",
            "- The Model claims facts (names, dates, numbers) that are NOT present in the Observation",
            "- The Model makes inferences not supported by the retrieved information",
            "- The Model hallucinates or fabricates information",
            "",
            "Evaluation Steps:",
            "1. First, carefully read the 'Observation' and identify what factual information it actually contains.",
            "2. Second, check if the Model's 'Thought' or 'Action' makes claims beyond what the Observation supports.",
            "3. Third, verify if the step moves toward the Correct Answer using ONLY the information in Observation.",
            "4. Finally, determine if the step is GOOD or BAD.",
            "",
            "Output Format:",
            "Thinking Process:",
            "[Write down your step-by-step analysis here. Be verbose and critical.]",
            "",
            "Final Verification:",
            "Reasoning: [Summary of your judgment - explain if hallucination was detected]",
            "Label: [GOOD or BAD]"
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

IMPORTANT: Output ONLY the following two lines:
Reasoning: [One sentence, max 50 words]
Label: [GOOD or BAD]
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

    def _parse_judge_response(self, response: str) -> tuple[str, str, float]:
        """Parse LLM judge response.

        QwQ-32B generates long chain-of-thought reasoning before final judgment.
        This parser extracts ONLY the final structured output (last occurrence).

        Args:
            response: Raw LLM response (may contain long CoT)

        Returns:
            Tuple of (label, reasoning, confidence)
        """
        import re

        # Default values - CRITICAL: BAD instead of GOOD to avoid false positives
        label = "BAD"  # Conservative: parsing failure = don't trust the step
        reasoning = ""
        confidence = 1.0  # Fixed confidence since we removed it from prompt

        # Strategy: Parse from END to get final structured output
        # QwQ often outputs long reasoning, then "Reasoning: ... Label: ..." at end
        lines = response.strip().split("\n")

        # Regex patterns for flexible matching (handles **Label:**, Final Label:, etc.)
        label_pattern = re.compile(r'(?:Final\s+)?Label:\s*\*?\*?([A-Z]+)\*?\*?', re.IGNORECASE)
        reasoning_pattern = re.compile(r'Reasoning:\s*(.+)', re.IGNORECASE)

        # Reverse iterate to find LAST occurrence of each field
        for line in reversed(lines):
            line = line.strip()

            # Find last Label using regex
            if not label or label == "BAD":  # Only override default if we find valid label
                match = label_pattern.search(line)
                if match:
                    found_label = match.group(1).upper()
                    if "GOOD" in found_label:
                        label = "GOOD"
                    elif "BAD" in found_label:
                        label = "BAD"

            # Find last Reasoning using regex
            if reasoning == "":
                match = reasoning_pattern.search(line)
                if match:
                    reasoning = match.group(1).strip()
                    # If found both, we're done
                    if label in ["GOOD", "BAD"]:
                        break

        # Smart truncate: keep full if short, otherwise truncate at sentence boundary
        max_length = 200
        if len(reasoning) > max_length:
            # Try to find last sentence boundary before max_length
            truncated = reasoning[:max_length]
            # Look for sentence endings: ., !, ?
            last_period = max(truncated.rfind('. '), truncated.rfind('.'))
            last_exclaim = truncated.rfind('!')
            last_question = truncated.rfind('?')
            last_boundary = max(last_period, last_exclaim, last_question)

            if last_boundary > max_length // 2:  # Only use if boundary is past halfway
                reasoning = reasoning[:last_boundary + 1]
            else:
                reasoning = truncated + "..."

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
