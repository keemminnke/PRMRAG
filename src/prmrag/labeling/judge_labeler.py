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

    def _parse_step_sections(self, step) -> dict:
        """Parse step content into XML tag sections.

        XML format:
        <think>reasoning</think>
        <search>query</search> or <answer>answer</answer>
        <documents>retrieved passages</documents>

        Returns:
            dict with: think, search, answer, documents, action_type
        """
        # Get full content from either 'text', 'content', or 'action' field
        full_content = getattr(step, 'text', None) or getattr(step, 'content', None) or getattr(step, 'action', '') or ''

        think = ''
        search = ''
        answer = ''
        documents = ''
        action_type = 'reason'

        # Extract XML tags
        think_match = re.search(r'<think>(.*?)</think>', full_content, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()

        search_match = re.search(r'<search>(.*?)</search>', full_content, re.DOTALL)
        if search_match:
            search = search_match.group(1).strip()
            action_type = 'search'

        answer_match = re.search(r'<answer>(.*?)</answer>', full_content, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            action_type = 'answer'

        docs_match = re.search(r'<documents>(.*?)</documents>', full_content, re.DOTALL)
        if docs_match:
            documents = docs_match.group(1).strip()

        # Fallback to step attributes if XML not found
        if not think:
            think = getattr(step, 'think', '') or getattr(step, 'thought', '') or ''
        if not documents:
            documents = getattr(step, 'observation', '') or getattr(step, 'documents', '') or ''

        return {
            'think': think,
            'search': search,
            'answer': answer,
            'documents': documents,
            'action_type': action_type,
        }

    def _build_whole_trajectory_prompt(self, trajectory: Trajectory) -> str:
        """Build prompt for evaluating all steps in one call.

        Strict Process Supervision: 검색 전략과 증거 기반 추론을 엄격히 평가.
        """
        # 1. 전체 Trajectory 구성 (XML 태그 형식)
        interaction_history = []
        for i, step in enumerate(trajectory.steps, 1):
            sections = self._parse_step_sections(step)

            step_text = f"## Step {i}\n"

            # Think (reasoning)
            if sections['think']:
                step_text += f"<think>{sections['think']}</think>\n"

            # Action (search or answer)
            if sections['action_type'] == 'search' and sections['search']:
                step_text += f"<search>{sections['search']}</search>\n"
            elif sections['action_type'] == 'answer' and sections['answer']:
                step_text += f"<answer>{sections['answer']}</answer>\n"

            # Documents (retrieved passages)
            if sections['documents']:
                step_text += f"<documents>{sections['documents']}</documents>\n"

            interaction_history.append(step_text)

        history_str = "\n\n".join(interaction_history)
        gold_answer = trajectory.gold_answer if self.use_gold_answer else "N/A"

        # Show final answer clearly
        final_answer_section = ""
        if trajectory.final_answer:
            final_answer_section = f"## Model's Final Answer\n{trajectory.final_answer}"

        # 2. Strict Process Supervisor 프롬프트 (XML 태그 형식)
        prompt = f"""You are a **Strict Process Supervisor** for an Active RAG (Retrieval-Augmented Generation) agent.
Your goal is NOT just to check if the answer is correct, but to judge whether the agent's **search strategy**, **question grounding**, and **reliance on evidence** are flawless.
You must penalize "lucky guesses" (correct answers without evidence) and reward "strategic resilience" (retrying after failure).

The trajectory uses these XML tags:
- <think>reasoning</think> - Agent's internal reasoning
- <search>query</search> - Search action with query
- <answer>final answer</answer> - Final answer submission
- <documents>passages</documents> - Retrieved documents from search

# Key Rules (Non-negotiable)
- You must NOT use your own world knowledge. Treat ANY factual claim not supported by <documents> as a hallucination.
- A "successful search" means the <documents> contains information DIRECTLY relevant to the Question's entities and required relations.
  If the <documents> is about a different entity (namesake, fictional character vs real person, wrong relation direction), it is NOT successful.

# Task
You will be given a full interaction trajectory consisting of multiple steps.
Evaluate EACH step independently (but you may use previous steps to determine whether search has succeeded) and assign GOOD or BAD.

# Evaluation Criteria (Strict Process Supervision)

**Case 0: QUESTION MISINTERPRETATION (Entity/Relation Error)**
- **BAD if:** The <think> assumes the wrong entity type/domain (e.g., fictional character vs real person), wrong relation direction (son vs father), or drifts to a namesake.
- **GOOD if:** The step maintains correct entity grounding and relation direction consistent with the Question.

**Case 1: MISSING SEARCH (The 'Overconfidence' Error)**
If the Agent uses <answer> with factual details WITHOUT any prior successful <search>:
- **BAD** unless the question is truly trivial/general knowledge.

**Case 2: Step has <search> tag**
- **GOOD if:** The query is specific, grounded in the Question (correct entities/relations), and logically derived.
  If previous search failed, the agent tries a meaningfully different strategy (synonyms, disambiguation terms, relation-focused query).
- **BAD if:** Repeats the same failed query, is too vague, targets the wrong entity/domain, or ignores needed disambiguation.
- **BAD if:** The query is derived from any unsupported assumption or hallucinated claim in <think> or prior steps.

**Case 3: Step has only <think> (Reasoning step)**
- **GOOD if:** It only plans next actions or summarizes <documents> WITHOUT introducing new factual claims.
- **BAD if:** It asserts any factual claim not explicitly supported by the <documents> (hallucination).

**Case 4: Step has <answer> tag (Final Answer)**
- **BAD if any of the following:**
  (a) The answer is "Uncertain", "Unknown", "None", "N/A", or empty (non-response).
  (b) The <think> contains uncertainty phrases like "cannot determine", "not sure", "insufficient information".
  (c) The answer is NOT logically derivable from the accumulated <documents> - the agent made a logical leap.
  (d) The agent finishes despite having insufficient evidence in <documents> (premature conclusion).
  (e) The answer contradicts information explicitly stated in <documents>.
- **GOOD if and only if:** (1) The answer is specific and definitive, (2) there is sufficient evidence in <documents>, AND (3) the answer can be logically derived from the accumulated <documents>.
- IMPORTANT: Evaluate ONLY whether the reasoning chain is logically sound. Do NOT directly compare the answer to the Ground Truth.

**Case 5: Handling Retrieval Failures**
If the previous <documents> was IRRELEVANT or EMPTY:
- **GOOD if:** The agent admits failure and performs another <search> immediately (with a changed strategy).
- **BAD if:** The agent proceeds to <answer> as if the search succeeded.

# Input Data
**Question:** {trajectory.question}
**Ground Truth Answer (for reference only - do NOT use this to judge correctness):** {gold_answer}
## Full Trajectory
{history_str}
{final_answer_section}

# Output Instructions
Return a JSON list:
```json
[
  {{"step": 1, "label": "GOOD", "reasoning": "..."}},
  ...
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
                reasoning = result.get('reasoning', '')  # No truncation
            else:
                # Default if parsing failed
                label = 'BAD'
                reasoning = 'Failed to parse response'

            labels.append(JudgeLabel(
                step_id=step_idx,
                label=label,
                reasoning=reasoning,
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
        """Label a batch of trajectories using vLLM batch processing.

        This method batches all trajectory prompts and processes them in a single
        vLLM call for much faster inference.

        Args:
            trajectories: List of trajectories
            show_progress: Whether to show progress bar

        Returns:
            List of label lists
        """
        if not trajectories:
            return []

        # Build prompts for all trajectories
        prompts = []
        num_steps_list = []
        for traj in trajectories:
            prompt = self._build_whole_trajectory_prompt(traj)
            formatted = self._format_judge_prompt_for_chat(prompt)
            prompts.append(formatted)
            num_steps_list.append(len(traj.steps))

        if show_progress:
            print(f"  Judge labeling {len(trajectories)} trajectories in batch...")

        # Batch generate with vLLM
        if hasattr(self.model_client, 'batch_generate'):
            responses = self.model_client.batch_generate(
                prompts=prompts,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        else:
            # Fallback to sequential if batch not available
            responses = []
            iterator = tqdm(prompts, desc="Judge labeling") if show_progress else prompts
            for prompt in iterator:
                response = self.model_client.generate(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                responses.append(response)

        # Parse all responses
        all_labels = []
        for response, num_steps in zip(responses, num_steps_list):
            labels = self._parse_json_response(response, num_steps)
            all_labels.append(labels)

        if show_progress:
            print(f"  ✓ Judge labeling complete")

        return all_labels


def create_judge_labeler(config: Dict[str, Any]) -> JudgeLabeler:
    """Create JudgeLabeler with vLLM backend.

    Args:
        config: Configuration with model settings

    Returns:
        JudgeLabeler instance with vLLM model
    """
    return JudgeLabeler(config)