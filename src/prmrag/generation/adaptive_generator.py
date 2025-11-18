"""Adaptive MC-CoT + RAG trajectory generation.

This module implements the core Stage 1 logic:
- Generate CoT steps with MC monitoring using step forcing
- Intervene with RAG when MC drops
- Track mc_before, mc_after for each step
- Support multi-path sampling (up to 2048 paths per problem)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from tqdm import tqdm
import re
import string
from collections import Counter

from ..data.schemas import ActionType
from ..retrieval import BM25Retriever
from .step_forcing import (
    ForcedStep,
    StepForcingPrompt,
    StepParser,
    MultiPathSampler,
    format_trajectory_with_steps,
)


# ============================================================================
# SQuAD-style Answer Normalization and Matching
# ============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer text using SQuAD-style normalization.

    This is the standard normalization used in RAG/QA papers for datasets like
    HotpotQA, Natural Questions, TriviaQA.

    Steps:
    1. Lowercase
    2. Remove articles (a, an, the)
    3. Remove punctuation
    4. Remove extra whitespace

    Args:
        text: Raw answer text

    Returns:
        Normalized answer text
    """
    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text.strip()


def extract_answer_from_text(text: str) -> str:
    """Extract answer part from model output.

    Common patterns in RAG outputs:
    - "Answer: ..."
    - "The answer is ..."
    - "(answer)" or "answer."
    - First sentence

    Args:
        text: Model output text

    Returns:
        Extracted answer
    """
    text = text.strip()

    # Pattern 1: "Answer: ..." or "The answer is ..."
    answer_patterns = [
        r'(?:final\s+)?answer\s*(?:is)?\s*:?\s*(.+?)(?:\.|$)',
        r'therefore,?\s+(.+?)(?:\.|$)',
        r'(?:in\s+)?conclusion,?\s+(.+?)(?:\.|$)',
        r'the\s+answer\s+is\s+\(?(.+?)\)?(?:\.|$)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text.lower())
        if match:
            answer = match.group(1).strip()
            # Remove parentheses
            answer = re.sub(r'[()]', '', answer)
            return answer

    # Pattern 2: Check for content in parentheses at the end
    paren_match = re.search(r'\(([^)]+)\)[.,;]?\s*$', text)
    if paren_match:
        return paren_match.group(1).strip()

    # Pattern 3: Take first sentence as fallback
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        return sentences[0].strip()

    return text


def compute_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 score (SQuAD-style).

    Args:
        predicted: Predicted answer (normalized)
        gold: Gold answer (normalized)

    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = predicted.split()
    gold_tokens = gold.split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_em(predicted: str, gold: str) -> bool:
    """Compute exact match (SQuAD-style).

    Args:
        predicted: Predicted answer (normalized)
        gold: Gold answer (normalized)

    Returns:
        True if exact match
    """
    return predicted == gold


class StepType(str, Enum):
    """Type of step in adaptive trajectory."""
    COT = "cot"          # Pure reasoning step
    RAG = "rag"          # Retrieval-augmented step
    ANSWER = "answer"    # Final answer


@dataclass
class AdaptiveStep:
    """A step in an adaptive MC-CoT + RAG trajectory.

    This includes MC values before/after for monitoring.
    Uses step forcing format: "Step N: {content}"
    """
    step_id: int
    step_type: StepType
    text: str                           # The reasoning/retrieval text (with "Step N:" prefix)
    content: str                        # Just the content (without "Step N:" prefix)
    used_passages: List[Dict[str, Any]] # Retrieved passages (if RAG)
    mc_before: float                    # MC(s_{t-1})
    mc_after: float                     # MC(s_t)
    rpe: float                          # mc_after / mc_before
    label: Optional[str] = None         # RPE-based label: 'good' if rpe >= 0.8, 'bad' if rpe < 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step_id': self.step_id,
            'step_type': self.step_type.value,
            'text': self.text,
            'content': self.content,
            'used_passages': self.used_passages,
            'mc_before': self.mc_before,
            'mc_after': self.mc_after,
            'rpe': self.rpe,
            'label': self.label,
            'metadata': self.metadata,
        }


@dataclass
class AdaptiveTrajectory:
    """An adaptive RAG-CoT trajectory."""
    trajectory_id: str
    question: str
    steps: List[AdaptiveStep]
    final_answer: str
    gold_answer: Optional[str] = None
    supporting_facts: Optional[List[str]] = None
    is_correct: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trajectory_id': self.trajectory_id,
            'question': self.question,
            'steps': [s.to_dict() for s in self.steps],
            'final_answer': self.final_answer,
            'gold_answer': self.gold_answer,
            'supporting_facts': self.supporting_facts,
            'is_correct': self.is_correct,
            'metadata': self.metadata,
        }


class AdaptiveTrajectoryGenerator:
    """Generator for adaptive MC-CoT + RAG trajectories.

    Algorithm:
    1. Generate CoT step and compute RPE
    2. If RPE >= 0.8: accept as 'good' step
    3. If RPE < 0.8: try RAG intervention
       - If RAG RPE >= 0.8: accept as 'good' step
       - If RAG RPE < 0.8: accept as 'bad' step (no termination)
    4. Continue until max_steps or answer found

    All steps are kept in trajectory with binary labels ('good'/'bad')
    for PRM training.
    """

    def __init__(
        self,
        policy_model,           # Small model for CoT generation
        retriever: BM25Retriever,
        config: Dict[str, Any],
    ):
        """Initialize adaptive trajectory generator.

        Args:
            policy_model: Model for generating CoT steps
            retriever: BM25 retriever for RAG
            config: Configuration dict with:
                - num_rollouts: Number of MC rollouts
                - max_steps: Maximum steps per trajectory
                - num_rag_queries: Number of RAG query candidates
                - top_k_passages: Number of passages to retrieve
                - use_step_forcing: Whether to use step forcing (default: True)
                - max_paths: Maximum paths to sample per problem (default: 2048)
                - temperature: Sampling temperature (default: 0.8)

                Note: delta and epsilon parameters are deprecated.
                Fixed threshold of 0.8 is used for RPE-based labeling.
        """
        self.policy_model = policy_model
        self.retriever = retriever

        self.num_rollouts = config.get('num_rollouts', 5)
        self.delta = config.get('delta', 0.1)           # CoT tolerance (deprecated)
        self.epsilon = config.get('epsilon', 0.1)       # RAG improvement threshold (deprecated)
        self.max_steps = config.get('max_steps', 10)
        self.num_rag_queries = config.get('num_rag_queries', 1)  # Default to 1 query
        self.top_k_passages = config.get('top_k_passages', 5)
        self.temperature = config.get('temperature', 0.8)

        # Step forcing settings
        self.use_step_forcing = config.get('use_step_forcing', True)
        self.step_forcing_prompt = StepForcingPrompt()
        self.step_parser = StepParser()

        # Multi-path sampling
        self.max_paths = config.get('max_paths', 2048)
        self.multi_path_sampler = MultiPathSampler(max_paths=self.max_paths)

    def generate_trajectory(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        supporting_facts: Optional[List[str]] = None,
        trajectory_id: Optional[str] = None,
    ) -> Optional[AdaptiveTrajectory]:
        """Generate a single adaptive trajectory for a question.

        Args:
            question: The question to answer
            gold_answer: Gold answer (for evaluation)
            supporting_facts: Supporting facts (for evaluation)
            trajectory_id: Unique ID for this trajectory

        Returns:
            AdaptiveTrajectory or None if generation failed
        """
        trajectory_id = trajectory_id or f"traj_{hash(question)}"

        # Initialize state
        current_state = {
            'question': question,
            'reasoning_history': [],
            'forced_steps': [],  # Track ForcedStep objects
            'passages': [],
        }

        steps = []
        mc_prev = self._monte_carlo_estimate(current_state, gold_answer)

        for t in range(self.max_steps):
            step_num = t + 1

            # (A) Try CoT step with step forcing
            if self.use_step_forcing:
                cot_step_content = self._generate_cot_step_forced(current_state, step_num)
                cot_step_text = f"Step {step_num}: {cot_step_content}"
            else:
                cot_step_content = self._generate_cot_step(current_state)
                cot_step_text = cot_step_content

            # Create temporary state with CoT step
            cot_state = self._apply_cot_step(current_state, cot_step_text, step_num)
            mc_cot = self._monte_carlo_estimate(cot_state, gold_answer)

            # Compute RPE for CoT
            rpe_cot = mc_cot / (mc_prev + 1e-8)

            # (B) Check if CoT is acceptable (threshold: 0.8, using > 0.79 to avoid floating point issues)
            if rpe_cot > 0.79:
                # Accept CoT step with 'good' label
                step = AdaptiveStep(
                    step_id=t,
                    step_type=StepType.COT,
                    text=cot_step_text,
                    content=cot_step_content,
                    used_passages=[],
                    mc_before=mc_prev,
                    mc_after=mc_cot,
                    rpe=rpe_cot,
                    label='good',
                    metadata={'accepted': 'cot', 'threshold': 0.8},
                )
                steps.append(step)
                current_state = cot_state
                mc_prev = mc_cot

                # Check if we have an answer
                if self._has_answer(cot_step_content):
                    final_answer = self._extract_answer(cot_step_content)
                    break

            else:
                # (C) CoT below threshold (RPE < 0.8), try RAG intervention
                print(f"  Step {step_num}: CoT RPE={rpe_cot:.3f} < 0.8, trying RAG...")

                rag_result = self._try_rag_intervention(
                    current_state, mc_prev, gold_answer, step_num
                )

                # RAG always returns a result (never None)
                rag_step, rag_state, mc_rag = rag_result
                rpe_rag = mc_rag / (mc_prev + 1e-8)

                # Label as 'good' if RPE > 0.79 (effective 0.8 with floating point tolerance), else 'bad'
                label = 'good' if rpe_rag > 0.79 else 'bad'

                step = AdaptiveStep(
                    step_id=t,
                    step_type=StepType.RAG,
                    text=rag_step['text'],
                    content=rag_step['content'],
                    used_passages=rag_step['passages'],
                    mc_before=mc_prev,
                    mc_after=mc_rag,
                    rpe=rpe_rag,
                    label=label,
                    metadata={'accepted': 'rag', 'threshold': 0.8},
                )
                steps.append(step)
                current_state = rag_state
                mc_prev = mc_rag

        # Extract final answer
        final_answer = self._extract_final_answer(steps)
        is_correct = self._check_answer(final_answer, gold_answer) if gold_answer else None

        return AdaptiveTrajectory(
            trajectory_id=trajectory_id,
            question=question,
            steps=steps,
            final_answer=final_answer,
            gold_answer=gold_answer,
            supporting_facts=supporting_facts,
            is_correct=is_correct,
            metadata={
                'num_steps': len(steps),
                'num_cot_steps': sum(1 for s in steps if s.step_type == StepType.COT),
                'num_rag_steps': sum(1 for s in steps if s.step_type == StepType.RAG),
            },
        )

    def _generate_rag_step_with_passages(
        self,
        state: Dict[str, Any],
        passages: List[Dict[str, Any]],
        step_num: int,
    ) -> str:
        """Generate a reasoning step based on retrieved passages.

        Args:
            state: Current state with question and reasoning history
            passages: Retrieved passages to use for reasoning
            step_num: Step number to generate

        Returns:
            Step content (without "Step N:" prefix)
        """
        if self.policy_model is None:
            # Placeholder for testing without model
            passage_text = self._format_passages(passages)
            return f"Based on retrieved information: {passage_text}"

        # Format passages for the prompt
        passage_text = self._format_passages(passages)

        # Build prompt with passages
        forced_steps = state.get('forced_steps', [])

        if step_num == 1:
            # Initial step with passages
            prompt_lines = [
                f"Question: {state['question']}\n",
                "Retrieved Information:",
                passage_text,
                "\nBased on the retrieved information above, provide the first reasoning step.",
                f"Respond with EXACTLY ONE step in this format:",
                f'"Step {step_num}: [your reasoning based on the documents]"',
                '\nIf this is your final step, include: "Final Answer: [your answer]"\n',
                f"Step {step_num}:"
            ]
        else:
            # Continuation step with passages
            steps_text = "\n".join(str(step) for step in forced_steps)
            prompt_lines = [
                f"Question: {state['question']}\n",
                steps_text,
                "\nRetrieved Information:",
                passage_text,
                f"\nBased on the retrieved information above, continue solving.",
                f"You MUST respond with EXACTLY ONE step in this format:",
                f'"Step {step_num}: [your reasoning based on the documents]"',
                '\nIf this is your final step, include: "Final Answer: [your answer]"',
                f"\nDo NOT write multiple steps. Write ONLY Step {step_num}.\n",
                f"Step {step_num}:"
            ]

        prompt = "\n".join(prompt_lines)

        # Generate with model
        response = self.policy_model.generate_with_chat_template(
            user_message=prompt,
            max_tokens=800,
            temperature=self.temperature,
            top_p=0.95,
            stop_sequences=["\nStep", "\n\n"],
        )

        # Parse response to extract step content
        import re
        # Remove any future steps
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        # Parse to get content without "Step N:" prefix
        return self.step_parser.parse_single_step_response(response, step_num)

    def _try_rag_intervention(
        self,
        current_state: Dict[str, Any],
        mc_prev: float,
        gold_answer: Optional[str],
        step_num: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """Try RAG intervention when CoT is below threshold.

        Always returns the best RAG result found (never None).
        The caller decides whether to label it 'good' or 'bad' based on RPE >= 0.8.

        Returns:
            Tuple of (rag_step_info, new_state, mc_rag)
        """
        # Generate multiple query candidates
        query_candidates = self._generate_rag_queries(current_state)

        best_rag = None
        best_mc = -1.0

        for query in query_candidates:
            # Retrieve passages
            passages = self.retriever.retrieve(query, top_k=self.top_k_passages)

            # Generate RAG reasoning step based on passages
            rag_step_content = self._generate_rag_step_with_passages(
                current_state,
                passages,
                step_num
            )
            rag_step_text = f"Step {step_num}: {rag_step_content}"

            # Create state with RAG step
            rag_state = self._apply_rag_step(current_state, query, passages, step_num, rag_step_content)

            # Compute MC
            mc_rag = self._monte_carlo_estimate(rag_state, gold_answer)

            if mc_rag > best_mc:
                best_mc = mc_rag
                best_rag = {
                    'query': query,
                    'passages': passages,
                    'text': rag_step_text,
                    'content': rag_step_content,
                    'state': rag_state,
                }

        # Always return best RAG result (no threshold check)
        return (best_rag, best_rag['state'], best_mc)

    def _generate_cot_step(self, state: Dict[str, Any]) -> str:
        """Generate next CoT reasoning step (legacy, without step forcing)."""
        if self.policy_model is None:
            # Placeholder
            return "[Reasoning step generated by LLM]"

        # Format prompt
        prompt = self._format_cot_prompt(state)

        # Generate with model
        # In real implementation, use:
        # response = self.policy_model.generate(prompt, temperature=self.temperature)

        # Placeholder
        return "Let me think about this step by step..."

    def _generate_cot_step_forced(self, state: Dict[str, Any], step_num: int) -> str:
        """Generate next CoT step with step forcing.

        Args:
            state: Current state
            step_num: Step number to generate

        Returns:
            Step content (without "Step N:" prefix)
        """
        if self.policy_model is None:
            # Placeholder for testing without model
            return f"This is reasoning step {step_num} [generated by LLM]"

        # Build step forcing prompt
        forced_steps = state.get('forced_steps', [])
        if step_num == 1:
            prompt = self.step_forcing_prompt.build_initial_prompt(
                question=state['question'],
                context=""
            )
        else:
            prompt = self.step_forcing_prompt.build_continuation_prompt(
                question=state['question'],
                previous_steps=forced_steps,
                context=""
            )

        # Generate with model (use chat template for Qwen)
        # Stop sequences prevent generating multiple steps at once
        response = self.policy_model.generate_with_chat_template(
            user_message=prompt,
            max_tokens=800,  # Allow longer reasoning for complex steps
            temperature=self.temperature,
            top_p=0.95,
            stop_sequences=["\nStep", "\n\n"],  # Stop at next step or double newline
        )

        # Parse response to extract step content (removes "Step N:" if present)
        import re
        # First remove any future steps
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        # Then parse to get content without "Step N:" prefix
        return self.step_parser.parse_single_step_response(response, step_num)

    def _generate_rag_queries(self, state: Dict[str, Any]) -> List[str]:
        """Generate query candidates for RAG retrieval using LLM.

        Uses the policy model to generate focused search queries based on
        the question and current reasoning state.

        Args:
            state: Current state with question and reasoning history

        Returns:
            List of search queries for retrieval (up to num_rag_queries)
        """
        queries = []

        if self.policy_model is None:
            # Fallback: use question as-is
            return [state['question']]

        # Build prompt for query generation
        prompt_lines = [f"Question: {state['question']}\n"]

        if state.get('reasoning_history'):
            prompt_lines.append("Reasoning so far:")
            for step in state['reasoning_history']:
                prompt_lines.append(step)

        prompt_lines.append(
            "\nWhat specific information do we need to retrieve to answer this question? "
            "Generate a focused search query."
        )

        prompt = "\n".join(prompt_lines)

        try:
            response = self.policy_model.generate_with_chat_template(
                user_message=prompt,
                max_tokens=150,
                temperature=0.7,  # Some diversity for queries
            )

            # Parse queries from response (simple splitting)
            # Look for numbered list or newlines
            import re
            query_matches = re.findall(r'(?:^|\n)\s*(?:\d+\.|[-*])\s*(.+?)(?=\n|$)', response, re.MULTILINE)
            if query_matches:
                queries = [q.strip() for q in query_matches[:self.num_rag_queries]]
            else:
                # Fallback: split by newlines
                query_lines = [line.strip() for line in response.split('\n') if line.strip()]
                queries = query_lines[:self.num_rag_queries]

        except Exception as e:
            # If query generation fails, use question as fallback
            print(f"Warning: Query generation failed: {e}")
            queries = [state['question']]

        # Ensure we always return at least one query
        if not queries:
            queries = [state['question']]

        return queries[:self.num_rag_queries]

    def _format_passages(self, passages: List[Dict[str, Any]]) -> str:
        """Format retrieved passages for display."""
        text_parts = []
        for i, passage in enumerate(passages, 1):
            title = passage.get('title', 'Document')
            content = passage.get('text', '')[:200]
            text_parts.append(f"[{i}] {title}: {content}...")
        return "\n".join(text_parts)

    def _format_rag_step(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Format RAG step text (legacy)."""
        text = f"Retrieved information for: {query}\n\n"
        text += self._format_passages(passages)
        return text

    def _apply_cot_step(
        self,
        state: Dict[str, Any],
        step_text: str,
        step_num: int,
    ) -> Dict[str, Any]:
        """Apply CoT step to state."""
        new_state = state.copy()
        new_state['reasoning_history'] = state['reasoning_history'] + [step_text]

        # Update forced steps if using step forcing
        if self.use_step_forcing:
            # Extract content from "Step N: content"
            content = step_text.replace(f"Step {step_num}: ", "", 1)
            forced_step = ForcedStep(step_number=step_num, content=content)
            new_state['forced_steps'] = state.get('forced_steps', []) + [forced_step]

        return new_state

    def _apply_rag_step(
        self,
        state: Dict[str, Any],
        query: str,
        passages: List[Dict[str, Any]],
        step_num: int,
        rag_step_content: str,
    ) -> Dict[str, Any]:
        """Apply RAG step to state.

        Args:
            state: Current state
            query: RAG query used
            passages: Retrieved passages
            step_num: Step number
            rag_step_content: LLM-generated reasoning content based on passages

        Returns:
            New state with RAG step added
        """
        new_state = state.copy()
        new_state['passages'] = state['passages'] + passages

        # Use LLM-generated content
        rag_step_text = f"Step {step_num}: {rag_step_content}"

        if self.use_step_forcing:
            forced_step = ForcedStep(step_number=step_num, content=rag_step_content)
            new_state['forced_steps'] = state.get('forced_steps', []) + [forced_step]

        new_state['reasoning_history'] = state['reasoning_history'] + [rag_step_text]
        return new_state

    def _monte_carlo_estimate(
        self,
        state: Dict[str, Any],
        gold_answer: Optional[str],
    ) -> float:
        """Estimate success probability via MC rollouts."""
        if self.policy_model is None:
            # Placeholder: random estimate
            return np.random.uniform(0.3, 0.9)

        successes = 0
        for _ in range(self.num_rollouts):
            final_answer = self._rollout(state)
            if gold_answer and self._check_answer(final_answer, gold_answer):
                successes += 1

        return successes / self.num_rollouts

    def _rollout(self, state: Dict[str, Any]) -> str:
        """Perform one rollout from current state.

        Generate a complete solution from the current state by continuing
        the reasoning until we reach a final answer.

        Args:
            state: Current state with 'question' and 'reasoning_history'

        Returns:
            Final answer extracted from the rollout
        """
        if self.policy_model is None:
            # Placeholder for testing without model
            return "rollout_answer"

        # Build rollout prompt
        lines = [f"Question: {state['question']}\n"]

        # Add existing reasoning history if any
        if state.get('reasoning_history'):
            lines.append("Reasoning so far:")
            for step_text in state['reasoning_history']:
                lines.append(step_text)
            lines.append("\nContinue solving and end with: 'Therefore, the answer is [your answer].'")
        else:
            lines.append("Solve this problem step by step.")
            lines.append("End your response with: 'Therefore, the answer is [your answer].'")

        prompt = "\n".join(lines)

        # Generate complete solution (longer max_tokens for rollout)
        response = self.policy_model.generate_with_chat_template(
            user_message=prompt,
            max_tokens=800,  # Allow longer generation for multi-hop reasoning
            temperature=self.temperature,
            top_p=0.95,
        )

        # Extract final answer from response
        return self._extract_answer(response)

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if answer is correct using substring inclusion.

        Uses exact substring matching after normalization (lowercase, remove
        articles/punctuation). No paraphrasing or partial similarity - only
        checks if normalized gold answer is contained in normalized prediction.

        Args:
            predicted: Predicted answer (raw)
            gold: Gold answer (raw)

        Returns:
            True if normalized gold is substring of normalized prediction
        """
        # Extract answer parts first
        pred_extracted = extract_answer_from_text(predicted)
        gold_extracted = extract_answer_from_text(gold)

        # Normalize both
        pred_norm = normalize_answer(pred_extracted)
        gold_norm = normalize_answer(gold_extracted)

        # Check substring inclusion
        return gold_norm in pred_norm

    def _format_cot_prompt(self, state: Dict[str, Any]) -> str:
        """Format prompt for CoT generation."""
        lines = [f"Question: {state['question']}\n"]

        if state['reasoning_history']:
            lines.append("Reasoning so far:")
            for i, step in enumerate(state['reasoning_history'], 1):
                lines.append(f"{i}. {step}")

        lines.append("\nNext reasoning step:")
        return "\n".join(lines)

    def _has_answer(self, text: str) -> bool:
        """Check if text contains the explicit 'Final Answer:' marker.

        This uses a single, clear termination marker that the model is instructed
        to use in its last step, rather than heuristic patterns.
        """
        text_lower = text.lower()
        return 'final answer:' in text_lower

    def _extract_answer(self, text: str) -> str:
        """Extract answer from text using SQuAD-style extraction.

        Args:
            text: Model output text

        Returns:
            Extracted answer
        """
        return extract_answer_from_text(text)

    def _extract_final_answer(self, steps: List[AdaptiveStep]) -> str:
        """Extract final answer from trajectory.

        Tries to find answer from:
        1. Last step (most likely to contain final answer)
        2. Any step with answer markers (fallback)
        3. Concatenated content (last resort)

        Args:
            steps: List of adaptive steps

        Returns:
            Extracted final answer
        """
        if not steps:
            return ""

        # Try to extract from last step first
        last_content = steps[-1].content
        extracted = extract_answer_from_text(last_content)

        # If we got a meaningful answer, return it
        if extracted and len(extracted) > 0:
            return extracted

        # Fallback: try all steps in reverse order
        for step in reversed(steps):
            if self._has_answer(step.content):
                extracted = extract_answer_from_text(step.content)
                if extracted and len(extracted) > 0:
                    return extracted

        # Last resort: return last step content
        return steps[-1].content if steps else ""

    def generate_batch(
        self,
        questions: List[Dict[str, Any]],
        num_trajectories_per_question: int = 1,
        show_progress: bool = True,
    ) -> List[AdaptiveTrajectory]:
        """Generate trajectories for multiple questions.

        Args:
            questions: List of question dicts with 'question', 'gold_answer', etc.
            num_trajectories_per_question: Number of trajectories per question
            show_progress: Show progress bar

        Returns:
            List of generated trajectories
        """
        all_trajectories = []

        iterator = tqdm(questions, desc="Generating trajectories") if show_progress else questions

        for q_data in iterator:
            for n in range(num_trajectories_per_question):
                trajectory = self.generate_trajectory(
                    question=q_data['question'],
                    gold_answer=q_data.get('gold_answer'),
                    supporting_facts=q_data.get('supporting_facts'),
                    trajectory_id=f"{q_data.get('id', 'q')}_{n}",
                )

                if trajectory is not None:
                    all_trajectories.append(trajectory)

        return all_trajectories
