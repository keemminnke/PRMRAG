"""Adaptive MC-CoT + RAG trajectory generation.

This module implements the core Stage 1 logic:
- Generate CoT steps with MC monitoring
- Intervene with RAG when MC drops
- Track mc_before, mc_after for each step
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from tqdm import tqdm

from ..data.schemas import ActionType
from ..retrieval import BM25Retriever


class StepType(str, Enum):
    """Type of step in adaptive trajectory."""
    COT = "cot"          # Pure reasoning step
    RAG = "rag"          # Retrieval-augmented step
    ANSWER = "answer"    # Final answer


@dataclass
class AdaptiveStep:
    """A step in an adaptive MC-CoT + RAG trajectory.

    This includes MC values before/after for monitoring.
    """
    step_id: int
    step_type: StepType
    text: str                           # The reasoning/retrieval text
    used_passages: List[Dict[str, Any]] # Retrieved passages (if RAG)
    mc_before: float                    # MC(s_{t-1})
    mc_after: float                     # MC(s_t)
    rpe: float                          # mc_after / mc_before
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step_id': self.step_id,
            'step_type': self.step_type.value,
            'text': self.text,
            'used_passages': self.used_passages,
            'mc_before': self.mc_before,
            'mc_after': self.mc_after,
            'rpe': self.rpe,
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
    1. Start with CoT reasoning
    2. Monitor MC at each step
    3. If MC drops (P_t^cot < 1-δ), rollback and try RAG
    4. If RAG improves MC (P_t^rag >= 1+ε), continue
    5. Otherwise, terminate trajectory
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
                - delta: CoT acceptance threshold (1-δ)
                - epsilon: RAG improvement threshold (1+ε)
                - max_steps: Maximum steps per trajectory
                - num_rag_queries: Number of RAG query candidates
                - top_k_passages: Number of passages to retrieve
        """
        self.policy_model = policy_model
        self.retriever = retriever

        self.num_rollouts = config.get('num_rollouts', 5)
        self.delta = config.get('delta', 0.1)           # CoT tolerance
        self.epsilon = config.get('epsilon', 0.1)       # RAG improvement threshold
        self.max_steps = config.get('max_steps', 10)
        self.num_rag_queries = config.get('num_rag_queries', 3)
        self.top_k_passages = config.get('top_k_passages', 5)
        self.temperature = config.get('temperature', 0.8)

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
            'passages': [],
        }

        steps = []
        mc_prev = self._monte_carlo_estimate(current_state, gold_answer)

        for t in range(self.max_steps):
            # (A) Try CoT step
            cot_step_text = self._generate_cot_step(current_state)

            # Create temporary state with CoT step
            cot_state = self._apply_cot_step(current_state, cot_step_text)
            mc_cot = self._monte_carlo_estimate(cot_state, gold_answer)

            # Compute RPE for CoT
            rpe_cot = mc_cot / (mc_prev + 1e-8)

            # (B) Check if CoT is acceptable
            if rpe_cot >= (1 - self.delta):
                # Accept CoT step
                step = AdaptiveStep(
                    step_id=t,
                    step_type=StepType.COT,
                    text=cot_step_text,
                    used_passages=[],
                    mc_before=mc_prev,
                    mc_after=mc_cot,
                    rpe=rpe_cot,
                    metadata={'accepted': 'cot', 'threshold': 1 - self.delta},
                )
                steps.append(step)
                current_state = cot_state
                mc_prev = mc_cot

                # Check if we have an answer
                if self._has_answer(cot_step_text):
                    final_answer = self._extract_answer(cot_step_text)
                    break

            else:
                # (C) CoT failed, try RAG intervention
                print(f"  Step {t}: CoT RPE={rpe_cot:.3f} < {1-self.delta:.3f}, trying RAG...")

                rag_result = self._try_rag_intervention(
                    current_state, mc_prev, gold_answer
                )

                if rag_result is None:
                    # RAG also failed, terminate trajectory
                    print(f"  Step {t}: RAG intervention failed, terminating")
                    return None

                # RAG succeeded
                rag_step, rag_state, mc_rag = rag_result
                step = AdaptiveStep(
                    step_id=t,
                    step_type=StepType.RAG,
                    text=rag_step['text'],
                    used_passages=rag_step['passages'],
                    mc_before=mc_prev,
                    mc_after=mc_rag,
                    rpe=mc_rag / (mc_prev + 1e-8),
                    metadata={'accepted': 'rag', 'threshold': 1 + self.epsilon},
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

    def _try_rag_intervention(
        self,
        current_state: Dict[str, Any],
        mc_prev: float,
        gold_answer: Optional[str],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """Try RAG intervention when CoT fails.

        Returns:
            Tuple of (rag_step_info, new_state, mc_rag) or None if failed
        """
        # Generate multiple query candidates
        query_candidates = self._generate_rag_queries(current_state)

        best_rag = None
        best_mc = -1.0

        for query in query_candidates:
            # Retrieve passages
            passages = self.retriever.retrieve(query, top_k=self.top_k_passages)

            # Create state with retrieved passages
            rag_state = self._apply_rag_step(current_state, query, passages)

            # Compute MC
            mc_rag = self._monte_carlo_estimate(rag_state, gold_answer)

            if mc_rag > best_mc:
                best_mc = mc_rag
                best_rag = {
                    'query': query,
                    'passages': passages,
                    'text': self._format_rag_step(query, passages),
                    'state': rag_state,
                }

        # Check if best RAG meets threshold
        rpe_rag = best_mc / (mc_prev + 1e-8)
        if rpe_rag >= (1 + self.epsilon):
            return (best_rag, best_rag['state'], best_mc)
        else:
            return None

    def _generate_cot_step(self, state: Dict[str, Any]) -> str:
        """Generate next CoT reasoning step."""
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

    def _generate_rag_queries(self, state: Dict[str, Any]) -> List[str]:
        """Generate query candidates for RAG retrieval."""
        if self.policy_model is None:
            # Placeholder
            return [
                state['question'],
                f"What is the answer to: {state['question']}",
                "Additional information needed",
            ][:self.num_rag_queries]

        # In real implementation, use LLM to generate queries
        # based on current reasoning history

        return [state['question']]  # Fallback to question

    def _format_rag_step(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Format RAG step text."""
        text = f"Retrieved information for: {query}\n\n"
        for i, passage in enumerate(passages, 1):
            text += f"[{i}] {passage['title']}: {passage['text'][:200]}...\n"
        return text

    def _apply_cot_step(self, state: Dict[str, Any], step_text: str) -> Dict[str, Any]:
        """Apply CoT step to state."""
        new_state = state.copy()
        new_state['reasoning_history'] = state['reasoning_history'] + [step_text]
        return new_state

    def _apply_rag_step(
        self,
        state: Dict[str, Any],
        query: str,
        passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply RAG step to state."""
        new_state = state.copy()
        new_state['passages'] = state['passages'] + passages
        new_state['reasoning_history'] = state['reasoning_history'] + [
            self._format_rag_step(query, passages)
        ]
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
        """Perform one rollout from current state."""
        # Placeholder
        return "rollout_answer"

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if answer is correct."""
        pred_norm = predicted.strip().lower()
        gold_norm = gold.strip().lower()
        return pred_norm == gold_norm or gold_norm in pred_norm

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
        """Check if text contains a final answer."""
        answer_markers = [
            'the answer is',
            'therefore',
            'final answer',
            'in conclusion',
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in answer_markers)

    def _extract_answer(self, text: str) -> str:
        """Extract answer from text."""
        # Simple extraction (can be improved)
        if 'answer is' in text.lower():
            return text.split('answer is')[-1].strip()
        return text.strip()

    def _extract_final_answer(self, steps: List[AdaptiveStep]) -> str:
        """Extract final answer from trajectory."""
        if not steps:
            return ""

        # Try to extract from last step
        last_text = steps[-1].text
        return self._extract_answer(last_text)

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
