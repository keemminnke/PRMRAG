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

from ..data.schemas import ActionType
from ..utils.answer_utils import (
    normalize_answer,
    extract_answer_from_text,
    check_answer_match,
)
from .step_forcing import (
    ForcedStep,
    StepForcingPrompt,
    StepParser,
    MultiPathSampler,
    format_trajectory_with_steps,
)


def parse_rag_content(content: str) -> Dict[str, Optional[str]]:
    """Parse RAG step content into structured fields.

    Expected format:
        Thought: <reasoning>
        Action: Search[query="..."] or Finish[answer="..."]
        Observation: <retrieved info>
        Sub-answer: <intermediate answer>

    Returns:
        Dict with keys: thought, action, action_input, observation, sub_answer
    """
    result = {
        'thought': None,
        'action': None,
        'action_input': None,
        'observation': None,
        'sub_answer': None,
    }

    # Parse Thought (supports both "Thought:" and "**Thought:**" markdown format)
    thought_match = re.search(r'\*?\*?Thought:\*?\*?\s*(.+?)(?=\n(?:\*?\*?Action:|\*?\*?Observation:|Sub-answer:|$))', content, re.DOTALL | re.IGNORECASE)
    if thought_match:
        result['thought'] = thought_match.group(1).strip()

    # Parse Action and Action Input
    # Match: Search[query="..."], Finish[answer="..."], Reason[content="..."], or legacy Search[query]
    # Supports both "Action:" and "**Action:**" markdown format
    action_match = re.search(r'\*?\*?Action:\*?\*?\s*(Search|Finish|Reason)\[(?:query=|answer=|content=)?"?(.+?)"?\]', content, re.IGNORECASE)
    if action_match:
        result['action'] = action_match.group(1)
        result['action_input'] = action_match.group(2).strip()

    # Parse Observation (supports both "Observation:" and "**Observation:**" markdown format)
    # Match until next section or end of string
    obs_match = re.search(r'\*?\*?Observation:\*?\*?\s*(.+?)(?=\n\s*(?:\*?\*?Thought:|\*?\*?Action:|Sub-answer:|Final Answer:)|$)', content, re.DOTALL | re.IGNORECASE)
    if obs_match:
        result['observation'] = obs_match.group(1).strip()
    else:
        # Fallback: match until end of string
        obs_match_simple = re.search(r'\*?\*?Observation:\*?\*?\s*(.+)', content, re.DOTALL | re.IGNORECASE)
        if obs_match_simple:
            result['observation'] = obs_match_simple.group(1).strip()

    # Parse Sub-answer
    sub_answer_match = re.search(r'Sub-answer:\s*(.+?)(?=\n\s*(?:Thought:|Action:|Final Answer:|Step\s+\d+:)|$)', content, re.DOTALL | re.IGNORECASE)
    if sub_answer_match:
        result['sub_answer'] = sub_answer_match.group(1).strip()

    return result


def extract_intermediate_answer(content: str) -> Optional[str]:
    """Extract intermediate answer from CoT content.

    Looks for patterns like:
    - "Sub-answer: ..." (most common)
    - "Therefore, ..."
    - "So, ..."
    - "Thus, ..."

    Returns:
        Intermediate answer if found, else None
    """
    # First try Sub-answer pattern (most reliable for CoT after RAG)
    sub_answer_match = re.search(r'Sub-answer:\s*(.+?)(?=\n\s*(?:Thought:|Action:|Final Answer:|Step\s+\d+:)|$)', content, re.DOTALL | re.IGNORECASE)
    if sub_answer_match:
        return sub_answer_match.group(1).strip()

    # Try common conclusion patterns
    patterns = [
        r'Therefore,\s*(.+?)(?:\n|$)',
        r'So,\s*(.+?)(?:\n|$)',
        r'Thus,\s*(.+?)(?:\n|$)',
        r'In conclusion,\s*(.+?)(?:\n|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


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

    # Structured fields (for all steps)
    thought: Optional[str] = None       # Thought process
    action: str = "Reason"              # "Search" (RAG), "Reason" (CoT), "Finish" (final answer)
    action_input: Optional[str] = None  # Search query (this IS the sub-query for RAG)
    observation: Optional[str] = None   # Retrieved information summary (RAG only)
    sub_answer: Optional[str] = None    # Intermediate answer extracted from observation

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step_id': self.step_id,
            'step_type': self.step_type.value,  # Keep for backward compatibility
            'text': self.text,
            'content': self.content,
            'used_passages': self.used_passages,
            'mc_before': self.mc_before,
            'mc_after': self.mc_after,
            'rpe': self.rpe,
            'label': self.label,
            # Structured fields
            'thought': self.thought,
            'action': self.action,
            'action_input': self.action_input,
            'observation': self.observation,
            'sub_answer': self.sub_answer,
            'metadata': self.metadata,
        }
#5090 vllm w get url error 

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
    # DPO Data: Rejected reasoning segments for preference learning
    # These are segments where model tried pure reasoning but failed (RPE < 0.8)
    rejected_segments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'trajectory_id': self.trajectory_id,
            'question': self.question,
            'steps': [s.to_dict() for s in self.steps],
            'final_answer': self.final_answer,
            'gold_answer': self.gold_answer,
            'supporting_facts': self.supporting_facts,
            'is_correct': self.is_correct,
            'metadata': self.metadata,
        }
        # Only include rejected_segments if not empty (for DPO training)
        if self.rejected_segments:
            result['rejected_segments'] = self.rejected_segments
        return result


@dataclass
class TrajectoryState:
    """State for a single trajectory during batch generation.

    This class tracks the progress of one trajectory during parallel batch generation,
    allowing multiple trajectories to be processed synchronously step-by-step.
    """
    trajectory_id: str
    question: str
    gold_answer: Optional[str]
    supporting_facts: Optional[List[str]]

    # Current generation state
    state: Dict[str, Any]  # Contains: question, reasoning_history, forced_steps, passages, current_passages
    steps: List[AdaptiveStep] = field(default_factory=list)
    rejected_segments: List[Dict[str, Any]] = field(default_factory=list)

    # MC tracking
    mc_prev: float = 0.0  # MC value from previous step
    mc_question: float = 0.0  # Baseline MC on question alone
    current_k: int = 8  # Dynamic K for this trajectory (updated after Step 1)

    # Failure tracking (GenPRM-style)
    failed_after_rag: bool = False

    # Termination
    is_finished: bool = False
    finish_reason: Optional[str] = None  # "answer_found", "max_steps", "failed"

    def current_step_num(self) -> int:
        """Get the next step number to generate."""
        return len(self.steps) + 1


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
        retriever,              # Any retriever with retrieve() method
        config: Dict[str, Any],
    ):
        """Initialize adaptive trajectory generator.

        Args:
            policy_model: Model for generating CoT steps
            retriever: Retriever for RAG (BGE, BM25, or Hybrid)
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

        self.num_rollouts = config.get('num_rollouts', 8)
        self.max_steps = config.get('max_steps', 10)
        self.top_k_passages = config.get('top_k_passages', 5)  
        self.temperature = config.get('temperature', 0.8)

        # Dynamic K settings for MC estimation (based on GenPRM)
        # K is adjusted based on problem difficulty (estimated from first step MC)
        # Increased from 8/16/32 to 32/64/128 based on experiments showing 60% accuracy improvement
        self.use_dynamic_k = config.get('use_dynamic_k', True)
        self.k_hard = config.get('k_hard', 128)     # MC(s1) < 0.1
        self.k_medium = config.get('k_medium', 64)  # 0.1 <= MC(s1) < 0.9
        self.k_easy = config.get('k_easy', 32)      # MC(s1) >= 0.9
        self.current_k = self.num_rollouts  # Will be updated after first step

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
            'passages': [],  # Accumulated passages from all RAG steps
            'current_passages': [],  # Passages from current step only (for rollout)
        }

        steps = []

        # DPO Data Collection: Store rejected reasoning segments for preference learning
        # These are segments where model attempted pure reasoning but MC was low (< 0.8)
        # They serve as negative samples for DPO/RLHF training
        rejected_segments = []

        # Calculate MC on question alone BEFORE Step 1 (for proper RPE calculation)
        print(f"\n[Pre-Step1] Calculating baseline MC on question alone...")
        mc_question = self._monte_carlo_estimate(current_state, gold_answer)
        print(f"[Pre-Step1] MC(question only) = {mc_question:.3f}")

        # Use mc_question as baseline for Step 1
        mc_prev = mc_question

        # GenPRM-style early stopping: if RAG fails (MC < 0.01), set all future MC to 0
        failed_after_rag = False

        for t in range(self.max_steps):
            step_num = t + 1

            # (A) Generate CoT step with step forcing
            cot_step_content = self._generate_cot_step_forced(current_state, step_num)
            cot_step_text = f"Step {step_num}: {cot_step_content}"

            # Create temporary state with CoT step
            cot_state = self._apply_cot_step(current_state, cot_step_text, step_num)

            # If already failed after RAG, set MC to 0 without rollout
            if failed_after_rag:
                mc_cot = 0.0
                print(f"  Step {step_num}: MC fixed to 0.0 (failed after RAG)")
            else:
                mc_cot = self._monte_carlo_estimate(cot_state, gold_answer)

            # Update dynamic K based on question difficulty (only on step 1)
            if step_num == 1 and self.use_dynamic_k:
                if mc_question < 0.1:
                    self.current_k = self.k_hard
                    difficulty = "hard"
                elif mc_question < 0.9:
                    self.current_k = self.k_medium
                    difficulty = "medium"
                else:
                    self.current_k = self.k_easy
                    difficulty = "easy"
                print(f"  [Dynamic K] MC(question)={mc_question:.3f} → difficulty={difficulty} → K={self.current_k}")

            # Intent-based routing (same logic for all steps)
            # CHECK INTENT FIRST: Does model want to finish or search?
            # Supports both "Action:" and "**Action:**" markdown format
            content_lower = cot_step_content.lower()
            has_finish_intent = "action: finish" in content_lower or "action:finish" in content_lower
            has_search_intent = "action: search" in content_lower or "action:search" in content_lower

            # ================================================================
            # PATH 0: VOLUNTARY FINISH - Model wants to provide final answer
            # ================================================================
            if has_finish_intent:
                print(f"  Step {step_num}: Model requests FINISH - extracting final answer")
                # Accept the CoT step as final answer step
                step = AdaptiveStep(
                    step_id=t,
                    step_type=StepType.COT,
                    text=cot_step_text,
                    content=cot_step_content,
                    used_passages=[],
                    mc_before=mc_prev,
                    mc_after=mc_cot,
                    rpe=mc_cot / (mc_prev + 0.01),
                    label='good',
                    thought=cot_step_content,
                    action="Finish",
                    action_input=None,
                    observation=None,
                    sub_answer=None,
                    metadata={'voluntary_finish': True},
                )
                steps.append(step)
                current_state = cot_state
                print(f"  Step {step_num}: ✓ Voluntary FINISH accepted, terminating trajectory.")
                break

            if has_search_intent:
                # ================================================================
                # PATH A: VOLUNTARY SEARCH - Trust the model completely
                # Key principle: "Trust Search, Verify Reasoning"
                # - Model explicitly requests external knowledge (metacognition)
                # - Skip MC verification entirely (no computation waste)
                # - Execute RAG immediately and continue
                # ================================================================

                # Extract model's search query from the generated content
                import re
                query_match = re.search(r'Search\s*\[\s*(?:query\s*=\s*)?["\']?(.+?)["\']?\s*\]', cot_step_content, re.IGNORECASE)
                model_query = query_match.group(1).strip() if query_match else None

                if model_query:
                    print(f"  Step {step_num}: Model requests search with query: '{model_query[:60]}...'")
                else:
                    print(f"  Step {step_num}: Model requests search (voluntary) - extracting query failed, will generate new")

                # Execute RAG with model's query (or generate new if extraction failed)
                rag_result = self._try_rag_intervention(
                    current_state, mc_prev, gold_answer, step_num,
                    provided_query=model_query  # Pass model's query
                )

                rag_step, rag_state, mc_rag = rag_result

                # RPE-based labeling (same as other steps)
                rpe_rag = mc_rag / (mc_prev + 0.01)
                label = 'good' if rpe_rag > 0.79 else 'bad'

                parsed_fields = parse_rag_content(rag_step['content'])
                has_answer = self._has_answer(rag_step['content'])

                # Determine action from parsed content (could be Search or Finish)
                parsed_action = parsed_fields.get('action') or "Search"

                step = AdaptiveStep(
                    step_id=t,
                    step_type=StepType.RAG,
                    text=rag_step['text'],
                    content=rag_step['content'],
                    used_passages=rag_step.get('passages', []),  # Key is 'passages' from _try_rag_intervention
                    mc_before=mc_prev,
                    mc_after=mc_rag,
                    rpe=rpe_rag,
                    label=label,
                    thought=parsed_fields.get('thought'),
                    action=parsed_action,
                    action_input=parsed_fields.get('action_input'),
                    observation=parsed_fields.get('observation'),
                    sub_answer=parsed_fields.get('sub_answer'),
                    metadata={
                        'voluntary_search': True,
                    },
                )
                steps.append(step)
                current_state = rag_state
                mc_prev = mc_rag

                print(f"  Step {step_num}: ✓ Voluntary search (MC={mc_rag:.3f}, RPE={rpe_rag:.3f}, label={label})")

                if has_answer:
                    final_answer = self._extract_answer(rag_step['content'])
                    print(f"  Step {step_num}: ✓ Found final answer, terminating trajectory.")
                    break
                continue

            # ================================================================
            # PATH B: VOLUNTARY REASONING - Verify with MC before accepting
            # ================================================================
            print(f"  Step {step_num}: Model attempts pure reasoning (voluntary) - verifying with MC...")

            # If failed after RAG, skip RPE check and just accept with MC=0
            if failed_after_rag:
                rpe_cot = 0.0
                # Accept step but with MC=0 and bad label
                # CoT step: thought = full reasoning, action = Reason, observation = intermediate answer
                has_answer = self._has_answer(cot_step_content)
                intermediate_ans = extract_intermediate_answer(cot_step_content)

                step = AdaptiveStep(
                    step_id=t,
                    step_type=StepType.COT,
                    text=cot_step_text,
                    content=cot_step_content,
                    used_passages=[],
                    mc_before=0.0,
                    mc_after=0.0,
                    rpe=0.0,
                    label='bad',
                    # CoT-specific fields
                    thought=cot_step_content,  # Full reasoning process
                    action="Reason",  # CoT always uses Reason action
                    action_input=None,  # No search input for CoT
                    observation=intermediate_ans,  # Intermediate conclusion if any
                    sub_answer=None,  # CoT doesn't have sub-answer
                    metadata={'accepted': 'cot_after_failure', 'threshold': 0.8},
                )
                steps.append(step)
                current_state = cot_state
                mc_prev = 0.0

                # Check if we have an answer (still need to check for termination)
                print(f"  Step {step_num}: CoT after failure (MC=0.0), checking for final answer...")
                print(f"  Step {step_num}: Has 'Final Answer:' marker? {has_answer}")
                if has_answer:
                    final_answer = self._extract_answer(cot_step_content)
                    print(f"  Step {step_num}: ✓ Found final answer, terminating trajectory.")
                    break
                continue

            # Use 0.01 as smoothing factor to avoid extreme RPE values when mc_prev is near 0
            rpe_cot = mc_cot / (mc_prev + 0.01)

            # (B) Check if CoT is acceptable (threshold: 0.8, using > 0.79 to avoid floating point issues)
            if rpe_cot > 0.79:
                # Accept CoT step with 'good' label
                # CoT step: thought = full reasoning, action = Reason, observation = intermediate answer
                has_answer = self._has_answer(cot_step_content)
                intermediate_ans = extract_intermediate_answer(cot_step_content)

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
                    # CoT-specific fields
                    thought=cot_step_content,  # Full reasoning process
                    action="Reason",  # CoT always uses Reason action
                    action_input=None,  # No search input for CoT
                    observation=intermediate_ans,  # Intermediate conclusion if any
                    sub_answer=None,  # CoT doesn't have sub-answer
                    metadata={'accepted': 'cot', 'threshold': 0.8},
                )
                steps.append(step)
                current_state = cot_state
                mc_prev = mc_cot

                # Check if we have an answer
                has_answer = self._has_answer(cot_step_content)
                print(f"  Step {step_num}: CoT accepted (RPE={rpe_cot:.3f}), checking for final answer...")
                print(f"  Step {step_num}: Has 'Final Answer:' marker? {has_answer}")
                if has_answer:
                    final_answer = self._extract_answer(cot_step_content)
                    print(f"  Step {step_num}: ✓ Found final answer, terminating trajectory.")
                    break

            else:
                # ================================================================
                # (C) BACKTRACKING: CoT below threshold (RPE < 0.8)
                # Key principle: "Verify Reasoning" - reject low-quality reasoning
                # 1. Save rejected segment for DPO training (negative sample)
                # 2. Discard the low-quality reasoning (backtrack)
                # 3. Force RAG intervention to correct the trajectory
                # ================================================================
                print(f"  Step {step_num}: CoT RPE={rpe_cot:.3f} < 0.8 - REJECTED (backtracking)")

                # Save rejected segment for DPO training
                # This is valuable training data: the model tried pure reasoning but failed
                rejected_segment = {
                    'step_num': step_num,
                    'content': cot_step_content,
                    'text': cot_step_text,
                    'mc_before': mc_prev,
                    'mc_after': mc_cot,
                    'rpe': rpe_cot,
                    'reason': 'low_rpe',
                    'context': current_state.get('reasoning_history', [])[-3:],  # Last 3 steps for context
                }
                rejected_segments.append(rejected_segment)
                print(f"  Step {step_num}: ✗ Rejected segment saved for DPO (total: {len(rejected_segments)})")

                # BACKTRACK: Do NOT add the rejected CoT step to the trajectory
                # Instead, force RAG intervention from the current state

                print(f"  Step {step_num}: Forcing RAG intervention (backtracking)...")
                rag_result = self._try_rag_intervention(
                    current_state, mc_prev, gold_answer, step_num
                )

                # RAG always returns a result (never None)
                rag_step, rag_state, mc_rag = rag_result
                # Use 0.01 as smoothing factor to avoid extreme RPE values when mc_prev is near 0
                rpe_rag = mc_rag / (mc_prev + 0.01)

                # Label as 'good' if RPE > 0.79 (effective 0.8 with floating point tolerance), else 'bad'
                label = 'good' if rpe_rag > 0.79 else 'bad'

                # Parse RAG step content into structured fields
                parsed_fields = parse_rag_content(rag_step['content'])
                has_answer = self._has_answer(rag_step['content'])

                # Determine action from parsed content (could be Search or Finish)
                parsed_action = parsed_fields.get('action') or "Search"

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
                    # Add parsed structured fields
                    thought=parsed_fields.get('thought'),
                    action=parsed_action,
                    action_input=parsed_fields.get('action_input'),
                    observation=parsed_fields.get('observation'),
                    sub_answer=parsed_fields.get('sub_answer'),
                    metadata={
                        'accepted': 'rag_after_backtrack',
                        'threshold': 0.8,
                        'backtracked_from': cot_step_content[:100],  # First 100 chars of rejected segment
                        'rejected_rpe': rpe_cot,
                    },
                )
                steps.append(step)
                current_state = rag_state
                mc_prev = mc_rag

                # GenPRM-style: if RAG also fails (MC < 0.01), mark trajectory as irreversibly failed
                if mc_rag < 0.01:
                    failed_after_rag = True
                    print(f"  Step {step_num}: RAG failed (MC={mc_rag:.3f} < 0.01), future steps will have MC=0")

                # Check if RAG step contains final answer
                has_answer = self._has_answer(rag_step['content'])
                print(f"  Step {step_num}: RAG step added (RPE={rpe_rag:.3f}, label={label}), checking for final answer...")
                print(f"  Step {step_num}: Has 'Final Answer:' marker? {has_answer}")
                if has_answer:
                    final_answer = self._extract_answer(rag_step['content'])
                    print(f"  Step {step_num}: ✓ Found final answer, terminating trajectory.")
                    break

        # Extract final answer
        final_answer = self._extract_final_answer(steps)
        is_correct = self._check_answer(final_answer, gold_answer) if gold_answer else None

        # Log DPO data collection summary
        if rejected_segments:
            print(f"\n[DPO Data] Collected {len(rejected_segments)} rejected segments for preference learning")

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
                # num_cot_steps and num_rag_steps removed - can be calculated from steps
                'has_rag': any(s.step_type == StepType.RAG for s in steps),
                'num_backtracks': len(rejected_segments),  # Number of times we backtracked
            },
            rejected_segments=rejected_segments,  # DPO training data
        )

    def _generate_rag_step_with_passages(
        self,
        state: Dict[str, Any],
        passages: List[Dict[str, Any]],
        step_num: int,
        query: str,
    ) -> str:
        """Generate a reasoning step based on retrieved passages.

        Args:
            state: Current state with question and reasoning history
            passages: Retrieved passages to use for reasoning
            step_num: Step number to generate
            query: The search query that was used to retrieve passages

        Returns:
            Step content (without "Step N:" prefix)
        """
        if self.policy_model is None:
            # Placeholder for testing without model
            passage_text = self._format_passages(passages)
            return f"Thought: I need to search for information.\nAction: Search[{query}]\nObservation: Based on retrieved information: {passage_text}"

        # Format passages for the prompt
        passage_text = self._format_passages(passages)

        # Build prompt with passages
        forced_steps = state.get('forced_steps', [])

        if step_num == 1:
            # Initial step with passages (instruction은 system prompt에서 처리)
            prompt_lines = [
                f"Question: {state['question']}",
                "",
                f"Retrieved Documents:",
                passage_text,
                "",
                f"Your search query was: {query}",
                "",
                f"Generate Step {step_num}.",
            ]
        else:
            # Continuation step with passages
            steps_text = "\n".join(str(step) for step in forced_steps)

            # Add previous documents titles for context
            previous_passages = state.get('passages', [])
            previous_docs_section = []
            if previous_passages:
                previous_docs_section.append("## Previous Documents (for reference)")
                for i, p in enumerate(previous_passages, 1):
                    previous_docs_section.append(f"[{i}] {p.get('title', 'Document')}")
                previous_docs_section.append("")

            prompt_lines = [
                f"## Question",
                state['question'],
                "",
                f"## Previous Reasoning Steps",
                steps_text,
                "",
            ]

            # Add previous docs titles if any
            if previous_docs_section:
                prompt_lines.extend(previous_docs_section)

            # instruction은 system prompt에서 처리
            prompt_lines.extend([
                f"Retrieved Documents (Current):",
                passage_text,
                "",
                f"Your search query was: {query}",
                "",
                f"Continue with Step {step_num}.",
            ])

        prompt = "\n".join(prompt_lines)

        # Generate with model
        # Don't use "\n\n" as it can cut off "Final Answer: \n\n [answer]" format
        response = self.policy_model.generate_with_chat_template(
            user_message=prompt,
            max_tokens=800,
            temperature=self.temperature,
            top_p=0.95,
            stop_sequences=["\nStep"],  # Only stop at next step marker
            allow_observation=True,  # RAG steps should include Observation/citations
        )

        # DEBUG: Show raw response
        print(f"    [RAG STEP DEBUG] Raw model response (first 200 chars): {response[:200]}")

        # Parse response to extract step content
        import re
        # Remove any future steps
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        # Parse to get content without "Step N:" prefix
        parsed_content = self.step_parser.parse_single_step_response(response, step_num)
        print(f"    [RAG STEP DEBUG] Parsed content (first 200 chars): {parsed_content[:200]}")

        # Add Observation section with actual retrieved document chunks
        # Insert after Action line (standard ReAct format: Thought -> Action -> Observation)
        observation_text = f"\nObservation:\n{passage_text}"

        # Insert Observation after Action line
        lines = parsed_content.split('\n')
        result_lines = []
        action_found = False

        for line in lines:
            result_lines.append(line)
            # Insert Observation after Action line
            if not action_found and line.strip().startswith('Action:'):
                action_found = True
                result_lines.append(observation_text)

        # If no Action found, append at end (fallback)
        if not action_found:
            result_lines.append(observation_text)

        final_content = '\n'.join(result_lines)
        print(f"    [RAG STEP DEBUG] Final content with Observation (first 300 chars): {final_content[:300]}")

        return final_content

    def _try_rag_intervention(
        self,
        current_state: Dict[str, Any],
        mc_prev: float,
        gold_answer: Optional[str],
        step_num: int,
        provided_query: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """Try RAG intervention when CoT is below threshold.

        Always returns a RAG result (never None).
        The caller decides whether to label it 'good' or 'bad' based on RPE >= 0.8.

        Args:
            current_state: Current trajectory state
            mc_prev: Previous MC value
            gold_answer: Gold answer for MC estimation
            step_num: Current step number
            provided_query: Optional query from model's Search action (if None, generate new)

        Returns:
            Tuple of (rag_step_info, new_state, mc_rag)
        """
        # Use provided query (from model's Search action) or generate new
        if provided_query:
            query = provided_query
            print(f"    [RAG DEBUG] Using model's query: {query[:80]}")
        else:
            query = self._generate_rag_query(current_state)
            print(f"    [RAG DEBUG] Generated query: {query[:80]}")

        # Retrieve passages
        passages = self.retriever.retrieve(query, top_k=self.top_k_passages)
        print(f"    [RAG DEBUG] Retrieved {len(passages)} passages, top title: {passages[0].get('title', 'N/A') if passages else 'None'}")

        # Generate RAG reasoning step based on passages
        rag_step_content = self._generate_rag_step_with_passages(
            current_state,
            passages,
            step_num,
            query
        )
        rag_step_text = f"Step {step_num}: {rag_step_content}"

        # Create state with RAG step
        rag_state = self._apply_rag_step(current_state, query, passages, step_num, rag_step_content)

        # Compute MC
        mc_rag = self._monte_carlo_estimate(rag_state, gold_answer)

        rag_step_info = {
            'query': query,
            'passages': passages,
            'text': rag_step_text,
            'content': rag_step_content,
            'state': rag_state,
        }

        return (rag_step_info, rag_state, mc_rag)

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
        # Don't use "\n\n" as it can cut off "Final Answer: \n\n [answer]" format
        response = self.policy_model.generate_with_chat_template(
            user_message=prompt,
            max_tokens=800,  # Allow longer reasoning for complex steps
            temperature=self.temperature,
            top_p=0.95,
            stop_sequences=["\nStep"],  # Only stop at next step marker
        )

        # Parse response to extract step content (removes "Step N:" if present)
        import re
        # First remove any future steps
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        # Then parse to get content without "Step N:" prefix
        return self.step_parser.parse_single_step_response(response, step_num)

    def _generate_rag_query(self, state: Dict[str, Any]) -> str:
        """Generate a single RAG query using LLM.

        Uses the policy model to generate a focused search query based on
        the question and current reasoning state.

        Args:
            state: Current state with question and reasoning history

        Returns:
            A single search query string
        """
        if self.policy_model is None:
            # Fallback: use question as-is
            return state['question']

        # Build prompt for query generation (based on iterative RAG paper)
        prompt_lines = []

        if state.get('reasoning_history'):
            # We have previous reasoning steps - generate follow-up question
            prompt_lines.extend([
                "You are searching for information to answer a question step by step.",
                "",
                f"## Main Question",
                state['question'],
                "",
                f"## Previous Reasoning Steps",
            ])
            for step in state['reasoning_history']:
                prompt_lines.append(step)
            prompt_lines.extend([
                "",
                "## Task",
                "Based on the reasoning so far, generate a simple follow-up search query to find the information needed to continue.",
                "- Ask a SIMPLE question that a search engine can understand",
                "- You may rephrase or decompose the main question if previous steps were not helpful",
                "- Use 3-8 keywords maximum",
                "- Do NOT write complex questions",
                "",
                "Respond with ONLY the search query. Do not explain yourself.",
                "",
                "Search query:"
            ])
        else:
            # First step - decompose the main question
            prompt_lines.extend([
                "You are searching for information to answer a question step by step.",
                "",
                f"## Main Question",
                state['question'],
                "",
                "## Task",
                "Generate a simple search query to find information that will help answer this question.",
                "- Break down the question if it requires multiple pieces of information",
                "- Ask a SIMPLE question that a search engine can understand",
                "- Use 3-8 keywords maximum",
                "",
                "Respond with ONLY the search query. Do not explain yourself.",
                "",
                "Search query:"
            ])

        prompt = "\n".join(prompt_lines)

        try:
            response = self.policy_model.generate_with_chat_template(
                user_message=prompt,
                max_tokens=100,
                temperature=0.7,
            )

            # Debug: print full response
            print(f"    [QUERY DEBUG] Raw LLM response: {response[:150]}")

            # Extract the first meaningful line as the query
            query = response.strip().split('\n')[0].strip()

            # Remove common prefixes like "1.", "-", "*"
            import re
            query = re.sub(r'^\s*(?:\d+\.|[-*])\s*', '', query)

            print(f"    [QUERY DEBUG] Extracted query: {query[:100]}")

            return query if query else state['question']

        except Exception as e:
            # If query generation fails, use question as fallback
            print(f"Warning: Query generation failed: {e}")
            return state['question']

    def _format_passages(self, passages: List[Dict[str, Any]]) -> str:
        """Format retrieved passages for display."""
        text_parts = []
        for i, passage in enumerate(passages, 1):
            title = passage.get('title', 'Document')
            # Use full text (HotpotQA passages are typically short, ~300-500 chars)
            content = passage.get('text', '')
            text_parts.append(f"[{i}] {title}: {content}")
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

        # Clear current_passages for CoT steps (no retrieval)
        new_state['current_passages'] = []

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
        # Store current step's passages separately for rollout
        new_state['current_passages'] = passages

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
        """Estimate success probability via parallel MC rollouts with RAG capability.

        All K rollouts are processed in parallel. When Search actions are detected,
        retrieval is performed and generation continues in parallel batches.
        """
        if self.policy_model is None:
            return np.random.uniform(0.3, 0.9)

        k = self.current_k if self.use_dynamic_k else self.num_rollouts
        max_rollout_steps = 5

        # Initialize K rollout states (copies of current state)
        rollout_states = []
        for _ in range(k):
            rollout_states.append({
                'question': state['question'],
                'reasoning_history': list(state.get('reasoning_history', [])),
                'passages': list(state.get('passages', [])),
                'current_passages': list(state.get('current_passages', [])),
                'finished': False,
                'final_answer': '',
            })

        # Iterative parallel generation with RAG
        for step in range(max_rollout_steps):
            # Get active (not finished) rollouts
            active_indices = [i for i, rs in enumerate(rollout_states) if not rs['finished']]
            if not active_indices:
                break

            # Build prompts for active rollouts
            prompts = []
            for idx in active_indices:
                prompt = self._build_rollout_prompt(rollout_states[idx])
                formatted = self.policy_model.format_prompt_for_qwen(prompt)
                prompts.append(formatted)

            # Batch generate for all active rollouts
            if hasattr(self.policy_model, 'batch_generate'):
                responses = self.policy_model.batch_generate(
                    prompts=prompts,
                    max_tokens=800,
                    temperature=self.temperature,
                    top_p=0.95,
                )
            else:
                responses = [self.policy_model.generate(p, max_tokens=800) for p in prompts]

            # Process responses and identify Search actions
            search_queries = []  # (active_idx_position, query)
            for pos, (idx, response) in enumerate(zip(active_indices, responses)):
                rs = rollout_states[idx]

                # Check for Finish action
                if self._has_answer(response):
                    rs['finished'] = True
                    rs['final_answer'] = self._extract_answer(response)
                    continue

                # Check for Search action
                query = self._extract_search_query(response)
                if query:
                    search_queries.append((idx, query, response))
                else:
                    # Pure reasoning - add to history
                    rs['reasoning_history'].append(response)
                    rs['current_passages'] = []

            # Batch retrieve for all Search actions
            if search_queries and self.retriever is not None:
                queries = [q[1] for q in search_queries]

                # Batch retrieval (if supported) or sequential
                if hasattr(self.retriever, 'batch_retrieve'):
                    all_passages = self.retriever.batch_retrieve(queries, top_k=5)
                else:
                    all_passages = [self.retriever.retrieve(q, top_k=5) for q in queries]

                # Update rollout states with observations
                for (idx, query, response), passages in zip(search_queries, all_passages):
                    rs = rollout_states[idx]
                    passage_text = self._format_passages(passages)
                    step_text = f"{response}\nObservation:\n{passage_text}"
                    rs['reasoning_history'].append(step_text)
                    rs['passages'].extend(passages)
                    rs['current_passages'] = passages

        # Calculate MC from final answers
        successes = 0
        rollout_answers = []
        for rs in rollout_states:
            answer = rs['final_answer'] or self._extract_answer(rs['reasoning_history'][-1] if rs['reasoning_history'] else '')
            rollout_answers.append(answer[:50] if answer else '')
            if gold_answer and self._check_answer(answer, gold_answer):
                successes += 1

        mc_value = successes / k

        if mc_value == 0.0:
            print(f"    [MC DEBUG] MC=0.0! K={k}, Gold: {gold_answer[:50] if gold_answer else 'None'}")
            print(f"    [MC DEBUG] Sample rollouts: {rollout_answers[:2]}")

        return mc_value

    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from model response."""
        patterns = [
            r'Action:\s*Search\s*\[\s*query\s*=\s*["\'](.+?)["\']\s*\]',
            r'Action:\s*Search\s*\[\s*["\'](.+?)["\']\s*\]',
            r'Action:\s*Search\s*\[\s*(.+?)\s*\]',
            r'Search\s*\[\s*query\s*=\s*["\'](.+?)["\']\s*\]',
            r'Search\s*\[\s*["\'](.+?)["\']\s*\]',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                query = match.group(1).strip().strip('"\'')
                if query:
                    return query
        return None

    def _build_rollout_prompt(self, state: Dict[str, Any]) -> str:
        """Build rollout prompt from state (extracted for batch generation).

        Args:
            state: Current state with 'question', 'reasoning_history', 'passages', 'current_passages'

        Returns:
            Formatted prompt string for rollout generation
        """
        lines = [f"Question: {state['question']}\n"]

        # Add previous documents titles for reference (if any exist beyond current)
        all_passages = state.get('passages', [])
        current_passages = state.get('current_passages', [])

        # Previous passages = all passages minus current passages
        if all_passages and len(all_passages) > len(current_passages):
            lines.append("Previous Documents (for reference):")
            # Show titles of previous documents
            num_previous = len(all_passages) - len(current_passages)
            for i, passage in enumerate(all_passages[:num_previous], 1):
                lines.append(f"[{i}] {passage.get('title', 'Document')}")
            lines.append("")

        # Current step's retrieved passages (full content)
        if current_passages:
            lines.append("Retrieved Information (Current):")
            for i, passage in enumerate(current_passages, 1):
                title = passage.get('title', 'Document')
                content = passage.get('text', '')
                lines.append(f"[{i}] {title}: {content}")
            lines.append("")

        # Add existing reasoning history if any
        if state.get('reasoning_history'):
            lines.append("Reasoning so far:")
            for step_text in state['reasoning_history']:
                lines.append(step_text)
            lines.append("")
            lines.append("Continue solving to reach the final answer.")
        else:
            lines.append("Solve this question and provide the final answer.")

        return "\n".join(lines)

    def _rollout(self, state: Dict[str, Any]) -> str:
        """Perform one rollout from current state.

        Generate a complete solution from the current state by continuing
        the reasoning until we reach a final answer.

        Note: For batch rollouts (MC estimation), use _monte_carlo_estimate()
        which calls batch_generate() for better performance with vLLM.

        Args:
            state: Current state with 'question' and 'reasoning_history'

        Returns:
            Final answer extracted from the rollout
        """
        if self.policy_model is None:
            # Placeholder for testing without model
            return "rollout_answer"

        # Use shared prompt builder
        prompt = self._build_rollout_prompt(state)

        # Generate complete solution (longer max_tokens for multi-hop reasoning)
        response = self.policy_model.generate_with_chat_template(
            user_message=prompt,
            max_tokens=1200,  # Increased to allow more thorough reasoning
            temperature=self.temperature,
            top_p=0.95,
        )

        # Extract final answer from response
        return self._extract_answer(response)

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if answer is correct using token-level accuracy.

        Uses token-level accuracy: percentage of gold tokens found in prediction.
        An answer is correct if accuracy >= 0.8 (80% of gold tokens present).

        Args:
            predicted: Predicted answer (raw)
            gold: Gold answer (raw)

        Returns:
            True if token accuracy >= 0.8
        """
        # Extract and normalize
        pred_extracted = extract_answer_from_text(predicted)
        gold_extracted = extract_answer_from_text(gold)

        pred_norm = normalize_answer(pred_extracted)
        gold_norm = normalize_answer(gold_extracted)

        # Check match using cover exact match
        return check_answer_match(pred_norm, gold_norm)

    def _has_answer(self, text: str) -> bool:
        """Check if text contains final answer markers.

        Checks for both:
        1. 'Final Answer:' marker (explicit instruction)
        2. 'Finish[answer=' pattern (ReAct format)
        """
        text_lower = text.lower()
        return ('final answer:' in text_lower or
                'finish[answer=' in text_lower)

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

    def _monte_carlo_estimate_batch(
        self,
        traj_states: List[TrajectoryState],
    ) -> List[float]:
        """Batch MC estimation with parallel RAG-enabled rollouts.

        All N×K rollouts are processed in parallel with RAG capability.
        When Search actions are detected, retrieval is performed in batch
        and generation continues in parallel.

        Args:
            traj_states: List of trajectory states (each has its own current_k)

        Returns:
            List of MC values (one per trajectory)
        """
        if not traj_states:
            return []

        if self.policy_model is None:
            return [np.random.uniform(0.3, 0.9) for _ in traj_states]

        max_rollout_steps = 5

        # Initialize all rollout states: N trajectories × K_i rollouts
        # Each rollout state: (traj_idx, rollout_idx, state_dict)
        all_rollout_states = []
        trajectory_ks = []

        for traj_idx, traj_state in enumerate(traj_states):
            k = traj_state.current_k
            trajectory_ks.append(k)

            for rollout_idx in range(k):
                all_rollout_states.append({
                    'traj_idx': traj_idx,
                    'rollout_idx': rollout_idx,
                    'question': traj_state.state['question'],
                    'reasoning_history': list(traj_state.state.get('reasoning_history', [])),
                    'passages': list(traj_state.state.get('passages', [])),
                    'current_passages': list(traj_state.state.get('current_passages', [])),
                    'finished': False,
                    'final_answer': '',
                    'gold_answer': traj_state.gold_answer,
                })

        total_rollouts = len(all_rollout_states)
        k_min, k_max = min(trajectory_ks), max(trajectory_ks)
        print(f"  [Batch MC+RAG] {total_rollouts} rollouts ({len(traj_states)} traj, K: {k_min}-{k_max})")

        # Iterative parallel generation with RAG
        for step in range(max_rollout_steps):
            # Get active (not finished) rollouts
            active_rollouts = [(i, rs) for i, rs in enumerate(all_rollout_states) if not rs['finished']]
            if not active_rollouts:
                break

            # Build prompts for all active rollouts
            prompts = []
            active_indices = []
            for global_idx, rs in active_rollouts:
                prompt = self._build_rollout_prompt(rs)
                formatted = self.policy_model.format_prompt_for_qwen(prompt)
                prompts.append(formatted)
                active_indices.append(global_idx)

            # Batch generate
            if hasattr(self.policy_model, 'batch_generate'):
                responses = self.policy_model.batch_generate(
                    prompts=prompts,
                    max_tokens=800,
                    temperature=self.temperature,
                    top_p=0.95,
                )
            else:
                responses = [self.policy_model.generate(p, max_tokens=800) for p in prompts]

            # Process responses and identify Search actions
            search_items = []  # (global_idx, query, response)
            for global_idx, response in zip(active_indices, responses):
                rs = all_rollout_states[global_idx]

                if self._has_answer(response):
                    rs['finished'] = True
                    rs['final_answer'] = self._extract_answer(response)
                    continue

                query = self._extract_search_query(response)
                if query:
                    search_items.append((global_idx, query, response))
                else:
                    rs['reasoning_history'].append(response)
                    rs['current_passages'] = []

            # Batch retrieve for all Search actions
            if search_items and self.retriever is not None:
                queries = [item[1] for item in search_items]

                if hasattr(self.retriever, 'batch_retrieve'):
                    all_passages = self.retriever.batch_retrieve(queries, top_k=5)
                else:
                    all_passages = [self.retriever.retrieve(q, top_k=5) for q in queries]

                for (global_idx, query, response), passages in zip(search_items, all_passages):
                    rs = all_rollout_states[global_idx]
                    passage_text = self._format_passages(passages)
                    step_text = f"{response}\nObservation:\n{passage_text}"
                    rs['reasoning_history'].append(step_text)
                    rs['passages'].extend(passages)
                    rs['current_passages'] = passages

        # Calculate MC for each trajectory
        mc_values = []
        for traj_idx, traj_state in enumerate(traj_states):
            # Get all rollouts for this trajectory
            traj_rollouts = [rs for rs in all_rollout_states if rs['traj_idx'] == traj_idx]
            k_traj = trajectory_ks[traj_idx]

            successes = 0
            for rs in traj_rollouts:
                answer = rs['final_answer'] or self._extract_answer(
                    rs['reasoning_history'][-1] if rs['reasoning_history'] else ''
                )
                if rs['gold_answer'] and self._check_answer(answer, rs['gold_answer']):
                    successes += 1

            mc_values.append(successes / k_traj)

        return mc_values


    def _generate_cot_steps_batch(
        self,
        traj_states: List[TrajectoryState],
    ) -> List[str]:
        """Generate CoT steps for multiple trajectories at once (vLLM optimized).

        Args:
            traj_states: List of trajectory states

        Returns:
            List of step contents (without "Step N:" prefix)
        """
        if not traj_states:
            return []

        if self.policy_model is None:
            # Placeholder for testing
            return [f"[CoT step {ts.current_step_num()}]" for ts in traj_states]

        # Build prompts for each trajectory
        prompts = []
        for traj_state in traj_states:
            step_num = traj_state.current_step_num()
            state = traj_state.state
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
            prompts.append(prompt)

        # Batch generate with chat template
        if hasattr(self.policy_model, 'batch_generate_with_chat_template'):
            # Use batch generation with chat template (Qwen format)
            responses = self.policy_model.batch_generate_with_chat_template(
                user_messages=prompts,
                max_tokens=800,
                temperature=self.temperature,
                top_p=0.95,
                stop_sequences=["\nStep"],
            )
        else:
            # Fallback: format and batch generate manually
            formatted_prompts = []
            for prompt in prompts:
                if hasattr(self.policy_model, 'format_prompt_for_qwen'):
                    formatted_prompts.append(self.policy_model.format_prompt_for_qwen(prompt))
                else:
                    formatted_prompts.append(prompt)

            if hasattr(self.policy_model, 'batch_generate'):
                responses = self.policy_model.batch_generate(
                    prompts=formatted_prompts,
                    max_tokens=800,
                    temperature=self.temperature,
                    top_p=0.95,
                    stop_sequences=["\nStep"],
                )
            else:
                # Sequential fallback
                responses = []
                for prompt in formatted_prompts:
                    resp = self.policy_model.generate(
                        prompt=prompt,
                        max_tokens=800,
                        temperature=self.temperature,
                        top_p=0.95,
                        stop_sequences=["\nStep"],
                    )
                    responses.append(resp)

        # Parse responses to extract step contents
        step_contents = []
        for idx, (response, traj_state) in enumerate(zip(responses, traj_states)):
            step_num = traj_state.current_step_num()

            # Remove any future steps
            import re
            next_step_match = re.search(r'\n+Step\s+\d+:', response)
            if next_step_match:
                response = response[:next_step_match.start()].strip()

            # Parse to get content without "Step N:" prefix
            content = self.step_parser.parse_single_step_response(response, step_num)
            step_contents.append(content)

        return step_contents

    def _generate_rag_queries_batch(
        self,
        traj_states: List[TrajectoryState],
    ) -> List[str]:
        """Generate RAG queries for multiple trajectories at once.

        Args:
            traj_states: List of trajectory states

        Returns:
            List of search queries
        """
        if not traj_states:
            return []

        if self.policy_model is None:
            # Fallback: use questions as-is
            return [ts.state['question'] for ts in traj_states]

        # Build prompts for query generation
        prompts = []
        for traj_state in traj_states:
            state = traj_state.state
            prompt_lines = []

            if state.get('reasoning_history'):
                # Follow-up question
                prompt_lines.extend([
                    "You are searching for information to answer a question step by step.",
                    "",
                    f"## Main Question",
                    state['question'],
                    "",
                    f"## Previous Reasoning Steps",
                ])
                for step in state['reasoning_history']:
                    prompt_lines.append(step)
                prompt_lines.extend([
                    "",
                    "## Task",
                    "Based on the reasoning so far, generate a simple follow-up search query to find the information needed to continue.",
                    "- Ask a SIMPLE question that a search engine can understand",
                    "- You may rephrase or decompose the main question if previous steps were not helpful",
                    "- Use 3-8 keywords maximum",
                    "- Do NOT write complex questions",
                    "",
                    "Respond with ONLY the search query. Do not explain yourself.",
                    "",
                    "Search query:"
                ])
            else:
                # First step - decompose the main question
                prompt_lines.extend([
                    "You are searching for information to answer a question step by step.",
                    "",
                    f"## Main Question",
                    state['question'],
                    "",
                    "## Task",
                    "Generate a simple search query to find information that will help answer this question.",
                    "- Break down the question if it requires multiple pieces of information",
                    "- Ask a SIMPLE question that a search engine can understand",
                    "- Use 3-8 keywords maximum",
                    "",
                    "Respond with ONLY the search query. Do not explain yourself.",
                    "",
                    "Search query:"
                ])

            prompts.append("\n".join(prompt_lines))

        # Batch generate queries
        if hasattr(self.policy_model, 'batch_generate_with_chat_template'):
            responses = self.policy_model.batch_generate_with_chat_template(
                user_messages=prompts,
                max_tokens=100,
                temperature=0.7,
            )
        else:
            # Fallback: format and batch generate manually
            formatted_prompts = []
            for prompt in prompts:
                if hasattr(self.policy_model, 'format_prompt_for_qwen'):
                    formatted_prompts.append(self.policy_model.format_prompt_for_qwen(prompt))
                else:
                    formatted_prompts.append(prompt)

            if hasattr(self.policy_model, 'batch_generate'):
                responses = self.policy_model.batch_generate(
                    prompts=formatted_prompts,
                    max_tokens=100,
                    temperature=0.7,
                )
            else:
                # Sequential fallback
                responses = []
                for prompt in formatted_prompts:
                    resp = self.policy_model.generate(
                        prompt=prompt,
                        max_tokens=100,
                        temperature=0.7,
                    )
                    responses.append(resp)

        # Extract queries from responses
        queries = []
        for response, traj_state in zip(responses, traj_states):
            query = response.strip().split('\n')[0].strip()
            # Remove common prefixes like "1.", "-", "*"
            import re
            query = re.sub(r'^\s*(?:\d+\.|[-*])\s*', '', query)
            queries.append(query if query else traj_state.state['question'])

        return queries

    def _generate_rag_steps_batch(
        self,
        traj_states: List[TrajectoryState],
        queries: List[str],
        passages_list: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """Generate RAG steps for multiple trajectories at once.

        Args:
            traj_states: List of trajectory states
            queries: List of search queries (one per trajectory)
            passages_list: List of passage lists (one list per trajectory)

        Returns:
            List of RAG step contents (without "Step N:" prefix)
        """
        if not traj_states:
            return []

        if self.policy_model is None:
            # Placeholder
            return [f"[RAG step {ts.current_step_num()}]" for ts in traj_states]

        # Build prompts for each trajectory
        prompts = []
        for traj_state, query, passages in zip(traj_states, queries, passages_list):
            step_num = traj_state.current_step_num()
            state = traj_state.state
            forced_steps = state.get('forced_steps', [])

            # Format passages
            passage_text = self._format_passages(passages)

            if step_num == 1:
                # Initial step with passages (instruction은 system prompt에서 처리)
                prompt_lines = [
                    f"Question: {state['question']}",
                    "",
                    f"Retrieved Documents:",
                    passage_text,
                    "",
                    f"Your search query was: {query}",
                    "",
                    f"Generate Step {step_num}.",
                ]
            else:
                # Continuation step with passages
                steps_text = "\n".join(str(step) for step in forced_steps)

                # Add previous documents titles for context
                previous_passages = state.get('passages', [])
                previous_docs_section = []
                if previous_passages:
                    previous_docs_section.append("## Previous Documents (for reference)")
                    for i, p in enumerate(previous_passages, 1):
                        previous_docs_section.append(f"[{i}] {p.get('title', 'Document')}")
                    previous_docs_section.append("")

                prompt_lines = [
                    f"## Question",
                    state['question'],
                    "",
                    f"## Previous Reasoning Steps",
                    steps_text,
                    "",
                ]

                # Add previous docs titles if any
                if previous_docs_section:
                    prompt_lines.extend(previous_docs_section)

                # instruction은 system prompt에서 처리
                prompt_lines.extend([
                    f"Retrieved Documents (Current):",
                    passage_text,
                    "",
                    f"Your search query was: {query}",
                    "",
                    f"Continue with Step {step_num}.",
                ])

            prompts.append("\n".join(prompt_lines))

        # Batch generate with chat template (allow observation for RAG)
        if hasattr(self.policy_model, 'batch_generate_with_chat_template'):
            responses = self.policy_model.batch_generate_with_chat_template(
                user_messages=prompts,
                max_tokens=800,
                temperature=self.temperature,
                top_p=0.95,
                stop_sequences=["\nStep"],
            )
        else:
            # Fallback: format and batch generate manually
            formatted_prompts = []
            for prompt in prompts:
                if hasattr(self.policy_model, 'format_prompt_for_qwen'):
                    formatted_prompts.append(self.policy_model.format_prompt_for_qwen(prompt))
                else:
                    formatted_prompts.append(prompt)

            if hasattr(self.policy_model, 'batch_generate'):
                responses = self.policy_model.batch_generate(
                    prompts=formatted_prompts,
                    max_tokens=800,
                    temperature=self.temperature,
                    top_p=0.95,
                    stop_sequences=["\nStep"],
                )
            else:
                # Sequential fallback
                responses = []
                for prompt in formatted_prompts:
                    resp = self.policy_model.generate(
                        prompt=prompt,
                        max_tokens=800,
                        temperature=self.temperature,
                        top_p=0.95,
                        stop_sequences=["\nStep"],
                    )
                    responses.append(resp)

        # Parse responses and add Observations
        step_contents = []
        for response, traj_state, query, passages in zip(responses, traj_states, queries, passages_list):
            step_num = traj_state.current_step_num()

            # Remove any future steps
            import re
            next_step_match = re.search(r'\n+Step\s+\d+:', response)
            if next_step_match:
                response = response[:next_step_match.start()].strip()

            # Parse to get content without "Step N:" prefix
            parsed_content = self.step_parser.parse_single_step_response(response, step_num)

            # Add Observation section with actual retrieved document chunks
            passage_text = self._format_passages(passages)
            observation_text = f"\nObservation:\n{passage_text}"

            # Insert Observation after Action line
            lines = parsed_content.split('\n')
            result_lines = []
            action_found = False

            for line in lines:
                result_lines.append(line)
                # Insert Observation after Action line
                if not action_found and line.strip().startswith('Action:'):
                    action_found = True
                    result_lines.append(observation_text)

            # If no Action found, append at end (fallback)
            if not action_found:
                result_lines.append(observation_text)

            final_content = '\n'.join(result_lines)
            step_contents.append(final_content)

        return step_contents

    def generate_batch_parallel(
        self,
        questions: List[Dict[str, Any]],
        num_trajectories_per_question: int = 1,
        show_progress: bool = True,
    ) -> List[AdaptiveTrajectory]:
        """Generate trajectories in parallel (reinforcement learning style).

        All trajectories are processed synchronously step-by-step:
        - Step 1: Generate all trajectories' Step 1 in one batch
        - MC estimation: All trajectories × K rollouts in one batch
        - Step 2: Generate all remaining trajectories' Step 2 in one batch
        - ... until all trajectories finish or reach max_steps

        This maximizes vLLM batch efficiency (e.g., 16 traj × 8 rollouts = 128 prompts at once).

        Args:
            questions: List of question dicts
            num_trajectories_per_question: Number of trajectories per question
            show_progress: Show progress bar

        Returns:
            List of generated trajectories
        """
        # Initialize all trajectory states
        traj_states: List[TrajectoryState] = []
        for q_data in questions:
            for n in range(num_trajectories_per_question):
                trajectory_id = f"{q_data.get('id', 'q')}_{n}"
                initial_state = {
                    'question': q_data['question'],
                    'reasoning_history': [],
                    'forced_steps': [],
                    'passages': [],
                    'current_passages': [],
                }
                traj_state = TrajectoryState(
                    trajectory_id=trajectory_id,
                    question=q_data['question'],
                    gold_answer=q_data.get('gold_answer'),
                    supporting_facts=q_data.get('supporting_facts'),
                    state=initial_state,
                    current_k=self.num_rollouts,  # Initialize with default K
                )
                traj_states.append(traj_state)

        print(f"\n[Batch Parallel] Starting generation for {len(traj_states)} trajectories")
        print(f"  Questions: {len(questions)}, Trajectories per question: {num_trajectories_per_question}")

        # Calculate baseline MC (question only) for all trajectories - BATCH
        print(f"\n[Pre-Step1] Calculating baseline MC for all {len(traj_states)} trajectories...")
        mc_questions = self._monte_carlo_estimate_batch(traj_states)
        for traj_state, mc_q in zip(traj_states, mc_questions):
            traj_state.mc_question = mc_q
            traj_state.mc_prev = mc_q
            print(f"  [{traj_state.trajectory_id}] MC(question) = {mc_q:.3f}")

        # Main generation loop - process all trajectories step by step
        for step_num in range(1, self.max_steps + 1):
            print(f"\n{'='*80}")
            print(f"STEP {step_num} - Processing batch")
            print(f"{'='*80}")

            # Filter active trajectories (not finished)
            active_states = [ts for ts in traj_states if not ts.is_finished]
            if not active_states:
                print(f"  All trajectories finished. Stopping.")
                break

            print(f"  Active trajectories: {len(active_states)}/{len(traj_states)}")

            # Update dynamic K based on question difficulty (only on step 1)
            if step_num == 1 and self.use_dynamic_k:
                print(f"  [Dynamic K] Updating K for each trajectory based on MC(question)...")
                for ts in active_states:
                    if ts.mc_question < 0.1:
                        difficulty = "hard"
                        ts.current_k = self.k_hard
                    elif ts.mc_question < 0.9:
                        difficulty = "medium"
                        ts.current_k = self.k_medium
                    else:
                        difficulty = "easy"
                        ts.current_k = self.k_easy
                    print(f"    [{ts.trajectory_id}] MC(q)={ts.mc_question:.3f} → {difficulty} → K={ts.current_k}")

                # Also update for finished trajectories (for consistency)
                for ts in traj_states:
                    if ts.is_finished:
                        continue
                    if ts.mc_question < 0.1:
                        ts.current_k = self.k_hard
                    elif ts.mc_question < 0.9:
                        ts.current_k = self.k_medium
                    else:
                        ts.current_k = self.k_easy

            # (A) Generate CoT steps for all active trajectories - BATCH
            print(f"  Generating CoT steps (batch size: {len(active_states)})...")
            cot_contents = self._generate_cot_steps_batch(active_states)

            # Create temporary states with CoT steps
            cot_states_temp = []
            for ts, cot_content in zip(active_states, cot_contents):
                cot_text = f"Step {step_num}: {cot_content}"
                temp_state = self._apply_cot_step(ts.state, cot_text, step_num)
                cot_states_temp.append(temp_state)

            # (B) Calculate MC for all CoT steps - BATCH
            print(f"  Calculating MC for CoT steps (batch)...")
            # Create temporary TrajectoryState objects for MC estimation
            temp_traj_states = []
            for ts, temp_state in zip(active_states, cot_states_temp):
                temp_ts = TrajectoryState(
                    trajectory_id=ts.trajectory_id,
                    question=ts.question,
                    gold_answer=ts.gold_answer,
                    supporting_facts=ts.supporting_facts,
                    state=temp_state,
                )
                # Skip MC if already failed
                if ts.failed_after_rag:
                    temp_ts.mc_prev = 0.0
                temp_traj_states.append(temp_ts)

            # Batch MC estimation (only for non-failed trajectories)
            non_failed_indices = [i for i, ts in enumerate(active_states) if not ts.failed_after_rag]
            failed_indices = [i for i, ts in enumerate(active_states) if ts.failed_after_rag]

            mc_cot_values = [0.0] * len(active_states)
            if non_failed_indices:
                non_failed_temp = [temp_traj_states[i] for i in non_failed_indices]
                non_failed_mc = self._monte_carlo_estimate_batch(non_failed_temp)
                for idx, mc_val in zip(non_failed_indices, non_failed_mc):
                    mc_cot_values[idx] = mc_val

            # (C) Process each trajectory based on intent and RPE
            # We need to handle different routing for each trajectory
            # Collect trajectories that need RAG
            rag_needed_indices = []
            rag_needed_states = []
            rag_provided_queries = []

            for idx, (ts, cot_content, mc_cot) in enumerate(zip(active_states, cot_contents, mc_cot_values)):
                print(f"\n  [{ts.trajectory_id}] Step {step_num}:")
                print(f"    CoT MC = {mc_cot:.3f}, MC_prev = {ts.mc_prev:.3f}")

                # Check intent (supports both "Action:" and "**Action:**" markdown format)
                content_lower = cot_content.lower()
                has_finish_intent = "action: finish" in content_lower or "action:finish" in content_lower
                has_search_intent = "action: search" in content_lower or "action:search" in content_lower

                # PATH 0: VOLUNTARY FINISH
                if has_finish_intent:
                    print(f"    → Finish intent detected, terminating")
                    cot_text = f"Step {step_num}: {cot_content}"
                    rpe = mc_cot / (ts.mc_prev + 0.01)
                    step = AdaptiveStep(
                        step_id=len(ts.steps),
                        step_type=StepType.COT,
                        text=cot_text,
                        content=cot_content,
                        used_passages=[],
                        mc_before=ts.mc_prev,
                        mc_after=mc_cot,
                        rpe=rpe,
                        label='good',
                        thought=cot_content,
                        action="Finish",
                        action_input=None,
                        observation=None,
                        sub_answer=None,
                        metadata={'voluntary_finish': True},
                    )
                    ts.steps.append(step)
                    ts.state = cot_states_temp[idx]
                    ts.is_finished = True
                    ts.finish_reason = "voluntary_finish"
                    continue

                # PATH A: VOLUNTARY SEARCH
                if has_search_intent:
                    print(f"    → Search intent detected (voluntary)")
                    # Extract query
                    import re
                    query_match = re.search(r'Search\s*\[\s*(?:query\s*=\s*)?["\']?(.+?)["\']?\s*\]', cot_content, re.IGNORECASE)
                    model_query = query_match.group(1).strip() if query_match else None
                    rag_needed_indices.append(idx)
                    rag_needed_states.append(ts)
                    rag_provided_queries.append(model_query)
                    continue

                # PATH B: VOLUNTARY REASONING - Verify with MC
                print(f"    → Reason intent (voluntary) - checking RPE")
                rpe_cot = mc_cot / (ts.mc_prev + 0.01)
                print(f"    RPE = {rpe_cot:.3f}")

                if ts.failed_after_rag or rpe_cot > 0.79:
                    # Accept CoT step
                    label = 'good' if rpe_cot > 0.79 else 'bad'
                    print(f"    → Accept CoT (label={label})")
                    cot_text = f"Step {step_num}: {cot_content}"
                    has_answer = self._has_answer(cot_content)
                    intermediate_ans = extract_intermediate_answer(cot_content)

                    step = AdaptiveStep(
                        step_id=len(ts.steps),
                        step_type=StepType.COT,
                        text=cot_text,
                        content=cot_content,
                        used_passages=[],
                        mc_before=ts.mc_prev if not ts.failed_after_rag else 0.0,
                        mc_after=mc_cot,
                        rpe=rpe_cot,
                        label=label,
                        thought=cot_content,
                        action="Reason",
                        action_input=None,
                        observation=intermediate_ans,
                        sub_answer=None,
                        metadata={'accepted': 'cot' if not ts.failed_after_rag else 'cot_after_failure', 'threshold': 0.8},
                    )
                    ts.steps.append(step)
                    ts.state = cot_states_temp[idx]
                    ts.mc_prev = mc_cot

                    if has_answer:
                        print(f"    → Final answer found, terminating")
                        ts.is_finished = True
                        ts.finish_reason = "answer_found"
                else:
                    # RPE < 0.8: BACKTRACK and force RAG
                    print(f"    → RPE < 0.8, BACKTRACKING and forcing RAG")
                    # Save rejected segment
                    rejected_segment = {
                        'step_num': step_num,
                        'content': cot_content,
                        'text': f"Step {step_num}: {cot_content}",
                        'mc_before': ts.mc_prev,
                        'mc_after': mc_cot,
                        'rpe': rpe_cot,
                        'reason': 'low_rpe',
                        'context': ts.state.get('reasoning_history', [])[-3:],
                    }
                    ts.rejected_segments.append(rejected_segment)
                    # Force RAG
                    rag_needed_indices.append(idx)
                    rag_needed_states.append(ts)
                    rag_provided_queries.append(None)  # Generate new query

            # (D) Process RAG for all trajectories that need it - BATCH
            if rag_needed_states:
                print(f"\n  Processing RAG for {len(rag_needed_states)} trajectories (batch)...")

                # Generate queries (batch) - only for those without provided query
                queries_to_generate = []
                query_indices = []
                for i, query in enumerate(rag_provided_queries):
                    if query is None:
                        queries_to_generate.append(rag_needed_states[i])
                        query_indices.append(i)

                if queries_to_generate:
                    print(f"    Generating {len(queries_to_generate)} RAG queries (batch)...")
                    generated_queries = self._generate_rag_queries_batch(queries_to_generate)
                    for qi, gen_q in zip(query_indices, generated_queries):
                        rag_provided_queries[qi] = gen_q

                final_queries = rag_provided_queries

                # Retrieve passages (sequential for now - retriever may not support batch)
                print(f"    Retrieving passages for {len(final_queries)} queries...")
                passages_list = []
                for query in final_queries:
                    passages = self.retriever.retrieve(query, top_k=self.top_k_passages)
                    passages_list.append(passages)

                # Generate RAG steps (batch)
                print(f"    Generating RAG steps (batch)...")
                rag_contents = self._generate_rag_steps_batch(rag_needed_states, final_queries, passages_list)

                # Apply RAG steps and calculate MC
                rag_temp_states = []
                rag_temp_traj_states = []
                for ts, query, passages, rag_content in zip(rag_needed_states, final_queries, passages_list, rag_contents):
                    temp_rag_state = self._apply_rag_step(ts.state, query, passages, step_num, rag_content)
                    rag_temp_states.append(temp_rag_state)
                    temp_ts = TrajectoryState(
                        trajectory_id=ts.trajectory_id,
                        question=ts.question,
                        gold_answer=ts.gold_answer,
                        supporting_facts=ts.supporting_facts,
                        state=temp_rag_state,
                    )
                    rag_temp_traj_states.append(temp_ts)

                # Calculate MC for RAG steps (batch)
                print(f"    Calculating MC for RAG steps (batch)...")
                mc_rag_values = self._monte_carlo_estimate_batch(rag_temp_traj_states)

                # Apply RAG results
                for ts, query, passages, rag_content, temp_rag_state, mc_rag in zip(
                    rag_needed_states, final_queries, passages_list, rag_contents, rag_temp_states, mc_rag_values
                ):
                    rpe_rag = mc_rag / (ts.mc_prev + 0.01)
                    label = 'good' if rpe_rag > 0.79 else 'bad'

                    parsed_fields = parse_rag_content(rag_content)
                    has_answer = self._has_answer(rag_content)
                    parsed_action = parsed_fields.get('action') or "Search"

                    rag_text = f"Step {step_num}: {rag_content}"
                    step = AdaptiveStep(
                        step_id=len(ts.steps),
                        step_type=StepType.RAG,
                        text=rag_text,
                        content=rag_content,
                        used_passages=passages,
                        mc_before=ts.mc_prev,
                        mc_after=mc_rag,
                        rpe=rpe_rag,
                        label=label,
                        thought=parsed_fields.get('thought'),
                        action=parsed_action,
                        action_input=parsed_fields.get('action_input'),
                        observation=parsed_fields.get('observation'),
                        sub_answer=parsed_fields.get('sub_answer'),
                        metadata={'voluntary_search': query in rag_provided_queries},
                    )
                    ts.steps.append(step)
                    ts.state = temp_rag_state
                    ts.mc_prev = mc_rag

                    # Check for failure (GenPRM-style)
                    if mc_rag < 0.01:
                        ts.failed_after_rag = True
                        print(f"  [{ts.trajectory_id}] RAG failed (MC < 0.01), future steps will have MC=0")

                    # Check for termination
                    if has_answer:
                        print(f"  [{ts.trajectory_id}] Final answer found, terminating")
                        ts.is_finished = True
                        ts.finish_reason = "answer_found"

        # Mark remaining active trajectories as finished (max_steps reached)
        for ts in traj_states:
            if not ts.is_finished:
                ts.is_finished = True
                ts.finish_reason = "max_steps"

        # Convert to AdaptiveTrajectory objects
        print(f"\n[Batch Parallel] Finalizing {len(traj_states)} trajectories...")
        trajectories = []
        for ts in traj_states:
            final_answer = self._extract_final_answer(ts.steps)
            is_correct = self._check_answer(final_answer, ts.gold_answer) if ts.gold_answer else None

            trajectory = AdaptiveTrajectory(
                trajectory_id=ts.trajectory_id,
                question=ts.question,
                steps=ts.steps,
                final_answer=final_answer,
                gold_answer=ts.gold_answer,
                supporting_facts=ts.supporting_facts,
                is_correct=is_correct,
                metadata={
                    'num_steps': len(ts.steps),
                    'has_rag': any(s.step_type == StepType.RAG for s in ts.steps),
                    'num_backtracks': len(ts.rejected_segments),
                    'finish_reason': ts.finish_reason,
                },
                rejected_segments=ts.rejected_segments,
            )
            trajectories.append(trajectory)

        print(f"\n[Batch Parallel] Generation complete!")
        print(f"  Total trajectories: {len(trajectories)}")
        print(f"  Correct: {sum(1 for t in trajectories if t.is_correct)}/{len([t for t in trajectories if t.gold_answer])}")

        return trajectories
