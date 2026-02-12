"""Critic-guided trajectory regenerator.

Takes a truncated trajectory (good steps) and generates a continuation
from the truncation point, using the policy model with the same action
space (search/answer) as SimpleTrajectoryGenerator.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from .truncator import TruncationResult


class CriticGuidedRegenerator:
    """Regenerate trajectories from truncation point."""

    def __init__(
        self,
        policy_model,
        retriever,
        max_steps: int = 10,
        top_k_passages: int = 5,
        temperature: float = 0.7,
    ):
        """Initialize regenerator.

        Args:
            policy_model: PolicyModelVLLM instance
            retriever: Retriever with .retrieve() method
            max_steps: Maximum total steps (including good steps)
            top_k_passages: Number of passages to retrieve per search
            temperature: Sampling temperature for generation
        """
        self.policy_model = policy_model
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k_passages = top_k_passages
        self.temperature = temperature

    def build_regeneration_prompt(self, truncation_result: TruncationResult) -> str:
        """Build the user message for regeneration.

        Structure:
            Question: {question}
            {good steps reconstructed as XML}
            [Critic Feedback] Your Step N was incorrect.
            Continue solving from Step N.

        Args:
            truncation_result: Result from TrajectoryTruncator

        Returns:
            User message string (to be wrapped with format_prompt_for_qwen)
        """
        parts = [f"Question: {truncation_result.question}"]

        # Reconstruct good steps as they appeared in the original trajectory
        for step in truncation_result.good_steps:
            step_text = self._reconstruct_step_text(step)
            parts.append(step_text)

        # Add critic feedback (simple: step N was incorrect)
        bad_step_id = truncation_result.bad_step_id
        parts.append(
            f"\n[Critic Feedback] Your Step {bad_step_id} was incorrect.\n\n"
            f"Continue solving from Step {bad_step_id}."
        )

        return "\n\n".join(parts)

    def _reconstruct_step_text(self, step: Dict[str, Any]) -> str:
        """Reconstruct a step's text from its parsed fields.

        Steps have fields: think, search, answer, documents.
        We reconstruct the XML format that the model originally produced.
        """
        parts = []
        if step.get('think'):
            parts.append(f"<think>{step['think']}</think>")
        if step.get('search'):
            parts.append(f"<search>{step['search']}</search>")
        if step.get('documents'):
            parts.append(f"<documents>\n{step['documents']}\n</documents>")
        if step.get('answer'):
            parts.append(f"<answer>{step['answer']}</answer>")
        return "\n".join(parts)

    def regenerate_single(
        self,
        truncation_result: TruncationResult,
    ) -> Dict[str, Any]:
        """Regenerate a single trajectory from truncation point.

        Uses the same generation loop as SimpleTrajectoryGenerator:
        generate → parse action → if search: retrieve → next step → until finish or max_steps.

        Args:
            truncation_result: Truncated trajectory

        Returns:
            Dict with regenerated trajectory data
        """
        # Build the initial prompt with good steps + feedback
        user_message = self.build_regeneration_prompt(truncation_result)

        # Track accumulated content for continuation prompts
        accumulated_content = user_message

        # Generate new steps from truncation point
        new_steps = []
        remaining_steps = self.max_steps - len(truncation_result.good_steps)

        for step_offset in range(remaining_steps):
            step_num = truncation_result.bad_step_id + step_offset

            # Format prompt and generate
            formatted_prompt = self.policy_model.format_prompt_for_qwen(accumulated_content)
            response = self.policy_model.generate(
                prompt=formatted_prompt,
                max_tokens=800,
                temperature=self.temperature,
                top_p=0.95,
                stop_sequences=["<documents>", "\n<documents>"],
            )

            step_content = self._parse_step_response(response, step_num)
            action_type, action_input = self._parse_action(step_content)

            if action_type == "finish":
                new_steps.append({
                    'step_id': step_num,
                    'step_type': 'answer',
                    'text': step_content,
                    **self._parse_step_fields(step_content),
                })
                accumulated_content += "\n\n" + step_content
                break

            elif action_type == "search":
                query = action_input or truncation_result.question
                passages = self.retriever.retrieve(query, top_k=self.top_k_passages)
                observation = self._format_observation(passages)
                full_content = f"{step_content}\n{observation}"

                new_steps.append({
                    'step_id': step_num,
                    'step_type': 'search',
                    'text': full_content,
                    **self._parse_step_fields(full_content),
                })
                accumulated_content += "\n\n" + full_content

            else:
                # Reason step (no action)
                new_steps.append({
                    'step_id': step_num,
                    'step_type': 'reason',
                    'text': step_content,
                    **self._parse_step_fields(step_content),
                })
                accumulated_content += "\n\n" + step_content

        # Extract final answer from new steps
        final_answer = self._extract_final_answer(new_steps)
        is_correct = self._check_answer(final_answer, truncation_result.gold_answer)

        # Combine good steps + new steps
        all_steps = []
        for step in truncation_result.good_steps:
            all_steps.append({
                'step_id': step['step_id'],
                'step_type': step['step_type'],
                'think': step.get('think', ''),
                'search': step.get('search', ''),
                'documents': step.get('documents', ''),
                'answer': step.get('answer', ''),
                'source': 'original',
            })
        for step in new_steps:
            step['source'] = 'regenerated'
            all_steps.append(step)

        return {
            'trajectory_id': truncation_result.trajectory_id + '_regen',
            'question_id': truncation_result.question_id,
            'question': truncation_result.question,
            'gold_answer': truncation_result.gold_answer,
            'predicted_answer': final_answer,
            'is_correct': is_correct,
            'case': 'B',
            'steps': all_steps,
            'metadata': {
                'original_trajectory_id': truncation_result.trajectory_id,
                'truncated_at_step': truncation_result.bad_step_id,
                'num_good_steps': len(truncation_result.good_steps),
                'num_new_steps': len(new_steps),
                'num_total_steps': len(all_steps),
                'original_predicted_answer': truncation_result.original_predicted_answer,
            },
        }

    def regenerate_batch(
        self,
        truncation_results: List[TruncationResult],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Regenerate multiple trajectories sequentially.

        Sequential processing is necessary because each trajectory has a different
        prefix length, making batching impractical.

        Args:
            truncation_results: List of truncated trajectories
            show_progress: Whether to print progress

        Returns:
            List of regenerated trajectory dicts
        """
        results = []
        for i, tr in enumerate(truncation_results):
            if show_progress:
                print(f"  Regenerating {i+1}/{len(truncation_results)}: "
                      f"{tr.question_id} (truncated at step {tr.bad_step_id}, "
                      f"{len(tr.good_steps)} good steps)")
            result = self.regenerate_single(tr)
            results.append(result)

            if show_progress and (i + 1) % 10 == 0:
                correct = sum(1 for r in results if r['is_correct'])
                print(f"    Progress: {i+1}/{len(truncation_results)}, "
                      f"correct so far: {correct}/{len(results)} "
                      f"({correct/len(results)*100:.1f}%)")

        return results

    # === Helper methods (replicated from SimpleTrajectoryGenerator) ===

    def _parse_step_response(self, response: str, step_num: int) -> str:
        """Parse step content from response, removing future steps."""
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        step_prefix = re.match(rf'^Step\s+{step_num}:\s*', response)
        if step_prefix:
            response = response[step_prefix.end():]

        return response.strip()

    def _parse_action(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse action from step content (XML format)."""
        answer_match = re.search(r'<answer>(.+?)</answer>', content, re.DOTALL)
        if answer_match:
            return ('finish', answer_match.group(1).strip())

        search_match = re.search(r'<search>(.+?)</search>', content, re.DOTALL)
        if search_match:
            return ('search', search_match.group(1).strip())

        # Legacy format
        finish_match = re.search(
            r'Action:\s*Finish\[answer=["\']?(.+?)["\']?\]',
            content, re.IGNORECASE | re.DOTALL
        )
        if finish_match:
            return ('finish', finish_match.group(1).strip())

        search_match = re.search(
            r'Action:\s*Search\[query=["\']?(.+?)["\']?\]',
            content, re.IGNORECASE | re.DOTALL
        )
        if search_match:
            return ('search', search_match.group(1).strip())

        think_match = re.search(r'<think>(.+?)</think>', content, re.DOTALL)
        if think_match:
            return ('reason', think_match.group(1).strip())

        if any(marker in content.lower() for marker in
               ['final answer:', 'the answer is', 'therefore, the answer']):
            return ('finish', None)

        return (None, None)

    def _format_observation(self, passages: List[Dict[str, Any]]) -> str:
        """Format retrieved passages as XML <documents> tag."""
        obs_parts = []
        for i, p in enumerate(passages, 1):
            title = p.get('title', 'Unknown')
            text = p.get('text', p.get('content', ''))[:500]
            obs_parts.append(f"[{i}] {title}: {text}")
        return "<documents>\n" + "\n".join(obs_parts) + "\n</documents>"

    def _parse_step_fields(self, content: str) -> Dict[str, str]:
        """Parse XML fields from step content."""
        fields = {}
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            fields['think'] = think_match.group(1).strip()
        search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
        if search_match:
            fields['search'] = search_match.group(1).strip()
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            fields['answer'] = answer_match.group(1).strip()
        docs_match = re.search(r'<documents>(.*?)</documents>', content, re.DOTALL)
        if docs_match:
            fields['documents'] = docs_match.group(1).strip()
        return fields

    def _extract_final_answer(self, steps: List[Dict[str, Any]]) -> str:
        """Extract final answer from generated steps."""
        if not steps:
            return ""

        last_content = steps[-1].get('text', '')

        answer_match = re.search(r'<answer>(.+?)</answer>', last_content, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        # Check parsed answer field
        if steps[-1].get('answer'):
            return steps[-1]['answer']

        finish_match = re.search(
            r'Finish\[answer=["\']?(.+?)["\']?\]',
            last_content, re.IGNORECASE | re.DOTALL
        )
        if finish_match:
            return finish_match.group(1).strip()

        final_match = re.search(r'final\s+answer:\s*(.+)', last_content, re.IGNORECASE)
        if final_match:
            return final_match.group(1).strip().split('\n')[0]

        answer_match = re.search(r'the\s+answer\s+is\s*:?\s*(.+)', last_content, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip().split('\n')[0]

        return ""

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold (cover EM)."""
        if not predicted or not gold:
            return False
        pred_norm = predicted.strip().lower()
        gold_norm = gold.strip().lower()
        return pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm
