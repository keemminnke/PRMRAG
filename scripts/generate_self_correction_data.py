import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models.policy_model_vllm import PolicyModelVLLM

# System prompt for synthesis (Qwen2.5-7B)
SYNTHESIS_SYSTEM_PROMPT = """You are an expert AI reasoning assistant. 
Your task is to rewrite a flawed reasoning step into a "self-correction" step.
You will be given a Question, the History of previous steps, a Flawed Step, and Critic Feedback explaining why it was wrong.

# TASK
Write a new version of the step that:
1. Starts with <think> tag.
2. In the <think> part, the agent should realize its mistake based on the evidence (as if it discovered the error itself) and explain the correct logic.
3. Follow with the correct Action: either <search>query</search> or <answer>final_answer</answer>.

# RULES
- Do NOT mention "Critic" or "Feedback" in the output. It must feel like the agent is thinking for itself.
- Use the exact same XML format: <think>...</think> followed by <search>...</search> or <answer>...</answer>.
- The new step should be logically sound and directly address the problem identified in the feedback.

# EXAMPLE
Question: Who is Harry Einstein's son in Curb Your Enthusiasm?
History: Step 1: <think>Search for Harry Einstein family</think><search>...</search><documents>...Bob Einstein plays Marty Funkhouser...</documents>
Flawed Step: <think>Bob Einstein is Harry's son.</think><answer>Bob Einstein</answer>
Critic Feedback: The documents say Bob Einstein *plays* a character, but don't confirm he is the son. Actually, Harry Einstein is the father of Albert Brooks and Bob Einstein in real life.

New Step:
<think>Wait, looking at the documents again, it says Bob Einstein plays Marty Funkhouser in the show. I shouldn't assume he is the son of Harry Einstein just because of the name. I need to verify the real-life relationship between Harry Einstein and the actors in the show to be sure who his son is.</think>
<search>Harry Einstein children actors</search>
"""

def build_synthesis_prompt(tokenizer, question: str, history_steps: List[Dict], bad_step: Dict, critic_reasoning: str) -> str:
    """Build prompt for the model to generate a self-corrected step."""
    
    # Format history
    history_text = ""
    for i, step in enumerate(history_steps):
        history_text += f"Step {i+1}: "
        if step.get('think'): history_text += f"<think>{step['think']}</think>"
        if step.get('search'): history_text += f"<search>{step['search']}</search>"
        elif step.get('answer'): history_text += f"<answer>{step['answer']}</answer>"
        if step.get('documents'): history_text += f"<documents>{step['documents'][:200]}...</documents>"
        history_text += "\n"

    # Format flawed step
    flawed_text = ""
    if bad_step.get('think'): flawed_text += f"<think>{bad_step['think']}</think>"
    if bad_step.get('search'): flawed_text += f"<search>{bad_step['search']}</search>"
    elif bad_step.get('answer'): flawed_text += f"<answer>{bad_step['answer']}</answer>"

    user_content = (
        f"Question: {question}\n\n"
        f"History:\n{history_text}\n"
        f"Flawed Step:\n{flawed_text}\n\n"
        f"Critic Feedback: {critic_reasoning}\n\n"
        "Generate the New Step:"
    )

    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to critic_results_per_trajectory.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to save self-correction pairs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of BAD steps to process")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # 1. Load BAD steps
    print(f"Loading BAD steps from {args.input}...")
    bad_steps_data = []
    with open(args.input, 'r') as f:
        for line in f:
            traj = json.loads(line)
            for i, step in enumerate(traj.get('steps', [])):
                if step.get('critic_label') == 0:
                    history = traj['steps'][:i]
                    bad_steps_data.append({
                        'question': traj['question'],
                        'history': history,
                        'bad_step': step,
                        'critic_reasoning': step.get('critic_reasoning', ''),
                        'gold_answer': traj.get('gold_answer', ''),
                        'trajectory_id': traj.get('trajectory_id', ''),
                        'step_id': step.get('step_id', i+1)
                    })
                    if len(bad_steps_data) >= args.limit:
                        break
            if len(bad_steps_data) >= args.limit:
                break
    
    print(f"Found {len(bad_steps_data)} BAD steps to regenerate.")

    if not bad_steps_data:
        print("No BAD steps found. Exit.")
        return

    # 2. Initialize Model
    model = PolicyModelVLLM(
        model_name=args.model,
        max_new_tokens=512,
        temperature=0.7,
        gpu_memory_utilization=0.8
    )

    # 3. Batch Generate
    all_prompts = [
        build_synthesis_prompt(model.tokenizer, d['question'], d['history'], d['bad_step'], d['critic_reasoning'])
        for d in bad_steps_data
    ]

    print(f"Starting batch generation (batch_size={args.batch_size})...")
    generated_steps = model.batch_generate(all_prompts)

    # 4. Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        for i, (data, gen_step) in enumerate(zip(bad_steps_data, generated_steps)):
            # Format history for the final prompt
            hist_parts = []
            for h in data['history']:
                p = []
                if h.get('think'): p.append(f"<think>{h['think']}</think>")
                if h.get('search'): p.append(f"<search>{h['search']}</search>")
                elif h.get('answer'): p.append(f"<answer>{h['answer']}</answer>")
                if h.get('documents'): p.append(f"<documents>{h['documents']}</documents>")
                hist_parts.append("\n".join(p))
            
            history_str = "\n".join(hist_parts)
            user_msg = f"Question: {data['question']}"
            if history_str:
                user_msg += f"\n\n{history_str}"
            
            # Simple manual prompt building for final training data to avoid import issues
            # We want it to look like a standard prompt for the policy model
            final_prompt = f"<|im_start|>system\nYou are an advanced AI agent capable of Adaptive RAG (Retrieval-Augmented Generation)...<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
            
            record = {
                'trajectory_id': data['trajectory_id'],
                'step_id': data['step_id'],
                'prompt': final_prompt,
                'completion': gen_step,
                'metadata': {
                    'original_bad_step': data['bad_step'],
                    'critic_reasoning': data['critic_reasoning'],
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("Done!")

if __name__ == "__main__":
    main()
