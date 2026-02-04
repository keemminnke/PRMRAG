#!/usr/bin/env python3
"""Convert ReAct format trajectories to XML tag format.

ReAct format:
    Thought: ...
    Action: Search[query="..."]
    Observation: ...

XML format:
    <think>...</think>
    <search>...</search>
    <documents>...</documents>
"""

import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_react_step(text: str) -> dict:
    """Parse ReAct format step into components."""
    result = {
        'think': '',
        'action_type': '',  # search, reason, finish
        'action_content': '',
        'observation': ''
    }

    # Extract Thought
    thought_match = re.search(
        r'Thought:\s*(.+?)(?=\nAction:|$)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if thought_match:
        result['think'] = thought_match.group(1).strip()

    # Extract Action
    # Search[query="..."]
    search_match = re.search(
        r'Action:\s*Search\[query=["\']?(.+?)["\']?\]',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if search_match:
        result['action_type'] = 'search'
        result['action_content'] = search_match.group(1).strip()

    # Finish[answer="..."]
    finish_match = re.search(
        r'Action:\s*Finish\[answer=["\']?(.+?)["\']?\]',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if finish_match:
        result['action_type'] = 'finish'
        result['action_content'] = finish_match.group(1).strip()

    # Reason[content="..."]
    reason_match = re.search(
        r'Action:\s*Reason\[content=["\']?(.+?)["\']?\]',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if reason_match:
        result['action_type'] = 'reason'
        result['action_content'] = reason_match.group(1).strip()

    # If no action found, check for simple action type
    if not result['action_type']:
        simple_action = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
        if simple_action:
            action = simple_action.group(1).lower()
            if action in ['search', 'finish', 'reason']:
                result['action_type'] = action

    # Extract Observation
    obs_match = re.search(
        r'Observation:\s*(.+?)$',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if obs_match:
        result['observation'] = obs_match.group(1).strip()

    return result


def convert_to_xml(parsed: dict) -> str:
    """Convert parsed components to XML format."""
    parts = []

    # Think (combine with reason content if exists)
    think_content = parsed['think']
    if parsed['action_type'] == 'reason' and parsed['action_content']:
        if parsed['action_content'] not in think_content:
            think_content = f"{think_content} {parsed['action_content']}".strip()

    if think_content:
        parts.append(f"<think>{think_content}</think>")

    # Action
    if parsed['action_type'] == 'search':
        parts.append(f"<search>{parsed['action_content']}</search>")
    elif parsed['action_type'] == 'finish':
        parts.append(f"<answer>{parsed['action_content']}</answer>")
    # reason은 think에 통합됨

    # Observation
    if parsed['observation']:
        parts.append(f"<documents>{parsed['observation']}</documents>")

    return '\n'.join(parts)


def convert_trajectory(trajectory: dict) -> dict:
    """Convert entire trajectory from ReAct to XML format."""
    new_trajectory = trajectory.copy()
    new_steps = []

    for step in trajectory.get('steps', []):
        new_step = step.copy()

        # Get text content (could be in 'text', 'content', or combined)
        text = step.get('content', '') or step.get('text', '')

        if text:
            # Parse ReAct format
            parsed = parse_react_step(text)

            # Convert to XML
            xml_content = convert_to_xml(parsed)

            # Update step
            new_step['content'] = xml_content
            new_step['text'] = xml_content

            # Store parsed components for easier access
            new_step['think'] = parsed['think']
            new_step['action_type'] = parsed['action_type']
            new_step['action_content'] = parsed['action_content']
            new_step['observation'] = parsed['observation']

        new_steps.append(new_step)

    new_trajectory['steps'] = new_steps
    return new_trajectory


def main():
    parser = argparse.ArgumentParser(description="Convert ReAct to XML format")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Converting: {input_path} -> {output_path}")

    # Load and convert
    trajectories = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    print(f"Loaded {len(trajectories)} trajectories")

    # Convert
    converted = []
    for traj in tqdm(trajectories, desc="Converting"):
        converted.append(convert_trajectory(traj))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for traj in converted:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path}")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE CONVERSION")
    print("=" * 60)

    sample_step = converted[0]['steps'][0]
    print(f"\nConverted content:")
    print(sample_step.get('content', '')[:500])


if __name__ == "__main__":
    main()
