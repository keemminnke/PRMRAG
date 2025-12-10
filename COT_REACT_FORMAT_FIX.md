# CoT Step ReAct Format Parsing

## Problem

Previously, only RAG steps were parsed into structured ReAct format fields (`thought`, `action`, `observation`, `sub_answer`). CoT steps had these fields set to `None`, making training data inconsistent:

```python
# OLD - CoT steps
step = AdaptiveStep(
    content="Thought: ... Action: Search[...] ...",  # Full text in content
    thought=None,        # ❌ Not parsed
    action="Reason",     # ❌ Hardcoded
    observation=None,    # ❌ Not parsed
    sub_answer=None,     # ❌ Not parsed
)

# RAG steps were already parsed
step = AdaptiveStep(
    content="Thought: ... Action: Search[...] Observation: ...",
    thought="...",       # ✅ Parsed
    action="Search",     # ✅ Parsed
    observation="...",   # ✅ Parsed
    sub_answer="...",    # ✅ Parsed
)
```

## Why This Matters for Training

When fine-tuning the model, having consistent structured fields is crucial:

1. **Data Filtering**: Can filter by action type (Search vs Reason)
2. **Quality Control**: Can validate that thought/observation are properly formatted
3. **Curriculum Learning**: Can train on specific reasoning patterns
4. **Loss Calculation**: Can apply different losses to different components
5. **Format Enforcement**: Model learns consistent ReAct format

## Solution

Now all CoT steps are parsed using the same `parse_rag_content()` function that RAG steps use:

```python
# NEW - CoT steps are parsed too
parsed_fields = parse_rag_content(cot_step_content)

step = AdaptiveStep(
    content="Thought: ... Action: Search[...] ...",
    thought=parsed_fields['thought'],       # ✅ Parsed
    action=parsed_fields['action'] or "Reason",  # ✅ Parsed or default
    action_input=parsed_fields['action_input'],  # ✅ Parsed
    observation=parsed_fields['observation'],    # ✅ Parsed
    sub_answer=parsed_fields['sub_answer'],      # ✅ Parsed
)
```

## Implementation Details

### Changes Made

Modified 3 locations in `adaptive_generator.py` where CoT steps are created:

1. **Step 1** (line ~298-319)
2. **Step 2+ after RAG failure** (line ~351-372)
3. **Step 2+ normal case** (line ~392-413)

All now include:
```python
# Parse CoT content into structured ReAct format
parsed_fields = parse_rag_content(cot_step_content)
has_answer = self._has_answer(cot_step_content)

step = AdaptiveStep(
    ...
    # Add parsed structured fields (same as RAG)
    thought=parsed_fields['thought'],
    action=parsed_fields['action'] if parsed_fields['action'] else "Reason",
    action_input=parsed_fields['action_input'],
    observation=parsed_fields['observation'],
    sub_answer=parsed_fields['sub_answer'],
    ...
)
```

### Parsing Function

Uses existing `parse_rag_content()` function (line 32-80):

```python
def parse_rag_content(content: str) -> Dict[str, Optional[str]]:
    """Parse RAG/CoT step content into structured fields.

    Expected format:
        Thought: <reasoning>
        Action: Search[query="..."] or Finish[answer="..."]
        Observation: <retrieved info>
        Sub-answer: <intermediate answer>
    """
    # Uses regex to extract:
    # - thought: Thought: ... (until next section)
    # - action: Search/Finish from Action:
    # - action_input: query/answer from Search[query="..."]
    # - observation: Observation: ... (until next section)
    # - sub_answer: Sub-answer: ... (until next section)
```

## Expected Format in Generated Steps

CoT steps should now follow ReAct format:

```
Thought: To answer this question, I need to find when Arthur's Magazine was founded.
Action: Search[query="Arthur's Magazine founding year"]
```

When LLM generates steps in this format, they will be parsed into:
- `thought`: "To answer this question, I need to find when Arthur's Magazine was founded."
- `action`: "Search"
- `action_input`: "Arthur's Magazine founding year"
- `observation`: None (filled after retrieval for RAG steps)
- `sub_answer`: None (filled after retrieval analysis)

## Impact on Saved Data

### Before Fix

```json
{
  "step_num": 1,
  "content": "Thought: Need to find founding year\nAction: Search[...]",
  "thought": null,
  "action": "Reason",
  "observation": null,
  "sub_answer": null
}
```

### After Fix

```json
{
  "step_num": 1,
  "content": "Thought: Need to find founding year\nAction: Search[...]",
  "thought": "Need to find founding year",
  "action": "Search",
  "action_input": "founding year",
  "observation": null,
  "sub_answer": null
}
```

## Training Benefits

1. **Consistent Format**: All steps (CoT and RAG) have same structure
2. **Better Supervision**: Can supervise thought generation separately
3. **Action Classification**: Can filter/weight by action type
4. **Format Validation**: Can detect when model deviates from ReAct format
5. **Easier Data Processing**: No special cases for CoT vs RAG

## Modified Files

- `/root/.local/PRMRAG/src/prmrag/generation/adaptive_generator.py`
  - Step 1 creation (lines ~298-319)
  - Step 2+ after failure (lines ~351-372)
  - Step 2+ normal case (lines ~392-413)
  - All now parse CoT content with `parse_rag_content()`

## Testing

To verify the fix works:

1. Run a test trajectory:
```bash
cd /root/.local/PRMRAG
python test_step1_fix.py
```

2. Check saved results:
```python
import json
with open('outputs/.../results_*.jsonl') as f:
    sample = json.loads(f.readline())
    step = sample['steps'][0]
    print(f"thought: {step['thought']}")
    print(f"action: {step['action']}")
    print(f"observation: {step['observation']}")
```

Expected: CoT steps now have `thought` and `action` parsed (not None).

## Backward Compatibility

Old data with `thought=None` will still load fine. New data will have parsed fields. When training, you can filter by checking:

```python
# Filter for data with parsed format
has_parsed_format = step['thought'] is not None
```
