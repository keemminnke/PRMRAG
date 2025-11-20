# Bug Fix: MC Estimation Was Always 0 and RAG Didn't Help

## Problem Summary

When running adaptive MC-CoT + RAG, we observed:
1. **MC estimation was 0 for many questions** - even when the model seemed to have the right reasoning
2. **RAG didn't improve MC** - even when it retrieved correct information

## Root Cause

The critical bug was in the `_rollout()` method (line 870 in `adaptive_generator.py`):

**Rollouts didn't have access to retrieved passages!**

### How MC Estimation Works
1. After each step, the system does `num_rollouts` completions from current state
2. It counts how many rollouts produce the correct answer
3. MC = successes / num_rollouts

### The Bug
When computing MC after a RAG step:
- The state included RAG step text in `reasoning_history`
- BUT the state also had a `passages` field with the actual retrieved documents
- The `_rollout()` method only used `reasoning_history`, **not `passages`**
- So rollouts were answering blindly without the retrieved information!

### Example (Question 8: Lewiston Maineiacs)
**Before fix:**
- RAG correctly retrieves: "seating capacity of 3,677"
- But rollouts don't see passages
- Rollouts answer: "3" (wrong)
- MC = 0.0 ❌

**After fix:**
- RAG retrieves: "seating capacity of 3,677"
- Rollouts now see passages in prompt
- Rollouts answer: "3,677 seated" (correct)
- MC > 0 ✅

## The Fix

Modified `_rollout()` to include retrieved passages in the rollout prompt:

```python
# CRITICAL FIX: Include retrieved passages if available
if state.get('passages'):
    lines.append("Retrieved Information:")
    # ... format passages ...
    for i, passage in enumerate(unique_passages, 1):
        title = passage.get('title', 'Document')
        content = passage.get('text', '')
        lines.append(f"[{i}] {title}: {content}")
```

Now rollouts can actually use the information that RAG retrieved!

## Results

**Before fix (from original run):**
- Total RAG interventions: 20
- Improved MC: 3/20 (15%)
- Degraded MC: 2/20 (10%)
- Unchanged MC: 15/20 (75%)
- Average MC improvement: +0.087

**After fix (Question 1 example):**
- Step 2 RAG: MC = 0.875 (was ~0 before)
- Step 3 RAG: MC = 1.000 (perfect!)
- Question answered correctly ✅

## Why This Matters

This fix is critical because:

1. **MC estimation now reflects actual model capability** when it has retrieved information
2. **RAG effectiveness can be properly measured** - we can see MC improve when good passages are retrieved
3. **Training labels are more accurate** - steps get correct good/bad labels based on actual performance
4. **The model can learn to use RAG effectively** - it sees when RAG helps vs doesn't help

## Files Changed

- `src/prmrag/generation/adaptive_generator.py`:
  - Modified `_rollout()` method (line 870-939)
  - Modified `_monte_carlo_estimate()` method (line 843-875) to add debug logging

## Testing

Run the test again to verify improvement:
```bash
python3 scripts/batch_test_adaptive.py --num-questions 10 --num-rollouts 8
```

You should now see:
- MC values > 0 after RAG steps retrieve good passages
- Debug output showing: `[MC DEBUG] Rollouts will use N retrieved passages`
- Higher RAG effectiveness scores
