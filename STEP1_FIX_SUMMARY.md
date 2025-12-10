# Step 1 RPE Calculation Fix

## Problem

Previously, Step 1 was always labeled as 'good' with a dummy RPE value of 1.0 because there was no baseline MC to compare against:

```python
# OLD CODE
if step_num == 1:
    step = AdaptiveStep(
        mc_before=0.0,      # ❌ No baseline
        mc_after=mc_cot,
        rpe=1.0,            # ❌ Dummy value
        label='good',       # ❌ Hardcoded
        metadata={'accepted': 'cot_baseline', 'step1': True}
    )
```

**Issue**: RPE = mc_after / mc_before, but mc_before=0.0 makes this calculation impossible. Step 1 could not be properly quality-assessed.

## Solution

Calculate MC on the question alone **BEFORE** generating Step 1. This provides a baseline to measure how much Step 1's reasoning improves the model's ability to answer correctly.

### Changes Made

**1. Pre-calculate mc_question (lines 258-264):**

```python
# NEW CODE - Before step loop
# Calculate MC on question alone BEFORE Step 1 (for proper RPE calculation)
print(f"\n[Pre-Step1] Calculating baseline MC on question alone...")
mc_question = self._monte_carlo_estimate(current_state, gold_answer)
print(f"[Pre-Step1] MC(question only) = {mc_question:.3f}")

# Use mc_question as baseline for Step 1
mc_prev = mc_question
```

**2. Proper RPE calculation for Step 1 (lines 290-314):**

```python
# NEW CODE - Step 1 handling
if step_num == 1:
    # Calculate RPE: how much did Step 1 reasoning improve MC?
    rpe_step1 = mc_cot / (mc_question + 0.01)  # Add epsilon to avoid division by zero
    label_step1 = 'good' if rpe_step1 >= self.threshold else 'bad'

    print(f"  Step 1: MC(question)={mc_question:.3f} → MC(step1)={mc_cot:.3f}, RPE={rpe_step1:.3f} → label={label_step1}")

    step = AdaptiveStep(
        step_id=t,
        step_type=StepType.COT,
        text=cot_step_text,
        content=cot_step_content,
        used_passages=[],
        mc_before=mc_question,  # ✅ Use question-only MC as baseline
        mc_after=mc_cot,
        rpe=rpe_step1,          # ✅ Real RPE calculation
        label=label_step1,      # ✅ Properly labeled based on RPE
        action="Reason",
        metadata={'accepted': 'step1_baseline', 'mc_question': mc_question},
    )
```

## Impact

### Before Fix
- Step 1: `mc_before=0.0`, `rpe=1.0` (dummy), `label='good'` (always)
- No quality assessment for Step 1
- Cannot filter low-quality Step 1 reasoning

### After Fix
- Step 1: `mc_before=mc_question`, `rpe=calculated`, `label=good/bad` (threshold-based)
- Proper quality assessment for Step 1
- Can filter Step 1 samples where reasoning doesn't improve MC

### Example Scenarios

**Scenario 1: Good Step 1**
```
MC(question) = 0.25
MC(step1) = 0.75
RPE = 0.75 / 0.25 = 3.0
Label = 'good' (RPE >= 0.79)
```

**Scenario 2: Bad Step 1**
```
MC(question) = 0.50
MC(step1) = 0.30
RPE = 0.30 / 0.50 = 0.6
Label = 'bad' (RPE < 0.79)
```

**Scenario 3: No improvement**
```
MC(question) = 0.60
MC(step1) = 0.60
RPE = 0.60 / 0.60 = 1.0
Label = 'bad' (RPE < 0.79 threshold, shows reasoning didn't help)
```

## Cost Analysis

### Additional Computation
- **Before**: No MC calculation before steps
- **After**: 1 additional MC estimation before Step 1
- **Cost per question**: K rollouts × avg_tokens_per_rollout
  - With K=8 (easy), ~4096 tokens
  - With K=32 (hard), ~16384 tokens

### Benefits
- **Data quality**: Can now filter bad Step 1 samples for training
- **Analysis**: Understand when Step 1 reasoning helps vs hurts
- **Consistency**: All steps now have proper RPE calculation
- **Better difficulty estimation**: Dynamic K now based on question difficulty, not Step 1 quality

## Verification

Run the test script to verify the fix:

```bash
cd /root/.local/PRMRAG
python test_step1_fix.py
```

Expected output:
```
✓ PASS: mc_before = 0.XXX (not 0.0)
✓ PASS: RPE = X.XXX (calculated from MC values)
✓ PASS: label = 'good'/'bad' (correctly determined by RPE >= 0.79)
✓ PASS: mc_question stored in metadata = 0.XXX
✓ PASS: mc_question == mc_before (0.XXX)
```

## Modified Files

- `/root/.local/PRMRAG/src/prmrag/generation/adaptive_generator.py`
  - Added mc_question pre-calculation (lines 258-264)
  - Updated Step 1 handling with proper RPE (lines 290-314)
  - **Updated dynamic K to use mc_question instead of mc_cot (lines 316-327)**

## Related Analysis

From 300-sample test before fix:
- All 300 Step 1 samples had `label='good'` (100%)
- All had `rpe=1.0` (dummy value)
- Could not assess Step 1 quality

Expected after fix:
- Step 1 labels will vary based on actual reasoning quality
- RPE values will reflect true MC improvement
- Can filter low-quality Step 1 for training data

## Additional Fix: Dynamic K Based on Question Difficulty

### Problem with Old Approach

Previously, dynamic K was determined by `mc_cot` (MC after Step 1):

```python
# OLD: Used mc_cot
if mc_cot < 0.1:
    self.current_k = self.k_hard  # K=32
```

**Issue**: Step 1 quality affects difficulty classification
- Bad Step 1 reasoning → low mc_cot → classified as "hard"
- Good Step 1 reasoning → high mc_cot → classified as "easy"
- **The question's true difficulty is mixed with Step 1 quality**

### New Approach

Now uses `mc_question` (MC before any reasoning):

```python
# NEW: Use mc_question
if mc_question < 0.1:
    self.current_k = self.k_hard  # K=32
elif mc_question < 0.9:
    self.current_k = self.k_medium  # K=16
else:
    self.current_k = self.k_easy  # K=8
```

**Benefits**:
- Difficulty based on question alone, not Step 1 quality
- Consistent K throughout trajectory (doesn't change if Step 1 is good/bad)
- More accurate resource allocation (hard questions get more rollouts)

### Example Impact

**Scenario**: Question where model has MC(question)=0.05

**Old behavior**:
- If Step 1 is good: mc_cot=0.60 → K=16 (medium) ❌ Wrong
- If Step 1 is bad: mc_cot=0.03 → K=32 (hard) ✓ Correct by luck

**New behavior**:
- Always: mc_question=0.05 → K=32 (hard) ✓ Always correct
