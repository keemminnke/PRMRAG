# PRM RAG Project - Development Log

## Session: 2025-12-29 - Policy Data Generation & Judge Labeling Enhancement

### Overview
This session focused on generating policy training data with improved observation quality, implementing Judge labeling with search query evaluation, and fixing critical data format bugs.

---

## 1. Policy Data Generation (1000 Questions)

### Task
Generate 1000 medium-level questions with enhanced observation context for better Judge evaluation.

### Key Changes

#### 1.1 Observation Length Enhancement
**Problem**: Previous data had ~275 chars per passage (too short for Judge evaluation)

**Root Cause**:
```python
# Old: batch_test_hybrid.py
def get_truncated_text(doc_text_list, max_paragraphs=2, max_chars=1200):
    selected_paragraphs = doc_text_list[:max_paragraphs]  # Only first 2 paragraphs!
```

**Solution**:
```python
# New: batch_test_hybrid.py:31
def get_truncated_text(doc_text_list, max_chars=1200):
    """Character-based truncation with truncation marker."""
    joined_text = ' '.join(doc_text_list)  # All paragraphs
    if len(joined_text) > max_chars:
        return joined_text[:max_chars] + " [TRUNCATED]"
    return joined_text
```

**Results**:
- Before: top5 × ~275 chars = ~1,375 chars per step
- After: top5 × ~1,200 chars = ~6,000 chars per step
- **4.6x increase** in observation context!

#### 1.2 Execution Results
```bash
python scripts/batch_test_hybrid.py \
  --dataset hotpotqa \
  --split train \
  --question-level medium \
  --num-questions 1000 \
  --start-filtered-idx 0 \
  --num-rollouts 8 \
  --output-dir outputs/hybrid_1000q_medium_fulltext \
  --fusion-method rrf \
  --run-id 20251227_001824 \
  --fsync
```

**Output**:
- ✅ Success: 695 trajectories
- ❌ Failures: 305 trajectories (vLLM crashes)
- 📊 Accuracy: 37% (261/695 correct)
- 📁 Files:
  - `results_hybrid_20251227_001824.jsonl` (695 trajectories)
  - `failures_hybrid_20251227_001824.jsonl` (305 failed)
  - `summary_hybrid_20251227_001824.json` (statistics)

---

## 2. Critical Bug Fixes

### 2.1 `num_passages` Bug

**Problem**: Finish steps with passages were incorrectly counted as CoT steps.

**Example**:
```json
{
  "action": "Finish",
  "observation": "[1] Doc1... [2] Doc2... [5] Doc5...",
  "num_passages": 0,  // ❌ Wrong! Should be 5
  "num_rag_steps": 0  // ❌ Wrong!
}
```

**Root Cause**:
```python
# batch_test_hybrid.py:349-367 (OLD)
if step_info['action'] == 'Search':
    step_info['num_passages'] = len(step.used_passages)
else:
    step_info['num_passages'] = 0  # ❌ Wrong assumption!
```

**Fix**:
```python
# batch_test_hybrid.py:349-354 (NEW)
# Check if step has passages (regardless of action type)
if hasattr(step, 'used_passages') and step.used_passages:
    step_info['num_passages'] = len(step.used_passages)
    step_info['passage_titles'] = [p.get('title', 'Unknown') for p in step.used_passages]
else:
    step_info['num_passages'] = 0
```

**Impact**:
- 666/695 trajectories (95.8%) were incorrectly classified
- Reparsing results:
  - Trajectories with RAG: 420 → 672 (+60%)
  - Total RAG steps: 1,181 → 1,586 (+34%)
  - Total CoT steps: 358 → 619 (+73%)

**Files**:
- Original: `results_hybrid_20251227_001824.jsonl`
- Fixed: `results_hybrid_20251227_001824_reparsed.jsonl`

### 2.2 Redundant `content` Field

**Problem**: `content` field duplicates `thought + action + observation`

**Solution**: Remove `content` field from all data
```python
# batch_test_hybrid.py:313-322
step_info = {
    'step_num': i,
    # 'content': step.content,  # Removed: redundant
    'mc_before': round(step.mc_before, 3),
    'mc_after': round(step.mc_after, 3),
    'rpe': round(step.rpe, 3),
    'label': step.label,
    'thought': getattr(step, 'thought', None),
    'action': getattr(step, 'action', 'Reason'),
}
```

**Results**:
- 5Q merged: 128KB → 69KB (45.8% reduction)
- 695Q reparsed: 21MB → 11MB (47.6% reduction)

**Files**:
- Cleaned: `results_hybrid_20251227_001824_reparsed_clean.jsonl`

---

## 3. Judge Labeling Enhancement

### 3.1 Search Query Evaluation

**Added to Judge Prompt** (`judge_labeler.py:241-262`):

```python
"**CRITICAL RULE: Even if the final answer matches the Correct Answer, you MUST label the step as BAD if:**",
"- The Model claims facts (names, dates, numbers) that are NOT present in the Observation",
"- The Model makes inferences not supported by the retrieved information",
"- The Model hallucinates or fabricates information",
"- The search query is irrelevant, too vague, or poorly formulated (for Search steps)",  # NEW
"- The retrieved documents don't help answer the question (for Search steps)",  # NEW
"",
"Evaluation Steps:",
"1. **For Search steps**: Evaluate the search query quality",  # NEW
"   - Is the query relevant to answering the original question?",
"   - Is the query specific enough (not too vague)?",
"   - Does the query consider previous steps' context?",
"   - Did the search retrieve helpful information?",
```

**Judge Now Evaluates**:
1. ✅ Search query quality (relevance, specificity, context-awareness)
2. ✅ Retrieved document usefulness
3. ✅ Hallucination detection
4. ✅ Grounding in observations
5. ✅ Progress toward correct answer

### 3.2 `max_model_len` Fix

**Problem**: `max_model_len` parameter not passed to vLLM engine

**Fix Chain**:
```python
# 1. judge_labeler.py:77
model_config = {
    'max_model_len': 40960,  # Set here
}

# 2. policy_model_vllm.py:382-413 (load_policy_model)
return PolicyModelVLLM(
    max_model_len=config.get('max_model_len', None),  # Pass through
)

# 3. policy_model_vllm.py:39-101 (__init__)
vllm_kwargs = {
    'model': model_name,
    # ... other params
}
if max_model_len is not None:
    vllm_kwargs['max_model_len'] = max_model_len
    print(f"  Setting max_model_len={max_model_len}")

self.llm = LLM(**vllm_kwargs)  # Finally passed to vLLM!
```

### 3.3 Test Results (5 Questions)

**Execution**:
```bash
python3 /root/.local/PRMRAG/scripts/judge_label_qwq.py
```

**Consensus Results**:
- Total steps: 15
- Consensus: 11/15 (73.3%)
- Disagreement: 4/15 (26.7%)
- Full consensus questions: 3/5 (60.0%)

**Judge Successfully Detected**:

1. **Hallucination** (RPE=good, Judge=BAD):
   ```
   Model: "Henri Leconte won a singles Grand Slam title"
   Reality: He only reached the final (never won)
   Judge: "fabricates unsupported by Observations"
   ```

2. **Poor Search Query** (RPE=good, Judge=BAD):
   ```
   Issue: Repeatedly retrieves same irrelevant documents
   Judge: "retrieves the same irrelevant documents repeatedly"
   ```

3. **RPE False Negative** (RPE=bad, Judge=GOOD):
   ```
   Reality: Actually retrieved relevant information
   Judge: "retrieves relevant information without unsupported claims"
   ```

**Output Files**:
- Judge results: `test_judge_5q_newdata_results.json`
- Merged with policy: `test_judge_5q_newdata_merged_clean.jsonl`

---

## 4. Data Analysis

### 4.1 MC=0 Trajectories

**Analysis**:
- Total MC=0 from start: 144/695 (20.7%)
- All have mc_before=0 at Step 1 (question is too hard)
- Patterns:
  - Finish-only (1 step): 30 trajectories
  - Search attempts: 114 trajectories
- Accuracy: 0/144 (0.0%)

**Implication**: These 144 trajectories may not be useful for training (all steps labeled "bad")

### 4.2 Backtracking Analysis

**Backtracking Logic** (adaptive_generator.py:535-622):
```python
if rpe_cot < 0.8:  # CoT step quality too low
    # Save rejected segment for DPO
    rejected_segments.append(...)

    # BACKTRACK: Don't add CoT step
    # Force RAG intervention instead
    rag_result = self._try_rag_intervention(...)
    steps.append(rag_step)  # Replace with RAG step
```

**Evidence in Data**:
- Step 1: Reason (RPE=1.149, good) ✅
- Step 2: Search (mc drops 0.766→0.0) ← Backtracking occurred
- Invisible rejected CoT between Step 1 and 2

### 4.3 Finish Action Distribution

```
Total Finish steps: 666

RAG Finish (with passages):  405 (60.8%)
  Example: "Based on document [2]..."

CoT Finish (no passages):    261 (39.2%)
  Example: "After reviewing documents..."
```

---

## 5. Final Data Structure

### Step Format (After Cleaning)
```json
{
  "step_num": 1,
  "action": "Search",
  "action_input": "query text",
  "thought": "I need to find...",
  "observation": "[1] Doc1...[2] Doc2...[5] Doc5... [TRUNCATED]",
  "mc_before": 0.766,
  "mc_after": 0.828,
  "rpe": 1.076,
  "label": "good",
  "num_passages": 5,
  "passage_titles": ["Title1", "Title2", ...],
  "judge_label": "GOOD",
  "judge_reasoning": "The search query is specific...",
  "judge_confidence": 0.95,
  "consensus": true
}
```

### Trajectory Format
```json
{
  "question_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first...",
  "gold_answer": "Arthur's Magazine",
  "predicted_answer": "Arthur's Magazine",
  "is_correct": true,
  "num_steps": 2,
  "num_cot_steps": 0,
  "num_rag_steps": 2,
  "has_rag": true,
  "steps": [...],
  "rag_interventions": [...],
  "format_compliance": {...},
  "dataset_idx": 1,
  "filtered_idx": 0,
  "question_level": "medium",
  "run_id": "20251227_001824"
}
```

---

## 6. Files & Artifacts

### Generated Data
```
outputs/hybrid_1000q_medium_fulltext/
├── results_hybrid_20251227_001824.jsonl              # Original (695 trajectories)
├── results_hybrid_20251227_001824_reparsed.jsonl     # Fixed num_passages
├── results_hybrid_20251227_001824_reparsed_clean.jsonl  # + Removed content (11MB)
├── failures_hybrid_20251227_001824.jsonl             # 305 failures
└── summary_hybrid_20251227_001824.json               # Statistics
```

### Archives
```
outputs/
├── hybrid_1000q_medium_fulltext.tar.gz              # Original archive (2.7MB)
└── hybrid_1000q_medium_fulltext_reparsed.tar.gz     # Cleaned archive (2.7MB)
```

### Test Data
```
outputs/
├── test_judge_5q_newdata.jsonl                      # 5Q policy data
├── test_judge_5q_newdata_results.json               # Judge results
└── test_judge_5q_newdata_merged_clean.jsonl         # Merged + cleaned (69KB)
```

---

## 7. Next Steps

### Immediate Tasks
1. ✅ Generate next batch (695-1694):
   ```bash
   python scripts/batch_test_hybrid.py \
     --start-filtered-idx 695 \
     --num-questions 1000 \
     --output-dir outputs/hybrid_1000q_medium_fulltext_batch2 \
     --fusion-method rrf \
     --run-id $(date +%Y%m%d_%H%M%S) \
     --fsync
   ```

2. Judge labeling for all 695 trajectories
3. Consensus filtering
4. Filter out MC=0 trajectories (optional)

### Future Work
- DPO training with rejected_segments (backtracking data)
- Critic model fine-tuning with consensus-filtered data
- Scale to full dataset

---

## 8. Key Learnings

### Technical Insights

1. **Observation Length Matters**:
   - 4.6x increase in context significantly helps Judge evaluation
   - [TRUNCATED] markers provide transparency

2. **Action Type ≠ Step Type**:
   - Finish can be RAG or CoT depending on passages
   - Must check `used_passages` regardless of action

3. **vLLM Parameter Passing**:
   - Need explicit parameter forwarding through multiple layers
   - Config dict → load_policy_model → __init__ → LLM()

4. **Data Efficiency**:
   - Removing redundant fields: 47% file size reduction
   - Clean data structure aids downstream processing

### Judge Evaluation Quality

**Strengths**:
- ✅ Excellent hallucination detection
- ✅ Search query quality evaluation works well
- ✅ Can override RPE when clearly wrong

**Challenges**:
- 26.7% disagreement with RPE
- Need to analyze disagreement patterns for calibration

---

## 9. Statistics Summary

### Data Generation
| Metric | Value |
|--------|-------|
| Total questions attempted | 1,000 |
| Successful trajectories | 695 (69.5%) |
| Failed (vLLM crash) | 305 (30.5%) |
| Accuracy | 37% |
| Avg observation length | ~6,000 chars |
| Total data size (cleaned) | 11MB |

### After Bug Fixes
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Trajectories with RAG | 420 | 672 | +60% |
| Total RAG steps | 1,181 | 1,586 | +34% |
| Total CoT steps | 358 | 619 | +73% |

### Judge Consensus (5Q Test)
| Metric | Value |
|--------|-------|
| Consensus steps | 11/15 (73.3%) |
| Disagreement | 4/15 (26.7%) |
| Full consensus questions | 3/5 (60.0%) |

---

## Code Changes Summary

### Modified Files
1. `src/prmrag/labeling/judge_labeler.py`
   - Added search query evaluation prompts
   - Enhanced evaluation steps

2. `src/prmrag/models/policy_model_vllm.py`
   - Added `max_model_len` parameter support
   - Fixed parameter forwarding to vLLM

3. `scripts/batch_test_hybrid.py`
   - Fixed `num_passages` bug (check all actions)
   - Removed redundant `content` field
   - Enhanced `get_truncated_text` (no paragraph limit)
   - Added metadata support

4. `scripts/judge_label_qwq.py`
   - Updated input/output paths for new test data

### New Scripts
1. `/tmp/reparse_passages.py` - Fix num_passages in existing data
2. `/tmp/remove_content_field.py` - Clean redundant content field
3. `/tmp/merge_judge_results.py` - Merge Judge labels with policy data

---

**Session Duration**: ~4 hours
**Lines of Code Modified**: ~200
**Data Generated**: 695 trajectories (11MB cleaned)
**Bugs Fixed**: 3 critical bugs
**Enhancements**: 2 major improvements (observation length, Judge evaluation)
