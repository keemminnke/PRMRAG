# Consensus Filtering 워크플로우 (Step 2)

## 개요

RPE Evaluator와 Judge Evaluator의 두 평가자가 동의하는 단계만 학습 데이터로 사용

## 전체 파이프라인

```
1000 Trajectories (Full Data)
         |
         ├──> Extract RPE Labels ────> rpe_labels.jsonl
         |
         └──> Prepare Judge Data ───> judge_input.jsonl
                      |
                      └──> GPT-OSS-120B Judge ───> judge_labels.jsonl
                                                           |
                                                           v
              rpe_labels.jsonl + judge_labels.jsonl ──> Consensus Filter
                                                           |
                                                           v
                                              filtered_training_data.jsonl
```

## Step 1: RPE Label 추출

### 스크립트
`scripts/extract_rpe_labels.py`

### 실행
```bash
python3 scripts/extract_rpe_labels.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    outputs/consensus/rpe_labels.jsonl
```

### 출력 형식
```json
{
  "question_id": "5a7a06935542990198eaf050",
  "steps": [
    {"step_num": 1, "rpe_label": "good"},
    {"step_num": 2, "rpe_label": "good"},
    {"step_num": 3, "rpe_label": "bad"}
  ]
}
```

### 통계 출력
```
✓ Extracted 1000 trajectories
✓ Total 4800 steps

RPE Label Distribution:
  Good: 3840 (80.0%)
  Bad:  960 (20.0%)
```

## Step 2: Judge 데이터 준비

### 스크립트
`scripts/prepare_judge_data.py`

### 실행
```bash
python3 scripts/prepare_judge_data.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    outputs/consensus/judge_input.jsonl
```

### 출력 형식 (RPE label 없음, 순수 추론만)
```json
{
  "question_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first...",
  "gold_answer": "Arthur's Magazine",
  "predicted_answer": "Arthur's Magazine",
  "is_correct": true,
  "num_steps": 5,
  "steps": [
    {
      "step_num": 1,
      "thought": "To determine which magazine was started first...",
      "action": "Search",
      "action_input": "founding year of Arthur's Magazine",
      "observation": "According to [1], Arthur's Magazine was founded in 1844.",
      "sub_answer": "Arthur's Magazine was founded in 1844."
    }
  ]
}
```

### 특징
- ✅ 순수 추론 필드만 (thought, action, action_input, observation, sub_answer)
- ✅ RPE label 완전 제외 (독립 평가 보장)
- ✅ content, metadata 제거 (99.1% 크기 감소)

## Step 3: GPT-OSS-120B Judge 실행

### 스크립트 (작성 예정)
`scripts/run_judge_labeling.py`

### 실행 (예정)
```bash
python3 scripts/run_judge_labeling.py \
    outputs/consensus/judge_input.jsonl \
    outputs/consensus/judge_labels.jsonl \
    --model gpt-oss-120b \
    --batch-size 10
```

### Judge 프롬프트
```python
JUDGE_PROMPT = """You are evaluating reasoning steps for quality.

Question: {question}
Gold Answer: {gold_answer}

Step {step_num}:
Thought: {thought}
Action: {action}
Action Input: {action_input}
Observation: {observation}
Sub-answer: {sub_answer}

Evaluate this step:
1. Is the reasoning logical and coherent?
2. Is the action appropriate for the situation?
3. Does the observation analysis make sense?
4. Does this step make progress toward the answer?

Respond with a single word: good or bad
"""
```

### 출력 형식
```json
{
  "question_id": "5a7a06935542990198eaf050",
  "steps": [
    {"step_num": 1, "judge_label": "good"},
    {"step_num": 2, "judge_label": "good"},
    {"step_num": 3, "judge_label": "bad"}
  ]
}
```

## Step 4: Consensus Filtering

### 스크립트
`scripts/consensus_filter.py`

### 실행
```bash
python3 scripts/consensus_filter.py \
    --full-data outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    --rpe-labels outputs/consensus/rpe_labels.jsonl \
    --judge-labels outputs/consensus/judge_labels.jsonl \
    --output outputs/training_data/consensus_filtered.jsonl \
    --keep-mode agree
```

### 옵션
- `--keep-mode agree`: 둘 다 동의하는 단계만 유지 (기본)
- `--keep-mode all`: 모든 단계 유지하고 consensus 정보만 추가

### Consensus 규칙

```python
if rpe_label == "good" and judge_label == "good":
    ✅ Keep (Both Agree - Good step)

elif rpe_label == "bad" and judge_label == "bad":
    ✅ Keep (Both Agree - Bad step, useful for learning)

else:
    ❌ Filter (Disagreement - uncertain quality)
```

### 출력 통계 예시
```
============================================================
CONSENSUS FILTERING
============================================================

Loading labels...
✓ RPE labels: 1000 trajectories
✓ Judge labels: 1000 trajectories

Comparing RPE vs Judge...
Total steps compared: 4800

Consensus Statistics:
  Both Good:   3456 (72.0%)
  Both Bad:     672 (14.0%)
  Disagree:     672 (14.0%)

Agreement Rate: 86.0%

Filtering trajectories (mode: agree)...

Filtering Results:
  Kept:     980 trajectories, 4128 steps
  Filtered: 20 trajectories, 672 steps

Kept 86.0% of steps

✓ Output: outputs/training_data/consensus_filtered.jsonl

============================================================
```

### 출력 형식
```json
{
  "question_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first...",
  "gold_answer": "Arthur's Magazine",
  "predicted_answer": "Arthur's Magazine",
  "is_correct": true,
  "num_steps": 4,
  "steps": [
    {
      "step_num": 1,
      "content": "Thought: To determine...\nAction: Search[...]",
      "thought": "To determine which magazine...",
      "action": "Search",
      "action_input": "founding year of Arthur's Magazine",
      "observation": "According to [1]...",
      "sub_answer": "Arthur's Magazine was founded in 1844.",
      "label": "good",
      "consensus": {
        "rpe_label": "good",
        "judge_label": "good",
        "agree": true
      }
    }
  ]
}
```

## 필터링 통계 분석

### 예상 결과 (1000개 궤적 기준)

| 메트릭 | 값 |
|--------|-----|
| 총 궤적 | 1000 |
| 총 단계 | ~4800 |
| RPE Good | ~3840 (80%) |
| RPE Bad | ~960 (20%) |
| **Agreement Rate** | **85-90%** |
| **Kept Steps** | **~4100 (86%)** |
| **Filtered Steps** | **~700 (14%)** |

### Agreement Rate 해석

- **High Agreement (>85%)**: RPE와 Judge가 대부분 동의 → 높은 품질 데이터
- **Medium Agreement (70-85%)**: 일부 불일치 → 추가 검토 필요
- **Low Agreement (<70%)**: 많은 불일치 → RPE 또는 Judge 재평가 필요

## 전체 워크플로우 실행

### 1단계: 1000개 질문 생성 (진행 중)
```bash
# 현재 실행 중
python3 scripts/batch_test_hybrid.py \
    --num-questions 411 \
    --start-idx 589 \
    --num-rollouts 4 \
    --output-dir outputs/hybrid_1000q_from_0
```

### 2단계: RPE Labels 추출
```bash
python3 scripts/extract_rpe_labels.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    outputs/consensus/rpe_labels.jsonl
```

### 3단계: Judge 데이터 준비
```bash
python3 scripts/prepare_judge_data.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    outputs/consensus/judge_input.jsonl
```

### 4단계: GPT-OSS-120B Judge 실행 (예정)
```bash
python3 scripts/run_judge_labeling.py \
    outputs/consensus/judge_input.jsonl \
    outputs/consensus/judge_labels.jsonl \
    --model gpt-oss-120b
```

### 5단계: Consensus Filtering
```bash
python3 scripts/consensus_filter.py \
    --full-data outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    --rpe-labels outputs/consensus/rpe_labels.jsonl \
    --judge-labels outputs/consensus/judge_labels.jsonl \
    --output outputs/training_data/consensus_filtered.jsonl
```

## 파일 구조

```
outputs/
├── hybrid_1000q_from_0/
│   └── results_hybrid_20251209_072136.jsonl  (Full data)
│
├── consensus/
│   ├── rpe_labels.jsonl          (RPE labels only)
│   ├── judge_input.jsonl         (Clean data for Judge)
│   └── judge_labels.jsonl        (Judge labels from GPT-OSS-120B)
│
└── training_data/
    └── consensus_filtered.jsonl  (Final training data)
```

## 다음 작업

### 즉시 가능
- ✅ RPE label 추출 스크립트 완성
- ✅ Judge 데이터 준비 스크립트 완성
- ✅ Consensus filtering 스크립트 완성

### 1000개 완료 후
1. RPE labels 추출
2. Judge 데이터 준비
3. GPT-OSS-120B Judge 스크립트 작성 및 실행
4. Consensus filtering 실행
5. 학습 데이터 품질 검증

## 품질 메트릭

### Agreement Rate
- 목표: >85% (RPE와 Judge의 일치도)
- 측정: `consensus_filter.py`에서 자동 계산

### Data Retention
- 목표: >80% 단계 유지
- 측정: Kept steps / Total steps

### Label Distribution (Filtered Data)
- Good steps: ~85%
- Bad steps: ~15%
- Balanced for PRM training

## 참고 문서

- `JUDGE_DATA_PREPARATION.md` - Judge 데이터 준비 상세
- `COT_FIELD_MAPPING.md` - CoT 필드 구조
- `REPARSE_REPORT.md` - 데이터 재파싱 보고서
