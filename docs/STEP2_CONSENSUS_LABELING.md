# Step 2: Consensus Labeling 가이드

## 개요

기존 labeling 모듈을 활용한 통합 Consensus Labeling 파이프라인:
- **RPELabeler**: MC 기반 정량 평가
- **JudgeLabeler**: LLM 기반 정성 평가 (VersaPRM 스타일)
- **ConsensusModule**: 두 평가자의 합의로 최종 라벨 결정

## 아키텍처

```
Trajectories (JSONL)
        |
        ├─> RPELabeler (MC rollouts)
        |       └─> RPE labels (GOOD/BAD based on threshold)
        |
        ├─> JudgeLabeler (LLM evaluation)
        |       └─> Judge labels (GOOD/BAD with reasoning)
        |
        └─> ConsensusModule
                ├─> Both GOOD → label=1 (positive)
                ├─> Both BAD  → label=0 (negative)
                └─> Disagree  → filtered
                        |
                        v
                High-quality labeled data
```

## 사용 방법

### 기본 실행

```bash
python3 scripts/run_consensus_labeling.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    --output outputs/training_data/consensus_labeled.jsonl
```

### 전체 옵션

```bash
python3 scripts/run_consensus_labeling.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    --output outputs/training_data/consensus_labeled.jsonl \
    \
    --rpe-model Qwen/Qwen2.5-7B-Instruct \
    --rpe-rollouts 5 \
    --rpe-threshold 0.5 \
    \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --judge-style versaprm \
    \
    --strategy strict \
    --min-agreement 0.8 \
    --trajectory-level \
    \
    --gpu-memory 0.8 \
    --tensor-parallel 1 \
    \
    --limit 100  # 테스트용
```

## 주요 파라미터

### RPE 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--rpe-model` | Qwen2.5-7B | RPE용 모델 (작은 모델 가능) |
| `--rpe-rollouts` | 5 | MC rollout 횟수 |
| `--rpe-threshold` | 0.5 | RPE >= 0.5 → GOOD, < 0.5 → BAD |

### Judge 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--judge-model` | Qwen2.5-72B | Judge용 모델 (큰 모델 권장) |
| `--judge-style` | versaprm | 프롬프트 스타일 (versaprm/default) |

### Consensus 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--strategy` | strict | 합의 전략 (strict/lenient/balanced) |
| `--min-agreement` | 0.8 | 최소 agreement rate (< 이면 필터) |
| `--trajectory-level` | False | 단계 충돌 시 전체 궤적 필터 |

## VersaPRM Judge 프롬프트

JudgeLabeler에서 사용하는 프롬프트 (`judge_labeler.py` 152-235줄):

```python
def _build_versaprm_prompt(trajectory, step_idx):
    """
    You are an expert evaluator for question-answering systems.

    # Task
    Evaluate whether a reasoning step is GOOD or BAD for answering the question correctly.

    # Evaluation Criteria
    A step is GOOD if:
    - It retrieves relevant information that helps answer the question
    - It makes correct logical inferences
    - It moves toward the correct answer
    - The retrieved passages are relevant and useful

    A step is BAD if:
    - It retrieves irrelevant or misleading information
    - It makes incorrect logical inferences
    - It leads away from the correct answer
    - The passages are off-topic or unhelpful

    # Question
    {question}

    # Correct Answer
    {gold_answer}

    # Supporting Facts (for reference)
    1. {fact_1}
    2. {fact_2}
    ...

    # Previous Steps
    Step 1: {action_1}
    Result: {observation_1}
    ...

    # Step to Evaluate
    Action: {action}
    Retrieved Passages:
    1. {passage_1}
    2. {passage_2}
    ...
    Observation: {observation}

    # Your Evaluation
    Provide your evaluation in the following format:

    Reasoning: [Explain why this step is good or bad]
    Label: [GOOD or BAD]
    Confidence: [0.0 to 1.0]
    """
```

## 출력 형식

### Labeled Trajectory

```json
{
  "question_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first...",
  "gold_answer": "Arthur's Magazine",
  "is_filtered": false,
  "filter_reason": null,
  "steps": [
    {
      "step_num": 1,
      "action": "Search",
      "observation": "According to [1]...",
      "label": 1,
      "rpe_label": "GOOD",
      "judge_label": "GOOD",
      "is_consensus": true,
      "confidence": 0.85
    },
    {
      "step_num": 2,
      "action": "Search",
      "observation": "According to [2]...",
      "label": 0,
      "rpe_label": "BAD",
      "judge_label": "BAD",
      "is_consensus": true,
      "confidence": 0.72
    }
  ]
}
```

### Filtered Trajectory

```json
{
  "question_id": "...",
  "question": "...",
  "is_filtered": true,
  "filter_reason": "trajectory_level_conflict",
  "steps": [
    {
      "step_num": 1,
      "label": null,
      "rpe_label": "GOOD",
      "judge_label": "BAD",
      "is_consensus": false,
      "filtered_reason": "rpe_GOOD_judge_BAD"
    }
  ]
}
```

## 출력 통계 예시

```
============================================================
CONSENSUS FILTERING REPORT
============================================================

Total steps evaluated: 4800

Consensus Results:
  ✓ Positive consensus (label=1): 3456
  ✓ Negative consensus (label=0): 672
  ✗ Disagreements (filtered):     672

Agreement rate: 86.0%
Filtered rate:  14.0%

Final label distribution:
  positive (1): 3456 (72.0%)
  negative (0): 672 (14.0%)
  filtered (None): 672 (14.0%)
============================================================

============================================================
SUMMARY
============================================================
Total trajectories:     1000
Kept (high quality):    860 (86.0%)
Filtered (low quality): 140 (14.0%)

Agreement rate: 86.0%
Output: outputs/training_data/consensus_labeled.jsonl
============================================================
```

## 모듈 구조

### 1. RPELabeler (`src/prmrag/labeling/rpe_labeler.py`)

MC 기반 RPE 계산:
- `MC(s_t)`: 현재 prefix에서의 성공률
- `MC(s_t, a_t)`: 이 step 포함 후 성공률
- `RPE = MC(s_t, a_t) / MC(s_t)`
- Threshold 기반 이진 라벨 (GOOD/BAD)

### 2. JudgeLabeler (`src/prmrag/labeling/judge_labeler.py`)

LLM 기반 평가 (VersaPRM 스타일):
- Question + Gold Answer + Supporting Facts 제공
- Previous steps context
- Retrieved passages 평가
- GOOD/BAD + Reasoning + Confidence 출력

### 3. ConsensusModule (`src/prmrag/labeling/consensus.py`)

합의 기반 필터링:
```python
# Positive consensus
if rpe == GOOD and judge == GOOD:
    label = 1  # Keep

# Negative consensus
if rpe == BAD and judge == BAD:
    label = 0  # Keep

# Disagreement
else:
    label = None  # Filter
```

## 전략 비교

### Strict (기본)
- RPE=GOOD, Judge=GOOD → 1
- RPE=BAD, Judge=BAD → 0
- 나머지 → 필터
- 높은 품질, 낮은 데이터량

### Lenient (현재는 동일)
- Binary label로 통일되어 strict와 동일

### Balanced (현재는 동일)
- Binary label로 통일되어 strict와 동일

## 예상 결과 (1000 궤적 기준)

| 메트릭 | 값 |
|--------|-----|
| 총 단계 | ~4800 |
| Positive consensus (1) | ~3456 (72%) |
| Negative consensus (0) | ~672 (14%) |
| Disagreement (filtered) | ~672 (14%) |
| **Agreement Rate** | **~86%** |
| **유지된 궤적** | **~860 (86%)** |

## 워크플로우

### 1단계: 1000개 생성 완료 대기
```bash
# 현재 실행 중
python3 scripts/batch_test_hybrid.py \
    --num-questions 411 \
    --start-idx 589 \
    --num-rollouts 4 \
    --output-dir outputs/hybrid_1000q_from_0
```

### 2단계: Consensus Labeling 실행
```bash
python3 scripts/run_consensus_labeling.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    --output outputs/training_data/consensus_labeled_1000q.jsonl \
    --rpe-model Qwen/Qwen2.5-7B-Instruct \
    --judge-model Qwen/Qwen2.5-72B-Instruct \
    --strategy strict \
    --min-agreement 0.8
```

### 3단계: 학습 데이터 품질 검증
```bash
# 통계 확인
python3 scripts/analyze_labels.py \
    outputs/training_data/consensus_labeled_1000q.jsonl

# 샘플 확인
head -3 outputs/training_data/consensus_labeled_1000q.jsonl | jq .
```

## 문제 해결

### Agreement Rate가 낮음 (<70%)
- RPE threshold 조정 (`--rpe-threshold 0.4` 또는 `0.6`)
- Judge 모델 크기 증가
- Rollout 횟수 증가 (`--rpe-rollouts 10`)

### 너무 많은 데이터 필터됨 (>30%)
- `--min-agreement` 낮추기 (0.7)
- `--trajectory-level` 제거 (step-level filtering)
- Strategy 변경 (현재는 모두 동일)

### OOM (Out of Memory)
- `--gpu-memory` 낮추기 (0.6)
- Batch 크기 줄이기 (`--limit` 사용)
- `--tensor-parallel` 증가 (multi-GPU)

## 참고 문서

- `judge_labeler.py` - VersaPRM 프롬프트 구현
- `rpe_labeler.py` - MC 기반 RPE 계산
- `consensus.py` - Consensus 로직
- `CONSENSUS_FILTERING_WORKFLOW.md` - 전체 워크플로우

## 다음 단계

1. 1000개 생성 완료
2. Consensus labeling 실행
3. 품질 검증 및 통계 분석
4. PRM 모델 학습 시작
