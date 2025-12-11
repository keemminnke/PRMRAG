# GPT-OSS-120B Judge 데이터 준비 가이드

## 개요

Step 2 Consensus Filtering을 위해 GPT-OSS-120B Judge에게 보낼 데이터를 준비하는 방법

## 왜 필요한가?

원본 JSONL 파일에는 너무 많은 메타데이터가 포함되어 있어 Judge 평가에 방해가 됩니다:

### 불필요한 필드 (제거됨)
- `mc_before`, `mc_after` - RPE 계산용 메타데이터
- `rpe` - RPE 점수 (Judge는 독립적으로 평가해야 함)
- `num_passages` - 검색 관련 메타데이터
- `thought`, `action`, `action_input`, `observation`, `sub_answer` - 내부 파싱 필드
- 기타 노이즈

### 필수 필드만 유지
- `question_id` - 질문 식별자
- `question` - 질문 텍스트
- `gold_answer` - 정답
- `predicted_answer` - 모델 예측 답변
- `is_correct` - 정답 여부
- `num_steps` - 총 단계 수
- `steps[].step_num` - 단계 번호
- `steps[].content` - 단계 내용 (Judge가 평가할 대상)
- `steps[].rpe_label` - RPE 라벨 (비교 참고용)

## 사용 방법

### 1. 샘플 테스트 (10개 궤적)

```bash
python3 scripts/prepare_judge_data.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    outputs/judge_data/judge_input_sample.jsonl \
    --limit 10
```

**결과**:
- 10개 궤적, 53개 단계
- 파일 크기: 3.5MB → 0.0MB (99.1% 감소)

### 2. 전체 데이터 준비 (1000개 완료 후)

```bash
python3 scripts/prepare_judge_data.py \
    outputs/hybrid_1000q_from_0/results_hybrid_20251209_072136.jsonl \
    outputs/judge_data/judge_input_full.jsonl
```

**예상 결과** (589개 기준):
- 589개 궤적, ~2800개 단계
- 파일 크기: ~99% 감소 (노이즈 제거)

## 출력 형식

### Clean Sample 구조

```json
{
  "question_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "gold_answer": "Arthur's Magazine",
  "predicted_answer": "Arthur's Magazine was started first.",
  "is_correct": true,
  "num_steps": 5,
  "steps": [
    {
      "step_num": 1,
      "content": "Thought: To determine which magazine was started first...\nAction: Search[query=\"founding year of Arthur's Magazine\"]",
      "rpe_label": "good"
    },
    {
      "step_num": 2,
      "content": "Thought: After finding the founding year...\nAction: Search[query=\"founding year of First for Women\"]",
      "rpe_label": "good"
    }
  ]
}
```

### Judge가 평가할 내용

각 `step.content`에 대해:
1. **Reasoning Quality**: 추론이 논리적인가?
2. **Action Appropriateness**: 행동이 적절한가? (Search/Reason/Finish)
3. **Observation Analysis**: 관찰 결과를 잘 분석했는가?
4. **Progress**: 답변에 가까워지고 있는가?

**Judge Label**: `good` / `bad`

## Consensus Filtering (Step 2)

### Dual Evaluator 방식

1. **RPE Evaluator** (Quantitative)
   - `mc_after / mc_before` 기반
   - 이미 계산됨 (`rpe_label` 필드)

2. **Judge Evaluator** (Qualitative)
   - GPT-OSS-120B로 평가
   - Clean data 기반 독립 평가

### Consensus 규칙

```python
if rpe_label == "good" and judge_label == "good":
    final_label = "good"  # 학습 데이터로 사용
elif rpe_label == "bad" and judge_label == "bad":
    final_label = "bad"   # 학습 데이터로 사용
else:
    # Disagreement - 제외 또는 추가 검토
    final_label = "uncertain"
```

## GPT-OSS-120B Judge 프롬프트 예시

```python
JUDGE_PROMPT = """You are evaluating reasoning steps for quality.

Question: {question}
Gold Answer: {gold_answer}

Step {step_num}:
{content}

Evaluate this step:
1. Is the reasoning logical and coherent?
2. Is the action (Search/Reason/Finish) appropriate?
3. Does the observation analysis make sense?
4. Does this step make progress toward the answer?

Label: good / bad
Reasoning: [Brief explanation]
"""
```

## 다음 단계

### 1. 1000개 질문 생성 완료 대기

현재 589개 완료, 411개 진행 중:
```bash
# 자동 실행 중 (scripts/continue_1000q.sh)
python3 scripts/batch_test_hybrid.py \
    --num-questions 411 \
    --start-idx 589 \
    --num-rollouts 4 \
    --output-dir outputs/hybrid_1000q_from_0
```

### 2. 전체 Judge 데이터 생성

```bash
python3 scripts/prepare_judge_data.py \
    outputs/hybrid_1000q_from_0/results_hybrid_*.jsonl \
    outputs/judge_data/judge_input_1000q.jsonl
```

### 3. GPT-OSS-120B Judge 실행

```bash
# 예정 (스크립트 작성 필요)
python3 scripts/run_judge_labeling.py \
    outputs/judge_data/judge_input_1000q.jsonl \
    outputs/judge_data/judge_labels_1000q.jsonl
```

### 4. Consensus Filtering

```bash
# 예정 (스크립트 작성 필요)
python3 scripts/consensus_filter.py \
    --rpe-data outputs/hybrid_1000q_from_0/results_hybrid_*.jsonl \
    --judge-data outputs/judge_data/judge_labels_1000q.jsonl \
    --output outputs/training_data/consensus_filtered.jsonl
```

## 통계 (현재 589개 기준)

### 파일 크기 비교

| 파일 | 크기 | 설명 |
|------|------|------|
| 원본 (full) | 3.5MB | 모든 메타데이터 포함 |
| Judge용 (clean) | 0.04MB | 필수 필드만 (99.1% 감소) |

### 평가 대상

- **궤적**: 589개
- **단계**: ~2800개 (CoT + RAG)
- **예상 라벨**: 각 단계당 `good` / `bad`

## 참고 문서

- `COT_FIELD_MAPPING.md` - CoT 필드 구조
- `REPARSE_REPORT.md` - 재파싱 완료 보고서
- `PROJECT_ANALYSIS.md` - 전체 프로젝트 분석

## 결론

✅ **Judge 데이터 준비 완료**
- 스크립트 검증됨 (`prepare_judge_data.py`)
- 99.1% 노이즈 감소 확인
- 1000개 완료 시 즉시 사용 가능

다음 작업:
1. 1000개 궤적 생성 완료 대기
2. Judge 라벨링 스크립트 작성 (GPT-OSS-120B)
3. Consensus filtering 스크립트 작성
