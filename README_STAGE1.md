# Stage 1: Auto-Labeling Pipeline with Consensus Filtering

**목표**: Policy Model의 RAG-CoT trajectory에 대해 RPE와 Judge를 활용한 consensus filtering으로 고품질 training label을 자동 생성

**핵심 아이디어**:
- MC-based RPE (Monte Carlo confidence change)와 LLM Judge (QwQ-32B)의 label이 **일치하는 step만 사용**
- Disagreement는 필터링 → 높은 신뢰도의 0/1 label 확보
- Judge의 reasoning도 함께 저장하여 rationale 학습 가능

---

## 전체 파이프라인

```
Step 1: Policy Model Trajectory Generation
         ↓
    RAG-CoT trajectories (.jsonl)
    - question, steps (thought, action, observation)
         ↓
Step 2: RPE Labeling (MC-based)
         ↓
    RPE labels (GOOD/BAD per step)
    - mc_before, mc_after, rpe score
         ↓
Step 3: Judge Labeling (QwQ-32B)
         ↓
    Judge labels + reasoning (GOOD/BAD per step)
    - judge_label, judge_reasoning (200 chars), confidence
         ↓
Step 4: Consensus Filtering & Merging
         ↓
    Final Training Data
    - Consensus steps only (RPE = Judge)
    - Includes judge reasoning
    - Binary labels: 0 (BAD), 1 (GOOD)
```

---

## Step 1: Policy Model로 Trajectory 생성

### 개요
- Policy model (Qwen2.5-7B-Instruct + vLLM)이 RAG-CoT 추론 수행
- HotpotQA 데이터셋에서 질문 샘플링
- Hybrid retrieval (BM25 + dense) + reranker (BGE-reranker-v2-m3) 사용

### 실행
```bash
python scripts/batch_test_hybrid.py \
  --dataset hotpotqa \
  --split train \
  --question-level medium \
  --num-questions 1000 \
  --use-reranker \
  --rerank-top-n 20 \
  --output-dir outputs/hotpotqa_train_medium_reranker \
  --run-id reranker_1000_medium
```

### 출력 형식
```json
{
  "question_id": "5ab42ebd5542992339550047",
  "question": "Who hosted both Miss USA 1968 and The Price Is Right?",
  "gold_answer": "Bob Barker",
  "predicted_answer": "Bob Barker",
  "is_correct": true,
  "steps": [
    {
      "step_num": 1,
      "thought": "To answer this question, I need to recall who hosted...",
      "action": "Reason",
      "content": "To answer this question, I need to recall who hosted...",
      "num_passages": 0
    },
    {
      "step_num": 2,
      "thought": "Based on the retrieved information, Bob Barker hosted Miss USA 1968...",
      "action": "Search",
      "content": "Bob Barker hosted Miss USA 1968...",
      "observation": "1. Miss USA 1968 was the 17th Miss USA pageant...",
      "num_passages": 5
    },
    ...
  ]
}
```

**파일 위치**: `outputs/hotpotqa_train_medium_reranker/results_*.jsonl`

---

## Step 2: RPE Labeling (MC-based)

### 개요
- Monte Carlo confidence 변화로 각 step의 품질 평가
- mc_after / mc_before 비율 (Relative Probability Elevation)
- Threshold 기반 GOOD/BAD 분류

### RPE Score 계산
```python
rpe = mc_after / mc_before if mc_before > 0 else 0

# Label assignment (threshold-based)
if rpe >= 1.5:
    label = "good"  # Significant improvement
elif rpe < 0.8:
    label = "bad"   # Significant degradation
else:
    label = "borderline"  # Filtered in binary mode
```

### 이미 포함됨
- Policy model 실행 시 자동으로 MC confidence 계산
- 결과 파일에 `mc_before`, `mc_after`, `label` 필드 포함

### 출력 예시
```json
{
  "step_num": 1,
  "mc_before": 0.164,
  "mc_after": 0.305,
  "rpe": 1.75,
  "label": "good"
}
```

---

## Step 3: Judge Labeling (QwQ-32B)

### 개요
- QwQ-32B (32B reasoning model)를 Oracle로 사용
- VersaPRM-style prompt로 각 step 평가
- **Oracle Input Format** 적용:
  - **Past steps**: Thought + Action만 (Observation 제거 - 이미 다음 step의 Thought에 반영됨)
  - **Current step**: Thought + Action + **Observation** (평가 대상 - 검색된 문서가 좋은지 나쁜지 판단)

### Judge Prompt 구조
```
# Question
<질문>

# Correct Answer
<정답>

# Previous Steps (Context Only)
Step 1:
Thought: ...
Action: ...
[Observation 제거 - 토큰 절약]

# Current Step to Evaluate
Thought: ...
Action: ...
Observation (Retrieved Information):
<현재 step의 검색 결과 - 평가 필수>

# Your Evaluation
Reasoning: [Brief explanation in 2-3 sentences]
Label: [GOOD or BAD]
Confidence: [0.0 to 1.0]
```

### 실행
```bash
python scripts/judge_label_qwq.py
```

**설정**:
```python
config = {
    'model_name': 'Qwen/QwQ-32B',
    'temperature': 0.3,
    'max_tokens': 3072,  # QwQ의 긴 CoT 완성 허용
    'gpu_memory_utilization': 0.95,
    'prompt_style': 'versaprm',
    'use_gold_answer': True,
}
```

### 출력 형식
```json
{
  "question_id": "5ab42ebd5542992339550047",
  "question": "Who hosted both Miss USA 1968 and The Price Is Right?",
  "gold_answer": "Bob Barker",
  "steps": [
    {
      "step_num": 1,
      "rpe_label": "good",
      "judge_label": "GOOD",
      "judge_reasoning": "The step correctly identifies the need to connect the host of both Miss USA 1968 and *The Price Is Right*, aligning with the question's requirements. While it does not yet retrieve specific informatio...",
      "judge_confidence": 0.95,
      "mc_before": 0.164,
      "mc_after": 0.305,
      "rpe": 1.75
    }
  ]
}
```

**파일 위치**: `outputs/test_consensus_5q_judge_labels.jsonl`

### 주요 개선사항

#### 1. max_tokens 증가 (1024 → 3072)
QwQ-32B는 reasoning model로 긴 chain-of-thought를 생성합니다.
```python
# Before: Truncated response
max_tokens = 1024  # ❌ Output 중간에 잘림

# After: Complete response
max_tokens = 3072  # ✅ Full CoT + final judgment
```

#### 2. Parser 개선 (역순 탐색)
```python
# Parse from END to get final structured output
for line in reversed(lines):
    if confidence == 0.5 and line.startswith("Confidence:"):
        confidence = float(line.split("Confidence:", 1)[1].strip().split()[0])
    if label == "GOOD" and line.startswith("Label:"):
        label_text = line.split("Label:", 1)[1].strip().upper()
        if "BAD" in label_text:
            label = "BAD"
    if reasoning == "" and line.startswith("Reasoning:"):
        reasoning = line.split("Reasoning:", 1)[1].strip()
        break  # Found all, done
```

#### 3. Reasoning Truncation (200자)
```python
if len(reasoning) > 200:
    reasoning = reasoning[:200] + "..."
```
- 데이터 크기 절약
- 핵심 판단 근거만 저장

---

## Step 4: Consensus Filtering & Merging

### 개요
- RPE label과 Judge label을 비교
- **일치하는 step만 training data로 사용** (high-quality)
- Judge reasoning도 함께 저장

### Consensus 규칙
```python
if rpe_label == "good" and judge_label == "good":
    final_label = 1  # GOOD
    agree = True
elif rpe_label == "bad" and judge_label == "bad":
    final_label = 0  # BAD
    agree = True
else:
    final_label = None  # Filtered (disagreement)
    agree = False
```

### 실행 (간단 버전)
```python
python /tmp/merge_with_reasoning.py
```

### 최종 Training 데이터 형식
```json
{
  "question_id": "5ab42ebd5542992339550047",
  "question": "Who hosted both Miss USA 1968 and The Price Is Right?",
  "gold_answer": "Bob Barker",
  "predicted_answer": "Bob Barker",
  "is_correct": true,
  "num_steps_original": 3,
  "num_steps_filtered": 3,
  "steps": [
    {
      "step_num": 1,
      "thought": "To answer this question, I need to recall who hosted Miss USA 1968 and The Price Is Right during the same period.",
      "action": "Reason",
      "content": "To answer this question, I need to recall who hosted Miss USA 1968 and The Price Is Right during the same period.",
      "num_passages": 0,
      "mc_before": 0.164,
      "mc_after": 0.305,
      "rpe": 1.75,
      "label": "good",
      "label_consensus": {
        "rpe_label": "good",
        "judge_label": "good",
        "judge_reasoning": "The step correctly identifies the need to connect the host of both Miss USA 1968 and *The Price Is Right*, aligning with the question's requirements. While it does not yet retrieve specific informatio...",
        "judge_confidence": 0.95,
        "agree": true,
        "final_label": 1
      }
    },
    {
      "step_num": 2,
      "thought": "Based on the retrieved information, Bob Barker hosted Miss USA 1968. I need to check if Bob Barker also hosted The Price Is Right during the same period.",
      "action": "Search",
      "observation": "1. Miss USA 1968 was the 17th Miss USA pageant...",
      "num_passages": 5,
      "mc_before": 0.305,
      "mc_after": 0.828,
      "rpe": 2.632,
      "label": "good",
      "label_consensus": {
        "rpe_label": "good",
        "judge_label": "good",
        "judge_reasoning": "The step correctly identifies Bob Barker as the host of Miss USA 1968 based on the provided information. The next logical action is to verify if he also hosted *The Price Is Right* during the same per...",
        "judge_confidence": 1.0,
        "agree": true,
        "final_label": 1
      }
    },
    {
      "step_num": 3,
      "thought": "Based on the information provided, Bob Barker hosted The Price Is Right from 1959 to 1969, which includes the period of Miss USA 1968.",
      "action": "Finish",
      "content": "Bob Barker",
      "num_passages": 0,
      "mc_before": 0.828,
      "mc_after": 1.0,
      "rpe": 1.193,
      "label": "good",
      "label_consensus": {
        "rpe_label": "good",
        "judge_label": "good",
        "judge_reasoning": "The step correctly identifies that Bob Barker hosted *The Price Is Right* from 1959 to 1969, which includes 1968. Since the question asks for someone who hosted both *Miss USA 1968* and *The Price Is ...",
        "judge_confidence": 1.0,
        "agree": true,
        "final_label": 1
      }
    }
  ]
}
```

### 출력 파일
1. **All data** (`*_merged_all.jsonl`): 모든 step (disagreement 포함, consensus 정보 있음)
2. **Filtered** (`*_merged_filtered.jsonl`): Consensus만 (training용)

**파일 위치**: `outputs/test_consensus_5q_merged_*.jsonl`

---

## 사용 방법

### 전체 파이프라인 실행

```bash
# 1. Trajectory 생성 (RPE 자동 계산됨)
python scripts/batch_test_hybrid.py \
  --dataset hotpotqa \
  --split train \
  --question-level medium \
  --num-questions 1000 \
  --use-reranker \
  --output-dir outputs/hotpotqa_medium_1k

# 2. Judge labeling
# Edit scripts/judge_label_qwq.py to set input/output paths
python scripts/judge_label_qwq.py

# 3. Merge with consensus
python scripts/merge_consensus.py \
  --trajectories outputs/hotpotqa_medium_1k/results_*.jsonl \
  --judge-labels outputs/judge_labels.jsonl \
  --output-all outputs/merged_all.jsonl \
  --output-filtered outputs/merged_filtered.jsonl
```

### 테스트 (5 questions)

```bash
# 이미 생성된 테스트 데이터 사용
INPUT=/root/.local/PRMRAG/outputs/test_consensus_5q.jsonl
JUDGE=/root/.local/PRMRAG/outputs/test_consensus_5q_judge_labels.jsonl

# Merge
python /tmp/merge_with_reasoning.py
```

---

## 결과 통계 (Test 5 Questions)

### Consensus Statistics
```
Total steps:              14
  Both Good (label=1):    10 (71.4%)
  Both Bad  (label=0):     2 (14.3%)
  Disagreement (filtered): 2 (14.3%)

Agreement Rate: 85.7%
```

### Disagreement 분석
```
Q3 Step 1: RPE=bad, Judge=GOOD
  - Judge reasoning: "The step correctly identifies that the initial document
    mentions El-P as the Definitive Jux label head (a record executive) but
    dismisses him as a guest vocalist..."

Q3 Step 2: RPE=good, Judge=BAD
  - Judge reasoning: "The step incorrectly identifies Chris Hicks as the
    guest executive without any evidence connecting him to the EP..."
```

→ Judge가 RPE보다 더 세밀하게 평가 (문서 내용 직접 확인)

---

## 주요 파일 구조

```
/root/.local/PRMRAG/
├── src/prmrag/
│   ├── labeling/
│   │   ├── judge_labeler.py      # QwQ-32B Judge 구현
│   │   ├── consensus.py           # Consensus filtering 로직
│   │   └── rpe_labeler.py         # RPE labeling 로직
│   └── models/
│       └── policy_model_vllm.py   # vLLM Policy model
├── scripts/
│   ├── batch_test_hybrid.py       # Trajectory 생성
│   ├── judge_label_qwq.py         # Judge labeling 실행
│   ├── consensus_filter.py        # 기존 consensus 스크립트
│   └── merge_consensus.py         # Merge 스크립트 (TODO)
└── outputs/
    ├── test_consensus_5q.jsonl                    # Policy trajectories
    ├── test_consensus_5q_judge_labels.jsonl       # Judge labels
    ├── test_consensus_5q_merged_all.jsonl         # All (with consensus info)
    └── test_consensus_5q_merged_filtered.jsonl    # Training data (consensus only)
```

---

## 핵심 개념 정리

### 1. RPE (Relative Probability Elevation)
- **정의**: `mc_after / mc_before`
- **의미**: 해당 step이 정답 확률을 얼마나 향상시켰는가
- **장점**: Fast, 모델 자체 신뢰도 활용
- **단점**: Calibration 문제, shallow reasoning

### 2. Judge (QwQ-32B Oracle)
- **정의**: 큰 reasoning model이 step 품질 직접 평가
- **의미**: 검색된 문서가 질문에 도움이 되는가
- **장점**: Deep reasoning, 실제 문서 내용 확인
- **단점**: Slow, expensive, hallucination 가능

### 3. Consensus Filtering
- **정의**: RPE와 Judge가 **둘 다 동의하는 step만 사용**
- **의미**: High-confidence labels only
- **효과**:
  - Label noise 제거
  - Training data quality ↑
  - Agreement rate: 85.7% (test)

### 4. Oracle Input Format
- **Past steps**: Thought + Action만
  - Observation은 이미 다음 Thought에 반영됨
  - 토큰 절약 (중복 제거)
- **Current step**: Thought + Action + **Observation**
  - 현재 검색 결과를 평가해야 하므로 필수
  - 문서가 좋은지 나쁜지 판단

---

## 다음 단계 (Stage 2)

Stage 1에서 생성한 고품질 label로 **PRM (Process Reward Model) 학습**:

1. **Input**: Trajectory prefix (question + steps 1~t)
2. **Output**: Step t의 품질 예측 (0 or 1)
3. **Training**: Consensus label 사용
4. **Objective**: Accurate step-level evaluation

→ `README_STAGE2.md` 참고 (TODO)

---

## 트러블슈팅

### Judge labeling 시 empty reasoning
**증상**: `judge_reasoning` 필드가 빈 문자열
**원인**:
1. `max_tokens` 부족 (QwQ의 긴 CoT가 잘림)
2. Parser가 중간 output 파싱 (최종 judgment 도달 못함)

**해결**:
```python
# max_tokens 증가
max_tokens = 3072  # 1024 → 3072

# Parser 역순 탐색 (최종 output 우선)
for line in reversed(lines):
    if line.startswith("Reasoning:"):
        reasoning = line.split("Reasoning:", 1)[1].strip()
```

### Consensus rate가 100%로 나옴
**증상**: RPE vs Judge 비교 시 100% agreement
**원인**: Judge labels 파일에 `rpe_label` 필드도 포함되어 있어, `load_labels()` 함수가 judge_label 대신 rpe_label을 가져옴

**해결**:
```python
# Judge labels 로드 시 judge_label만 명시적으로 추출
judge_labels = {
    step['step_num']: step['judge_label'].lower()
    for step in data['steps']
}
```

### GPU OOM
**증상**: QwQ-32B 로딩 시 메모리 부족
**해결**:
```bash
# 이전 프로세스 정리
nvidia-smi  # PID 확인
kill -9 <PID>

# gpu_memory_utilization 조정
gpu_memory_utilization = 0.95  # 0.8 → 0.95
```

---

## 참고 문헌

- **VersaPRM**: [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)
- **Math-Shepherd**: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- **QwQ-32B**: Qwen Reasoning Model
- **RPE**: Relative Probability Elevation (custom metric)

---

**문서 작성일**: 2025-12-18
**테스트 데이터**: HotpotQA train (5 questions, medium difficulty)
**Agreement Rate**: 85.7% (12/14 steps)
