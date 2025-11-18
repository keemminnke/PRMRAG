# Stage 1: Adaptive MC-CoT + RAG Trajectory Generation

## 개요

Stage 1은 **Adaptive MC-CoT + RAG Intervention**을 통해 고품질 trajectory를 생성합니다.

### 핵심 아이디어

"CoT를 하다가 MC가 떨어지면 → RAG로 개입 → MC가 회복되면 계속"

```
Question
  ↓
[CoT Step] → MC 계산 → RPE >= 1-δ?
  ↓ Yes                    ↓ No
Accept                  [Rollback]
  ↓                         ↓
Next step             [RAG Intervention]
                          ↓
                    Generate queries
                    Retrieve passages
                    Select best (MC)
                          ↓
                    RPE >= 1+ε?
                      ↓ Yes    ↓ No
                    Accept   Terminate
```

## 알고리즘

### 1. 초기 상태
```python
s_0 = {
    'question': q,
    'reasoning_history': [],
    'passages': []
}
MC_0 = MC(s_0)
```

### 2. 각 Step에서

#### (A) CoT Step 시도
```python
# 1. CoT step 생성
cot_step = LLM.generate_reasoning(s_{t-1})

# 2. MC 계산
MC_prev = MC(s_{t-1})
MC_cot = MC(s_t with cot_step)

# 3. RPE 계산
P_cot = MC_cot / MC_prev
```

#### (B) CoT 채택 여부
```python
if P_cot >= (1 - δ):  # 예: 0.9
    # Accept CoT
    steps.append({
        'type': 'cot',
        'text': cot_step,
        'mc_before': MC_prev,
        'mc_after': MC_cot,
        'rpe': P_cot
    })
    continue
```

#### (C) RAG Intervention
```python
else:
    # CoT 실패 → RAG 시도
    queries = generate_rag_queries(s_{t-1})

    best_mc = -inf
    for query in queries:
        passages = retrieve(query, top_k=5)
        s_rag = apply_rag(s_{t-1}, passages)
        mc_rag = MC(s_rag)

        if mc_rag > best_mc:
            best_mc = mc_rag
            best_rag = (query, passages, s_rag)

    P_rag = best_mc / MC_prev

    if P_rag >= (1 + ε):  # 예: 1.1
        # Accept RAG
        steps.append({
            'type': 'rag',
            'text': format_rag(query, passages),
            'passages': passages,
            'mc_before': MC_prev,
            'mc_after': best_mc,
            'rpe': P_rag
        })
        continue
    else:
        # 둘 다 실패 → trajectory 종료
        break
```

### 3. 출력 형식

```json
{
  "trajectory_id": "q001_0",
  "question": "What is...?",
  "steps": [
    {
      "step_id": 0,
      "step_type": "cot",
      "text": "Let me think...",
      "used_passages": [],
      "mc_before": 0.5,
      "mc_after": 0.7,
      "rpe": 1.4,
      "metadata": {}
    },
    {
      "step_id": 1,
      "step_type": "rag",
      "text": "Retrieved information...",
      "used_passages": [
        {"doc_id": "doc_1", "title": "...", "text": "...", "score": 0.9}
      ],
      "mc_before": 0.7,
      "mc_after": 0.85,
      "rpe": 1.21,
      "metadata": {}
    }
  ],
  "final_answer": "The answer is...",
  "gold_answer": "...",
  "is_correct": true,
  "metadata": {
    "num_steps": 2,
    "num_cot_steps": 1,
    "num_rag_steps": 1
  }
}
```

## 사용법

### 1. 예제 데이터 생성

```bash
python scripts/create_example_hotpotqa.py
```

생성되는 파일:
- `data/raw/example_hotpotqa_questions.jsonl`
- `data/raw/example_hotpotqa_corpus.jsonl`

### 2. Trajectory 생성

```bash
python scripts/generate_adaptive_trajectories.py \
    --config configs/adaptive_generation.yaml \
    --questions data/raw/example_hotpotqa_questions.jsonl \
    --corpus data/raw/example_hotpotqa_corpus.jsonl \
    --output data/generated/adaptive_trajectories.jsonl \
    --num-trajectories 3
```

### 3. 결과 확인

생성된 파일:
- `data/generated/adaptive_trajectories.jsonl` - 전체 trajectories
- `data/generated/adaptive_trajectories_stats.json` - 통계

## 설정

`configs/adaptive_generation.yaml`:

```yaml
adaptive:
  # MC estimation
  num_rollouts: 5
 

  # Generation limits
  max_steps: 15
  num_trajectories_per_question: 3

  # RAG
  num_rag_queries: 3
  top_k_passages: 5
```

### 주요 파라미터

- **delta (δ)**: CoT 허용 범위
  - δ=0.1 → MC가 10% 이상 떨어지면 RAG 개입
  - 작을수록 보수적 (RAG를 더 자주 사용)

- **epsilon (ε)**: RAG 개선 임계값
  - ε=0.1 → MC가 10% 이상 올라가야 RAG 채택
  - 클수록 보수적 (RAG 채택이 까다로움)

- **num_rollouts**: MC 추정 정확도
  - 많을수록 정확하지만 느림
  - 5-10이 일반적

## 구현 상세

### 파일 구조

```
src/prmrag/
├── retrieval/
│   └── __init__.py              # BM25Retriever, load_hotpotqa_corpus
├── generation/
│   ├── __init__.py
│   └── adaptive_generator.py   # AdaptiveTrajectoryGenerator
└── ...

scripts/
├── generate_adaptive_trajectories.py  # Main script
└── create_example_hotpotqa.py         # Example data

configs/
└── adaptive_generation.yaml           # Configuration
```

### 핵심 클래스

#### 1. `BM25Retriever`
```python
retriever = BM25Retriever(corpus)
results = retriever.retrieve(query, top_k=5)
# Returns: [{'doc_id', 'title', 'text', 'score'}, ...]
```

#### 2. `AdaptiveTrajectoryGenerator`
```python
generator = AdaptiveTrajectoryGenerator(
    policy_model=model,
    retriever=retriever,
    config=config
)

trajectory = generator.generate_trajectory(
    question="What is...?",
    gold_answer="...",
)
```

#### 3. `AdaptiveTrajectory`
```python
trajectory = AdaptiveTrajectory(
    trajectory_id="q001_0",
    question="...",
    steps=[...],        # List[AdaptiveStep]
    final_answer="...",
    is_correct=True,
)
```

## 실제 사용 시 주의사항

### 1. 모델 로딩

현재는 placeholder입니다. 실제 사용 시:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

generator = AdaptiveTrajectoryGenerator(
    policy_model=model,
    retriever=retriever,
    config=config
)
```

### 2. HotpotQA 데이터

실제 HotpotQA 데이터 다운로드:
```bash
# Download from: https://hotpotqa.github.io/
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

### 3. 대규모 처리

메모리 관리:
```python
# Batch processing
for batch in batched(questions, batch_size=16):
    trajectories = generator.generate_batch(batch)
    save_batch(trajectories)
    del trajectories  # Free memory
```

## 다음 단계

Stage 1 완료 후:
1. Generated trajectories → Stage 2 (Judge labeling)
2. Consensus filtering
3. PRM training

## 디버깅

문제가 발생하면:

1. **MC 계산 확인**
```python
# Add logging to _monte_carlo_estimate
print(f"MC estimation: {successes}/{num_rollouts} = {mc:.3f}")
```

2. **Threshold 조정**
```yaml
adaptive:
  delta: 0.2  # More lenient (less RAG)
  epsilon: 0.05  # Less strict (more RAG acceptance)
```

3. **Step 제한**
```yaml
adaptive:
  max_steps: 5  # Shorter trajectories for debugging
```

## 참고

- BM25: https://github.com/dorianbrown/rank_bm25
- HotpotQA: https://hotpotqa.github.io/
