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
if P_cot >= (1 - δ):  # 예: 0.8 (RPE threshold)
    # Accept CoT
    steps.append({
        'step_type': 'cot',
        'action': 'Reason',
        'text': cot_step,
        'mc_before': MC_prev,
        'mc_after': MC_cot,
        'rpe': P_cot,
        'label': 'good' if P_cot >= 0.8 else 'bad'
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
            'step_type': 'rag',
            'action': 'Search',
            'action_input': query,  # Sub-query
            'observation': format_passages(passages),
            'sub_answer': extract_answer(observation),
            'passages': passages,
            'mc_before': MC_prev,
            'mc_after': best_mc,
            'rpe': P_rag,
            'label': 'good' if P_rag >= 0.8 else 'bad'
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
      "text": "Step 1: Let me think...",
      "content": "Let me think...",
      "used_passages": [],
      "mc_before": 0.5,
      "mc_after": 0.7,
      "rpe": 1.4,
      "label": "good",

      "thought": "I need to understand the question first",
      "action": "Reason",
      "action_input": null,
      "observation": null,
      "sub_answer": null,
      "metadata": {"accepted": "cot", "threshold": 0.8}
    },
    {
      "step_id": 1,
      "step_type": "rag",
      "text": "Step 2: Thought: I need more information...",
      "content": "Thought: I need more information...\nAction: Search[query=\"...\"]\nObservation: ...\nSub-answer: ...",
      "used_passages": [
        {"doc_id": "doc_1", "title": "...", "text": "...", "score": 0.9}
      ],
      "mc_before": 0.7,
      "mc_after": 0.85,
      "rpe": 1.21,
      "label": "good",

      "thought": "I need more information about...",
      "action": "Search",
      "action_input": "who directed the film",
      "observation": "According to [1], the director is...",
      "sub_answer": "The director is David O. Russell",
      "metadata": {"accepted": "rag", "threshold": 0.8}
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

### 4. ReAct 구조

각 step은 **ReAct format**을 따릅니다:

```
Step N:
Thought: [reasoning about what's needed]
Action: Search[query="sub-question"] or Finish[answer="..."]
Observation: [retrieved information - RAG only]
Sub-answer: [intermediate answer extracted from observation]
```

#### 액션 타입:
- **Search**: 외부 정보 검색 필요 (RAG step)
  - Format: `Action: Search[query="specific sub-question"]`
  - action_input에 sub-query 저장

- **Reason**: 내부 추론 (CoT step)
  - Format: 자유 형식 추론
  - action_input은 null

- **Finish**: 최종 답변
  - Format: `Action: Finish[answer="direct answer"]` 또는 `Final Answer: ...`

## 사용법

### 1. DPR Wikipedia Corpus 준비

```bash
# DPR Wikipedia (21M passages) 다운로드 및 BGE-M3 임베딩 생성
python scripts/setup_dpr_wikipedia.py
# 예상 시간: ~6-8시간 (GPU 사양에 따라)
```

생성되는 파일:
- `~/.cache/huggingface/datasets/facebook___wiki_dpr/psgs_w100.nq.no_index/0.0.0/*/` - DPR corpus
- `data/embeddings/dpr_wikipedia_bge_m3.pkl` - BGE-M3 임베딩 (~60-80GB)

### 2. Trajectory 생성

```bash
python scripts/batch_test_adaptive.py \
    --start-idx 1 \
    --num-questions 10 \
    --split train \
    --corpus-type dpr \
    --output-dir outputs/batch_train_test \
    --num-rollouts 8
```

### 3. 결과 확인

생성된 파일:
- `outputs/batch_train_test/trajectories.jsonl` - 전체 trajectories
- `outputs/batch_train_test/summary.json` - 통계 요약

## 설정

`configs/adaptive_generation.yaml`:

```yaml
adaptive:
  # MC estimation
  num_rollouts: 8
  use_dynamic_k: true  # Step 1 이후 dynamic K 사용

  # RPE thresholds
  rpe_threshold: 0.8  # RPE >= 0.8 → 'good', < 0.8 → 'bad'

  # Generation limits
  max_steps: 15
  num_trajectories_per_question: 1

  # RAG
  num_rag_queries: 3
  top_k_passages: 5

  # Retriever
  corpus_type: "dpr"  # "dpr" or "hotpotqa"
  retriever_model: "BAAI/bge-m3"

# Policy model (vLLM)
policy_model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  max_tokens: 200
  temperature: 0.8
  top_p: 0.95
  seed: 42
```

### 주요 파라미터

- **rpe_threshold**: Step label 결정
  - 0.8 → RPE >= 0.8이면 'good', 아니면 'bad'
  - RPE = MC_after / MC_before

- **num_rollouts**: MC 추정 정확도
  - 많을수록 정확하지만 느림
  - 8이 기본값 (vLLM batch generation으로 빠름)

- **use_dynamic_k**: Dynamic K 사용 여부
  - true → Step 1 이후 K 자동 조정
  - false → 고정 num_rollouts 사용

## 구현 상세

### 파일 구조

```
src/prmrag/
├── retrieval/
│   ├── bge_retriever.py       # BGERetriever (dense retrieval)
│   ├── bm25_retriever.py      # BM25Retriever (sparse)
│   └── hybrid_retriever.py    # HybridRetriever (BM25+BGE)
├── generation/
│   ├── adaptive_generator.py  # AdaptiveTrajectoryGenerator
│   └── step_forcing.py        # StepParser, ForcedStep
├── models/
│   └── policy_model_vllm.py   # PolicyModelVLLM (vLLM wrapper)
├── labeling/
│   ├── judge_labeler.py       # JudgeLabeler (LLM-as-a-judge)
│   └── rpe_labeler.py         # RPELabeler (RPE-based)
└── utils/
    └── answer_utils.py        # Answer extraction & matching

scripts/
├── batch_test_adaptive.py     # Main batch testing script
├── setup_dpr_wikipedia.py     # DPR corpus & embedding setup
└── generate_adaptive_trajectories.py  # Single trajectory generation

configs/
└── adaptive_generation.yaml   # Configuration
```

### 핵심 클래스

#### 1. `BGERetriever` (Dense Retrieval)
```python
from prmrag.retrieval.bge_retriever import BGERetriever

retriever = BGERetriever(
    corpus_type="dpr",
    model_name="BAAI/bge-m3",
    embeddings_path="data/embeddings/dpr_wikipedia_bge_m3.pkl"
)

results = retriever.retrieve(query, top_k=5)
# Returns: [{'doc_id', 'title', 'text', 'score'}, ...]
```

#### 2. `PolicyModelVLLM` (vLLM)
```python
from prmrag.models.policy_model_vllm import PolicyModelVLLM

model = PolicyModelVLLM(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    temperature=0.8,
    seed=42
)

# Batch generation (optimized)
responses = model.batch_generate_with_chat_template(
    user_messages=prompts,
    max_tokens=200,
    temperature=0.8,
)
```

#### 3. `AdaptiveTrajectoryGenerator`
```python
from prmrag.generation.adaptive_generator import AdaptiveTrajectoryGenerator

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

#### 4. `AdaptiveStep` 데이터 구조
```python
@dataclass
class AdaptiveStep:
    step_id: int
    step_type: StepType  # COT or RAG
    text: str  # "Step N: ..."
    content: str  # Content without "Step N:" prefix
    used_passages: List[Dict]
    mc_before: float
    mc_after: float
    rpe: float
    label: str  # 'good' or 'bad'

    # ReAct structure
    thought: Optional[str] = None
    action: str = "Reason"  # "Search", "Reason", "Finish"
    action_input: Optional[str] = None  # Sub-query for Search
    observation: Optional[str] = None  # Retrieved info
    sub_answer: Optional[str] = None  # Intermediate answer

    metadata: Dict = field(default_factory=dict)
```

### System Instruction

모든 프롬프트는 **system instruction**을 활용합니다:

```python
# policy_model_vllm.py:199-269
system_prompt = """You are an expert question-answering agent...

# AVAILABLE ACTIONS
1. **Search** - Retrieve information
   Format: Action: Search[query="sub-question"]

2. **Finish** - Provide final answer
   Format: Action: Finish[answer="direct answer"]

# RESPONSE FORMAT
Step N:
Thought: [reasoning]
Action: Search[query="..."] or Finish[answer="..."]
Observation: [search results]
Sub-answer: [intermediate answer]

# IMPORTANT RULES
- One step at a time
- Only use Search or Finish
- Extract sub-answers from observations ONLY
- Concise final answers
..."""
```

User prompts는 매우 간결:
```python
# CoT prompt
f"Question: {question}\n\nGenerate Step 1."

# RAG prompt
f"Question: {question}\nRetrieved Documents:\n{passages}\n\nGenerate Step {N}."

# Rollout prompt
f"Question: {question}\nReasoning so far:\n{steps}\n\nContinue solving and provide your final answer."
```

## 실제 사용 시 주의사항

### 1. GPU 메모리 관리

vLLM 사용 시:
```python
# GPU 메모리 사용량 조절
gpu_memory_utilization=0.8  # 기본값
tensor_parallel_size=1      # Multi-GPU 시 조정
```

### 2. DPR Corpus 용량

- 원본 데이터: ~13GB (21M passages)
- BGE-M3 임베딩: ~60-80GB
- 총 필요 디스크: ~100GB

### 3. 대규모 처리

Batch processing:
```bash
# 100개씩 처리
for start in {1..1000..100}; do
    python scripts/batch_test_adaptive.py \
        --start-idx $start \
        --num-questions 100 \
        --output-dir outputs/batch_${start}
done
```

### 4. Answer Matching

```python
from prmrag.utils.answer_utils import (
    extract_answer_from_text,
    normalize_answer,
    check_answer_match
)

# Extract answer
answer = extract_answer_from_text(model_output)

# Normalize & match
predicted = normalize_answer(answer)
gold = normalize_answer(gold_answer)
is_correct = check_answer_match(predicted, gold, threshold=0.8)
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
# adaptive_generator.py에 로깅 추가
print(f"[MC DEBUG] MC={mc_value:.3f}, K={k}, successes={successes}/{k}")
```

2. **Threshold 조정**
```yaml
adaptive:
  rpe_threshold: 0.7  # More lenient
```

3. **Step 제한**
```yaml
adaptive:
  max_steps: 5  # Shorter trajectories
```

4. **파싱 확인**
```python
from prmrag.generation.step_forcing import StepParser

parsed = StepParser.parse_react_components(step_content)
print(f"Thought: {parsed['thought']}")
print(f"Action: {parsed['action']}")
print(f"Action Input: {parsed['action_input']}")
```

## 참고

- vLLM: https://github.com/vllm-project/vllm
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- DPR: https://github.com/facebookresearch/DPR
- HotpotQA: https://hotpotqa.github.io/
