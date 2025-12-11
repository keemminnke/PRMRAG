# 완전한 수정 계획 - 옵션 2

## 목표
996개 궤적을 깨끗하게 재생성 (환각 0%, 올바른 분류)

## 수정할 파일

### 1. src/prmrag/models/policy_model_vllm.py
#### 수정 A: 프롬프트 개선 (Line 286-291)

**현재**:
```python
**CRITICAL for Search steps:**
- After you write the Action, retrieved documents will be provided to you
- You MUST then write Observation by reading and summarizing those documents
- You MUST then write Sub-answer based on what you found in Observation
- Do NOT stop after writing Action - complete Observation and Sub-answer!
```

**수정 후**:
```python
**CRITICAL for Search steps:**
- When you need information, write ONLY: Thought + Action
- Format: Action: Search[query="your specific question"]
- STOP immediately after writing the Action line
- The system will automatically retrieve documents and provide Observation
- DO NOT write Observation yourself - it causes errors
- DO NOT write Sub-answer yourself - it will be provided
```

#### 수정 B: Stop sequence 추가 (Line 358-379)

**현재**:
```python
def generate_with_chat_template(
    self,
    user_message: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
) -> str:
    """Generate using Qwen chat template.
    ...
    """
    prompt = self.format_prompt_for_qwen(user_message)
    return self.generate(prompt, max_tokens, temperature, top_p, stop_sequences)
```

**수정 후**:
```python
def generate_with_chat_template(
    self,
    user_message: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
) -> str:
    """Generate using Qwen chat template.
    ...
    """
    prompt = self.format_prompt_for_qwen(user_message)

    # Prevent hallucinated observations by stopping before "Observation:"
    if stop_sequences is None:
        stop_sequences = []
    else:
        stop_sequences = list(stop_sequences)  # Copy to avoid modifying caller's list

    # Add stop sequences to prevent model from writing Observation
    stop_sequences.extend(["\nObservation:", "Observation:"])

    return self.generate(prompt, max_tokens, temperature, top_p, stop_sequences)
```

### 2. src/prmrag/generation/adaptive_generator.py

**확인 사항**: num_passages 필드가 모든 스텝에 저장되는지 확인

생성 로직에서 step dict를 만들 때:
```python
step = {
    'step_num': i,
    'content': content,
    'action': action,
    'num_passages': len(passages) if passages else 0,  # ← 이 부분 확인
    ...
}
```

## 실행 단계

### Phase 1: 백업 (1분)

```bash
# 현재 파일 백업
cp src/prmrag/models/policy_model_vllm.py src/prmrag/models/policy_model_vllm.py.backup
```

### Phase 2: 코드 수정 (5분)

수정 A, B 적용

### Phase 3: 검증 스크립트 작성 (5분)

```python
# scripts/validate_trajectories.py
```

### Phase 4: 2개 테스트 (5분)

```bash
python3 scripts/batch_test_hybrid.py \
    --num-questions 2 \
    --start-idx 0 \
    --num-rollouts 4 \
    --output-dir outputs/test_2q
```

### Phase 5: 검증 (5분)

```bash
python3 scripts/validate_trajectories.py outputs/test_2q/*.jsonl
```

환각률 0% 확인

### Phase 6: 10개 테스트 (15분)

```bash
python3 scripts/batch_test_hybrid.py \
    --num-questions 10 \
    --start-idx 0 \
    --num-rollouts 4 \
    --output-dir outputs/test_10q
```

### Phase 7: 최종 검증 (5분)

```bash
python3 scripts/validate_trajectories.py outputs/test_10q/*.jsonl
python3 scripts/quick_stats.py outputs/test_10q/*.jsonl
```

### Phase 8: 전체 생성 (8-12시간)

```bash
python3 scripts/batch_test_hybrid.py \
    --num-questions 996 \
    --start-idx 0 \
    --num-rollouts 4 \
    --output-dir outputs/hybrid_1000q_CLEAN
```

### Phase 9: 최종 검증 (10분)

```bash
python3 scripts/validate_trajectories.py outputs/hybrid_1000q_CLEAN/*.jsonl
python3 scripts/quick_stats.py outputs/hybrid_1000q_CLEAN/*.jsonl
```

## 예상 결과

**Phase 4 후** (2개):
- 환각: 0개
- 잘못 분류: 0개
- 모든 스텝에 num_passages 있음

**Phase 6 후** (10개):
- 환각률: 0%
- 정확도: ~50% (동일 예상)
- 깨끗한 데이터 확인

**Phase 8 후** (996개):
- 환각: 0개 (0%)
- 잘못 분류: 0개 (0%)
- 깨끗한 4,700+ 스텝
- Step 2 Consensus Labeling 준비 완료

## 실패 시 대응

### 만약 Phase 4에서 여전히 환각 발견:

1. Stop sequence가 작동하는지 확인
2. 프롬프트 더 강하게 수정
3. 디버깅용 2개 수동 확인

### 만약 Phase 6에서 문제 발견:

1. 원인 분석
2. 추가 수정
3. 다시 2개 테스트

## 총 소요 시간

- Phase 1-7: 약 1시간
- Phase 8: 8-12시간 (하룻밤)
- Phase 9: 10분

**총: ~9-13시간 (대부분 자동)**

## 시작 준비 완료

모든 단계가 명확합니다. 에러 최소화를 위해:
1. 한 번에 하나씩
2. 각 단계 검증
3. 문제 발견 시 즉시 수정

**시작하시겠습니까?**
