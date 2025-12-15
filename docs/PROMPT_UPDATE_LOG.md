# 프롬프트 업데이트 로그: Reason 액션 추가

## 변경 일시
2025-12-11

## 변경 사유
**문제**: 모델이 모든 스텝에서 "Action: Search"를 작성함 (100% 검색 시도율)

**원인 분석**:
1. 기존 프롬프트는 2가지 액션만 제공:
   - `Search` - 정보 검색
   - `Finish` - 최종 답변
2. 순수 추론(CoT) 액션이 없음
3. 중간 단계에서는 Finish 불가 → 무조건 Search 선택
4. Adaptive generator가 RPE >= 0.8이면 검색 차단
   - 결과: content에 "Action: Search" 텍스트, 하지만 action='Reason', num_passages=0
   - content 필드와 action 필드 불일치

## 변경 내용

### 1. AVAILABLE ACTIONS 섹션

**Before:**
```
You have two actions available:

1. **Search** - Retrieve information from the knowledge base
2. **Finish** - Provide the final answer
```

**After:**
```
You have three actions available:

1. **Reason** - Continue reasoning with your own knowledge
   - Use when you can make progress without external information
   - Just write your reasoning process

2. **Search** - Retrieve information from the knowledge base
   - Use when you need external information you don't have

3. **Finish** - Provide the final answer
   - Use when you can answer the main question
```

### 2. RESPONSE FORMAT 섹션

**추가된 내용:**
```
**When you can reason with existing knowledge (Reason):**
Step N:
Thought: [Your reasoning process]
[Continue your logical deduction based on what you already know]
```

### 3. IMPORTANT RULES 섹션

**Before:**
```
2. **Actions**: You can ONLY use Search or Finish - no other actions exist
```

**After:**
```
2. **Actions**: You can use Reason, Search, or Finish
   - Use **Reason** when you can make logical deductions from existing knowledge
   - Use **Search** when you need external information you don't have
   - Use **Finish** when you can provide the final answer

3. **Choose wisely**:
   - For questions requiring specific facts (dates, names, locations): use Search
   - For logical reasoning, comparisons, or synthesis: use Reason
   - Example: "When was X founded?" → Search needed
   - Example: "Which is older, X or Y?" → After getting dates, use Reason to compare
```

### 4. TASK 섹션

**Before:**
```
2. Decide whether to search for information or provide the final answer
```

**After:**
```
2. Decide whether to:
   - **Reason** with your existing knowledge
   - **Search** for external information
   - **Finish** with the final answer
```

## 파일 위치
`/root/.local/PRMRAG/src/prmrag/models/policy_model_vllm.py`

Lines 245-355 (system_prompt)

## 기대 효과

### 1. Content와 Action 일치
- 모델이 Reason 선택 → content에 순수 추론, action='Reason'
- 모델이 Search 선택 → content에 "Action: Search", action='Search' (또는 RPE로 차단시 'Reason')

### 2. 더 나은 의도 표현
- 모델이 "추론할 수 있다"고 판단하면 Reason 선택
- "외부 정보 필요"하면 Search 선택
- Adaptive generator는 여전히 RPE 기반 개입 가능

### 3. 데이터 품질 향상
- PRM training data에서 모델의 의도가 명확히 드러남
- "Action: Search" 텍스트가 있는데 action='Reason'인 혼란 제거

## 잠재적 위험

### 1. 모델이 자기 지식 과신
- HotpotQA는 대부분 외부 지식 필요
- 모델이 Reason을 과도하게 선택할 수 있음
- 결과: hallucination 증가 가능

### 2. Adaptive Generator 효과 감소
- 모델이 Reason 선택 → Adaptive generator 개입 기회 감소
- RPE 기반 RAG intervention이 덜 작동할 수 있음

## 검증 결과

### Phase 1: 2q 테스트 (2025-12-11 10:21-10:28)

**Test Data**: `outputs/test_2q_reason/results_hybrid_20251211_102146.jsonl`

**결과 요약**:
- ✓ 모델이 순수 추론(pure reasoning)을 시도함
- ⚠ 여전히 90%의 스텝에서 "Action: Search" 텍스트 작성
- ✓ 10%의 스텝에서 순수 추론 (Search 텍스트 없음)

**통계**:
- 총 스텝 수: 20개
- "Action: Search" 텍스트 포함: 18개 (90%)
- action='Reason': 18개 (90%)
- action='Search' (실제 검색): 2개 (10%)
- **순수 추론** (action='Reason', Search 텍스트 없음): **2개 (10%)**

**비교 (Old vs New)**:
```
OLD PROMPT (test_10q):
  - 100% steps had "Action: Search" text
  - 82% blocked by RPE → action='Reason'
  - 18% executed → action='Search'
  - 0% pure reasoning

NEW PROMPT (test_2q_reason):
  - 90% steps have "Action: Search" text (↓10%)
  - 80% blocked by RPE → action='Reason'
  - 10% executed → action='Search'
  - 10% pure reasoning (NEW!)
```

**핵심 발견**:
1. **Reason 액션 사용 시작**: Step 1에서 모델이 순수 추론 시도
   - Q1: "To compare the founding years... I need to recall or search..."
   - Q2: "To answer the question, I need to know which hotel company..."

2. **여전히 Search 편향 존재**: 90%는 여전히 Search 작성
   - 모델이 외부 정보 필요성 인식
   - HotpotQA 특성상 자연스러움

3. **Adaptive Generator 여전히 작동**:
   - 80%의 Search 시도를 RPE로 차단
   - 설계대로 작동 중

**결론**:
- ✅ **부분적 성공**: 모델이 순수 추론 옵션 사용 시작
- ⚠ **완전 해결 아님**: 여전히 대부분 Search 시도
- 🔍 **추가 분석 필요**: 왜 10%만 순수 추론인가?

### Phase 2: 10q 테스트
- TBD

### Phase 3: 100q 테스트 (선택)
- TBD

## 검증 스크립트

```python
# outputs/test_2q_reason/results_*.jsonl 분석
for step in trajectory['steps']:
    action = step['action']
    content = step['content']

    # Reason action이 사용되는지
    if action == 'Reason':
        print(f"✓ Reason 사용: {content[:100]}")

    # Content에 "Action: Search"가 있는데 action='Reason'인 경우
    if 'Action: Search' in content and action == 'Reason':
        print(f"⚠️ 여전히 불일치 발생")
```

## 롤백 절차

만약 결과가 좋지 않으면:
1. `git diff src/prmrag/models/policy_model_vllm.py` 확인
2. `git checkout src/prmrag/models/policy_model_vllm.py` 복구
3. 또는 이 문서의 "Before" 버전으로 수동 복구

## 참고 문서
- `/root/.local/PRMRAG/docs/MODEL_SEARCH_BIAS_ANALYSIS.md` - 원인 분석
- `/root/.local/PRMRAG/outputs/test_10q/` - 기존 프롬프트 결과 (100% Search 시도)
