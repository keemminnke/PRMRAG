# 모델 검색 편향 분석

## 핵심 발견

**10q 테스트 데이터 분석 결과:**
- 총 스텝 수: 100개
- 모델이 "Action: Search" 시도: **100개 (100%)**
- 실제 검색 실행: 18개 (18%)
- 검색 막힌 횟수: 82회

**결론: 모델은 모든 스텝에서 검색을 시도합니다.**

---

## 원인 분석

### 1. 시스템 프롬프트 구조 (policy_model_vllm.py:245-339)

프롬프트는 두 가지 액션만 제공합니다:

```
# AVAILABLE ACTIONS

1. **Search** - Retrieve information from the knowledge base
   - Use when you need external information
   - Format: Action: Search[query="your sub-question"]

2. **Finish** - Provide the final answer
   - Use when you can answer the main question
   - Format: Action: Finish[answer="direct answer"]
```

**문제점:**
- 순수 추론(CoT) 액션이 없음
- "Search" 또는 "Finish" 중 선택해야 함
- 중간 추론 단계에서는 Finish를 할 수 없으므로 → **무조건 Search를 선택**

### 2. 프롬프트 예시 구조

```
**When you need information (Search):**
Step N:
Thought: [Why you need this information]
Action: Search[query="specific sub-question"]
Observation: [Cite documents using [N] format]
Sub-answer: [Extract answer from observation]

**When you have the answer (Finish):**
Step N:
Thought: [Final reasoning]
Final Answer: [direct, concise answer]
```

**문제점:**
- Search는 Thought → Action → Observation → Sub-answer 전체 flow 제공
- Finish는 마지막 단계에만 사용
- **중간 추론을 위한 format이 없음**

### 3. 실제 데이터 예시

```
Step 1:
  action: Reason
  num_passages: 0
  RPE: 0.743
  Content:
    Thought: To determine which magazine was started first,
             I need to know the founding year of both magazines.
    Action: Search[query="founding year of Arthur's Magazine"]
```

모델은 논리적으로 행동:
1. 외부 정보가 필요함을 인식
2. 프롬프트에서 "Search" 또는 "Finish" 중 선택
3. 아직 답을 모르므로 Finish 불가
4. → **Search를 작성**

---

## Adaptive Generator의 개입

### adaptive_generator.py:440-462

```python
if rpe_cot >= 0.8:
    # (B) CoT good enough (RPE >= 0.8), accept it
    step = Step(
        thought=cot_step_content,  # "Action: Search..." 포함
        action="Reason",           # CoT로 분류
        action_input=None,
        num_passages=0,
        ...
    )
```

**동작 방식:**
1. 모델이 "Action: Search[...]" 작성
2. Adaptive generator가 RPE 계산
3. RPE >= 0.8이면 → 검색 실행 안함, action='Reason'으로 저장
4. RPE < 0.8이면 → 검색 실행, action='Search'로 저장

**결과:**
- content에는 "Action: Search" 텍스트가 남아있음
- 하지만 action='Reason', num_passages=0

---

## 설계 철학 문제

### 현재 설계의 모순

**모델의 관점:**
- "외부 정보가 필요하다" → Search 작성
- 프롬프트가 이렇게 하라고 지시함

**Adaptive Generator의 관점:**
- "RPE가 높으면 CoT만으로 충분하다" → 검색 무시
- 모델의 판단을 override

### 질문 1: 이것이 문제인가?

**문제일 수 있는 이유:**
- 모델과 시스템의 판단 불일치
- 모델: "검색 필요" vs 시스템: "검색 불필요"
- 혼란스러운 training signal

**문제가 아닐 수 있는 이유:**
- Adaptive Generator는 MC 기반 객관적 지표 사용
- 모델은 자신의 불확실성을 정확히 판단 못할 수 있음
- RPE >= 0.8 = "검색 없이도 정답 도달 가능"

### 질문 2: 프롬프트를 바꿔야 하나?

**옵션 A: 순수 추론 액션 추가**

```
# AVAILABLE ACTIONS

1. **Reason** - Continue reasoning without external information
   - Use when you can make progress with your own knowledge
   - Format: Just write your reasoning

2. **Search** - Retrieve information from the knowledge base
   - Use when you need external information
   - Format: Action: Search[query="..."]

3. **Finish** - Provide the final answer
   - Format: Action: Finish[answer="..."]
```

**장점:**
- 모델이 명시적으로 "Reason"을 선택할 수 있음
- content와 action 필드 일치

**단점:**
- 모델이 자신의 지식을 과신할 수 있음 (hallucination)
- RPE 기반 adaptive intervention의 효과 감소
- HotpotQA는 대부분 외부 지식 필요 (모델이 항상 Search 선택할 가능성)

**옵션 B: 현재 유지 + 데이터 후처리**

- 프롬프트는 유지
- Adaptive generator가 개입한 경우 content 정리
- "Action: Search" 텍스트를 제거하고 순수 Thought만 남김

**장점:**
- Adaptive intervention 효과 유지
- 데이터 일관성 확보

**단점:**
- 모델 출력 수정 필요
- 원본 의도 손실

---

## 통계적 근거

### 100% Search 시도의 의미

**HotpotQA 특성:**
- Multi-hop question answering
- 대부분 외부 지식 필요
- "Which magazine was started first?" → 창간 연도 정보 필요

**모델의 합리적 판단:**
- 외부 정보 없이는 답할 수 없음을 인식
- Search가 필요하다는 판단은 **정확함**

**Adaptive Generator의 역할:**
- 모델이 이미 충분한 정보를 가진 상태인지 판단 (MC 기반)
- RPE >= 0.8 → "추가 검색 불필요, 기존 정보로 답 도출 가능"

---

## 권장 사항

### 현재 설계 유지

**이유:**
1. **모델 판단이 실제로 정확함**
   - HotpotQA는 대부분 검색 필요
   - 100% Search 시도는 task 특성상 자연스러움

2. **Adaptive Generator가 효과적으로 작동**
   - 82% 케이스에서 불필요한 검색 방지
   - RPE 기반 객관적 판단

3. **데이터 품질 관점**
   - content 필드: 모델의 원본 의도 (PRM training에 유용)
   - action 필드: 실제 실행된 액션 (정확함)

### 개선 사항 (선택)

**1. Validation script 강화**
```python
# CoT step의 content에 "Action: Search"가 있는 경우
# 이것이 RPE >= 0.8으로 막힌 검색인지 확인
if action == 'Reason' and 'Action: Search' in content:
    # 이것은 정상 - adaptive generator가 막은 것
    pass
```

**2. 문서화 강화**
- 현재 동작을 명확히 문서화
- content vs action 차이 설명
- PRM training 시 이를 고려

**3. 프롬프트 미세 조정 (필요시)**
```
# 현재 프롬프트에 추가:
Note: The system will determine whether to execute the search
based on current problem-solving progress. Your role is to
indicate when external information would be helpful.
```

---

## 결론

**"일단 왜 모델이 무조건 검색을 매 스텝마다 하려고해?"**

**답변:**
1. **프롬프트가 Search와 Finish만 제공** - 중간 추론 액션 없음
2. **HotpotQA 특성** - 대부분 외부 정보 필요
3. **모델의 합리적 판단** - "외부 정보 필요" → Search 선택
4. **Adaptive Generator의 개입** - RPE >= 0.8이면 검색 실행 안함

**이것이 문제인가?**
- **No.** 이것은 의도된 설계입니다.
- 모델: "검색이 도움될 것" 제안
- Adaptive Generator: MC 기반으로 실제 실행 여부 결정

**개선이 필요한가?**
- 현재 설계는 작동함 (82% 불필요한 검색 방지)
- 데이터 일관성을 위한 문서화/validation 강화 권장
- 프롬프트 구조 변경은 신중히 고려 (trade-off 존재)
