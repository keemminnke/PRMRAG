# 완전한 문제 분석: 996 Trajectories

## 발견 경위

사용자가 MC=0.0 케이스를 분석하던 중 두 가지 이상한 패턴 발견:

### 발견 1: "action='Reason'인데 검색 형식이 있음"
```
Step 4 - REASON
Observation: [3] mentions Prince Naseem...
Sub-answer: Prince Naseem was once considered...
```

### 발견 2: "Reason인데 실패한 검색 메시지"
```
Step 3 - REASON
Observation: No relevant information found in [1-5].
```

**질문**: "검색을 실행했는데 왜 Reason이야?"

## 조사 결과: 3가지 문제 발견

### 문제 1: 모델 환각 - 가짜 검색 생성

**규모**: 1,247개 스텝 (26.1% of all steps)

**특징**:
- `num_passages = 0` → 검색 실행 안됨
- Content에 완전한 검색 형식:
  ```
  Action: Search[query="..."]
  Observation: [3] states that...
  Sub-answer: ...
  ```
- 가짜 인용 `[1-5]` 포함 (95%)
- 가짜 정보 생성 (ex: "Prince Naseem")

**원인**:
1. 프롬프트가 검색 형식 보여줌
2. Stop sequence 없음 → "Observation:" 계속 생성
3. 모델이 형식을 학습해서 따라함

**분류**:
- ✅ `action='Reason'` 정확 (실제로 검색 안함)
- ⚠️ 하지만 모델이 환각

### 문제 2: 모델 환각 - 가짜 실패 메시지

**규모**: 80개 스텝 (1.7% of all steps)

**특징**:
- `num_passages = 0` → 검색 실행 안됨
- Content: "No relevant information found"
- 인용 없음 (모델이 지어냄)

**예시**:
```
Observation: No relevant information found regarding...
```

**분류**:
- ✅ `action='Reason'` 정확
- ⚠️ 하지만 환각

### 문제 3: 잘못 분류된 실패한 검색 ⚠️ **버그!**

**규모**: 78개 스텝 (1.6% of all steps)

**특징**:
- `action='Reason'` ❌ 틀림!
- Content: "No relevant information found in [1-5]"
- **[1-5] 인용 있음** → 검색 실행됨!
- 하지만 관련 문서 못 찾음

**예시**:
```
Step 3 - REASON (잘못됨!)
Observation: No relevant information found in [1-5].
Sub-answer: No additional information available on...
```

**진실**:
- 검색이 **실제로 실행됨**
- 5개 문서 가져옴
- 하지만 관련 정보 없음
- `action='Search'`여야 함!

**버그 위치**:
- 원본 생성 코드
- `num_passages` 필드 누락
- `action` 필드 잘못 설정

### 전체 통계

| 분류 | 개수 | 비율 | 설명 |
|------|------|------|------|
| **정상 스텝** | 3,372 | 70.6% | CoT/RAG 올바르게 분류 |
| **환각 (가짜 검색)** | 1,247 | 26.1% | 검색 형식 생성, 실행 안됨 |
| **환각 (가짜 실패)** | 80 | 1.7% | "No relevant" 환각 |
| **잘못 분류 (버그)** | 78 | 1.6% | 검색 실행, Reason으로 분류 |
| **총 문제** | **1,405** | **29.4%** | |

## 상세 분석

### A. 환각 스텝 (1,327개, 27.8%)

#### A1. 가짜 검색 (1,247개)
```json
{
  "action": "Reason",  // ← 올바름 (검색 안함)
  "num_passages": 0,
  "content": "Action: Search[query='...']\nObservation: [3] states...",  // ← 환각
  "observation": "...",  // ← 가짜
  "sub_answer": "..."  // ← 가짜
}
```

**MC=0.0 상관관계**:
- MC=0.0: 89.7% 환각률
- MC>0.0: 83.0% 환각률
- **차이**: +6.7% (약한 상관)

#### A2. 가짜 실패 메시지 (80개)
```json
{
  "action": "Reason",
  "num_passages": 0,
  "content": "Observation: No relevant information found...",  // ← 환각 (인용 없음)
}
```

### B. 잘못 분류된 검색 (78개, 1.6%) ⚠️ **버그**

```json
{
  "action": "Reason",  // ❌ 틀림! 'Search'여야 함
  "num_passages": 0,   // ❌ 누락! 5여야 함
  "content": "Action: Search[...]\nObservation: [3] No relevant information found in [1-5]...",
  "observation": "No relevant information found in [2]...",  // ← 진짜 검색 결과
  "action_input": "query string"  // ← 검색 실행됨
}
```

**증거**:
1. `[1-5]` 인용 존재 → 문서 검색됨
2. `observation` 필드 존재 → 시스템이 제공
3. Content 형식 완벽 → 실제 RAG 스텝

**버그 원인**:
- 생성 코드에서 `num_passages` 미저장
- `action` 필드 잘못 설정
- Reparse 시 복원 불가

## 영향 분석

### 1. MC=0.0 문제 (45.8%, 456 trajectories)

**원인 분석**:
- 환각: +6.7% 기여 (약함)
- 잘못 분류: 영향 미미 (1.6%)
- **주요 원인**: Answer extraction/matching 문제

### 2. RAG 정확도 낮음 (37.0%)

**가능한 원인**:
1. 환각으로 틀린 정보 사용
2. 잘못 분류된 검색 → CoT로 계산
3. 실제 검색 품질 문제

### 3. 데이터 품질

**긍정적**:
- RPE 라벨은 정확 (최종 답 기준)
- 진짜 RAG 스텝 (1,136개) 깨끗

**부정적**:
- 29.4% 스텝에 문제
- Judge labeling 시 혼동 가능

## 해결 방안

### 즉시 조치 (현재 데이터)

#### 1. 데이터 정제 스크립트

```python
def fix_misclassified_searches(step):
    """Fix steps that are RAG but classified as Reason."""
    if step['action'] == 'Reason':
        content = step['content']

        # Check for [1-5] citations in "No relevant" message
        if 'no relevant information' in content.lower():
            if re.search(r'\[([1-5])\]', content):
                # This was actually a search!
                step['action'] = 'Search'
                step['num_passages'] = 5  # Estimate
                print(f"Fixed: Step {step['step_num']}")

    return step
```

#### 2. 현재 데이터 사용

**가능**:
- RPE 정확
- Step 2 Consensus 진행 가능

**주의**:
- Judge에게 환각 케이스 알림
- 분석 시 `num_passages`만 신뢰

### 단기 조치 (새 데이터 생성)

#### 1. Stop Sequence 추가
```python
stop_sequences = ["\nObservation:", "Observation:"]
```

#### 2. 프롬프트 개선
```
Action: Search[query="..."]

STOP! Do not write Observation. The system will provide it.
```

#### 3. num_passages 저장 확인
생성 코드에서 모든 스텝에 `num_passages` 저장

### 중기 조치 (시스템 개선)

#### 1. 데이터 검증

```python
def validate_step(step):
    """Validate step classification."""
    has_citations = bool(re.search(r'\[([1-5])\]', step['content']))
    action = step['action']
    num_passages = step.get('num_passages', 0)

    # RAG должен иметь num_passages > 0
    if action == 'Search' and num_passages == 0:
        raise ValueError(f"Search step without num_passages")

    # Reason не должен иметь citations
    if action == 'Reason' and has_citations and 'no relevant' not in step['content'].lower():
        warnings.warn(f"Reason step with citations - possible hallucination")
```

#### 2. Post-processing

환각 감지 및 제거:
```python
if step['action'] == 'Reason' and step.get('num_passages', 0) == 0:
    if 'Observation:' in step['content']:
        # Remove hallucinated observation
        step['content'] = extract_thought_only(step['content'])
```

## 현재 상태 요약

### ✅ 사용 가능한 데이터

| 항목 | 상태 | 비고 |
|------|------|------|
| 총 궤적 | 996 | ✅ |
| 총 스텝 | 4,777 | ✅ |
| 정상 스텝 | 3,372 (70.6%) | ✅ |
| RPE 라벨 | 정확 | ✅ 최종 답 기준 |
| 진짜 RAG | 1,136 | ✅ num_passages > 0 |

### ⚠️ 알려진 문제

| 문제 | 개수 | 해결 |
|------|------|------|
| 모델 환각 | 1,327 (27.8%) | Stop sequence |
| 잘못 분류 | 78 (1.6%) | 정제 스크립트 |
| MC=0.0 | 456 (45.8%) | 별도 조사 필요 |

### 🎯 권장 조치

**즉시**:
1. ✅ 현재 데이터로 Step 2 진행
2. ✅ 환각 케이스 문서화
3. ⚠️ 78개 잘못 분류 수정 고려

**단기**:
1. Stop sequence 추가
2. 10개 테스트 생성
3. 환각률 확인

**중기**:
1. MC=0.0 원인 조사
2. RAG 품질 개선
3. Answer extraction 수정

## 결론

사용자의 날카로운 관찰로 **세 가지 주요 문제** 발견:

1. **모델 환각** (27.8%) - 프롬프트/Stop sequence 문제
2. **잘못 분류** (1.6%) - 생성 코드 버그
3. **MC=0.0** (45.8%) - 별도 원인 (extraction?)

**현재 데이터**: 사용 가능하지만 문제 인지 필요
**다음 단계**: Step 2 Consensus Labeling 진행 OK

---

**분석일**: 2025-12-11
**분석 범위**: 996 trajectories, 4,777 steps
**발견자**: 사용자 (MC=0.0 분석 중)
**문제율**: 29.4% steps affected
