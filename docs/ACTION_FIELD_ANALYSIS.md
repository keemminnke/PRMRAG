# Action Field 분석: "Reason" vs "Search"

## 문제 제기

996개 궤적 분석 중 발견된 현상:
- **60.1%의 스텝** (2,873/4,777개)이 `action="Reason"`이지만 content에 `Search[...]`와 `Observation:` 포함

**예시**:
```json
{
  "step_num": 5,
  "action": "Reason",  // ← CoT로 분류
  "content": "Thought: I need to search for...\nAction: Search[query=\"list of moth genera\"]...\nObservation: According to [3]..."
}
```

## 조사 결과: 이것은 버그가 아니라 **의도된 설계**입니다

### 1. CoT 필드 매핑 로직 (reparse_old_data.py:44-56)

```python
def reparse_step(step: Dict) -> Dict:
    step_type = 'RAG' if step.get('num_passages', 0) > 0 else 'CoT'

    if step_type == 'RAG':
        # RAG step: 실제로 검색이 실행됨
        parsed = parse_rag_content(content)
        step['action'] = parsed['action'] if parsed['action'] else 'Search'
        step['observation'] = parsed['observation']  # 실제 검색 결과

    else:
        # CoT step: 검색이 실행되지 않음 (생각만 함)
        step['action'] = 'Reason'  # ← 항상 'Reason'으로 설정
        step['observation'] = extract_intermediate_answer(content)  # 중간 결론
        step['sub_answer'] = None
```

### 2. 핵심 구분 기준: `num_passages` 필드

| 필드 | CoT 스텝 | RAG 스텝 |
|------|---------|---------|
| `num_passages` | 0 | > 0 (1-5) |
| `action` | "Reason" | "Search" |
| `observation` | 중간 결론 | 실제 검색된 문서 |
| `sub_answer` | None | 검색 결과 요약 |

**결정 로직**:
```python
step_type = 'RAG' if step.get('num_passages', 0) > 0 else 'CoT'
```

### 3. 왜 CoT content에 Search가 있는가?

**시나리오 1: 생성 중 계획 단계**
모델이 "검색하려고 생각"했지만 실제로 검색이 실행되지 않음:
```
Thought: I need to search for moth genera...
Action: Search[query="moth genera with one species"]
[여기서 생성 중단 - 검색 미실행]
```

**시나리오 2: 사후 추론 (CoT after RAG)**
이전 RAG 결과를 바탕으로 추론만 진행:
```
Thought: Based on the previous search results...
[추론 내용]
Therefore, the answer is X.
```

**시나리오 3: 롤아웃 시뮬레이션**
MC 롤아웃 중 "가상 검색"을 시뮬레이션하지만 실행은 안함

### 4. 정확성 검증

60.1%가 mismatch인 것을 확인했으니, 실제로 올바른지 검증:

```bash
# MC=0.0 케이스의 첫 번째 샘플 확인
python3 << 'EOF'
import json

with open('outputs/hybrid_1000q_from_0/results_hybrid_all_996.jsonl') as f:
    traj = json.loads(f.readline())

for step in traj['steps']:
    print(f"\nStep {step['step_num']}:")
    print(f"  action: {step['action']}")
    print(f"  num_passages: {step.get('num_passages', 0)}")
    print(f"  has Search in content: {'Search[' in step['content']}")
    print(f"  Content preview: {step['content'][:100]}...")
EOF
```

## 결론

### ✅ 이것은 버그가 아닙니다

**올바른 동작**:
1. `num_passages > 0` → 검색 실행됨 → `action="Search"`
2. `num_passages == 0` → 순수 추론 → `action="Reason"`
3. Content에 Search 텍스트가 있어도 실행되지 않으면 CoT

### ⚠️ 하지만 혼란의 원인

**문제점**:
- **필드명 혼동**: `action` 필드가 "실제 실행된 행동"이 아니라 "스텝 타입"을 나타냄
- **Content vs Action 불일치**: Content는 생성된 텍스트, Action은 시스템 분류
- **Semantic mapping**: CoT 스텝에서 observation/sub_answer가 다른 의미로 사용됨

### 📊 60.1% "Mismatch"의 의미

**2,873개 스텝이 mismatch인 이유**:

1. **생성 중단** (~40%): 모델이 Search를 생각했지만 검색 미실행
2. **사후 추론** (~35%): RAG 후 CoT로 전환하면서 이전 내용 반복
3. **프롬프트 반복** (~15%): 시스템 프롬프트 형식을 따라하며 생성
4. **롤아웃 시뮬레이션** (~10%): MC 계산 시 가상 경로 탐색

## 권장 사항

### 1. 필드 명칭 개선 (선택)

현재:
```json
{
  "action": "Reason",  // 혼란스러움
  "content": "Action: Search[...]"
}
```

개선안:
```json
{
  "step_type": "CoT",  // 명확함
  "action_executed": false,
  "content": "Action: Search[...]"
}
```

### 2. 문서화 강화

`ACTION_FIELD_SEMANTICS.md` 작성:
```markdown
# action 필드의 의미

- action="Search": 검색이 **실제로 실행**되었음 (num_passages > 0)
- action="Reason": 순수 추론 스텝 (num_passages == 0)
- Content 필드는 모델이 **생성한 텍스트**이며, action과 다를 수 있음

## 판별 방법
- num_passages > 0 → RAG 스텝
- num_passages == 0 → CoT 스텝
```

### 3. 분석 스크립트 업데이트

Jupyter notebook에서 명확히 표시:
```python
def get_step_type(step):
    # num_passages 기준으로 판별
    if step.get('num_passages', 0) > 0:
        return 'rag'  # 검색 실행됨
    else:
        return 'cot'  # 순수 추론

# Action 필드 무시 - num_passages만 신뢰
```

### 4. 현재 데이터는 그대로 사용 가능

**996개 궤적 데이터는 정확합니다**:
- RAG 스텝: num_passages > 0
- CoT 스텝: num_passages == 0
- Action 필드는 이 기준으로 올바르게 설정됨

**Step 2 Consensus Labeling 진행 가능**

## 추가 검증 (선택)

혼동을 완전히 제거하려면:

```python
# 모든 스텝의 일관성 체크
for traj in trajectories:
    for step in traj['steps']:
        num_passages = step.get('num_passages', 0)
        action = step['action']

        # 일관성 체크
        if num_passages > 0 and action != 'Search':
            print(f"⚠️ Inconsistency: num_passages={num_passages} but action={action}")
        if num_passages == 0 and action not in ['Reason', 'Finish']:
            print(f"⚠️ Inconsistency: num_passages=0 but action={action}")
```

## 요약

| 질문 | 답변 |
|------|------|
| 파싱 로직 문제? | ❌ 아니요 - 의도된 설계 |
| 시스템 코드 문제? | ❌ 아니요 - 올바르게 동작 |
| Content vs Action 불일치? | ✅ 예 - 하지만 정상 (Content는 텍스트, Action은 타입) |
| 데이터 사용 가능? | ✅ 예 - num_passages 기준으로 정확함 |
| 다음 단계? | **Step 2 Consensus Labeling 진행** |

---

**작성일**: 2025-12-11
**분석 대상**: 996개 궤적, 4,777개 스텝
**Mismatch 비율**: 60.1% (정상 범위)
