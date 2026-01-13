# RPE vs Judge Label 비교 분석 보고서

**작성일**: 2025-01-13
**데이터셋**: HotpotQA Medium Level (1,043 trajectories, 2,837 steps)

---

## 1. Executive Summary

RPE(Relative Progress Estimation)와 Judge 라벨링 방식을 비교 분석한 결과, **Judge 라벨이 RPE보다 더 신뢰할 수 있는 Process Supervision을 제공**한다는 것을 발견했습니다.

| 방식 | Perfect Match 개수 | 정답률 |
|------|-------------------|--------|
| **Judge만 (All GOOD)** | 250 | **86.2%** |
| RPE만 (All ≥ 0.8) | 453 | 78.9% |
| Consensus (RPE == Judge) | 224 | - |

**핵심 발견**: Judge 라벨만 사용할 경우 더 높은 품질의 학습 데이터를 얻을 수 있습니다.

---

## 2. 배경

### 2.1 RPE (Relative Progress Estimation)
- **정의**: `RPE = MC(s_t) / MC(s_{t-1})`
- **방식**: Monte Carlo 롤아웃을 통해 각 스텝 후 정답 도달 확률 추정
- **라벨링**: RPE ≥ 0.8 → GOOD, RPE < 0.8 → BAD
- **특성**: Outcome Supervision (결과 기반)

### 2.2 Judge Model (QwQ-32B)
- **방식**: LLM이 각 스텝의 추론 과정을 평가
- **평가 기준**:
  - Hallucination 여부
  - 검색 쿼리의 적절성
  - 논리적 추론의 정확성
- **특성**: Process Supervision (과정 기반)

---

## 3. 정량적 분석

### 3.1 Step-level 라벨 분포

| RPE Label | Judge Label | 개수 | 비율 |
|-----------|-------------|------|------|
| GOOD | GOOD | 1,119 | 39.4% |
| GOOD | BAD | 920 | 32.4% |
| BAD | GOOD | 252 | 8.9% |
| BAD | BAD | 546 | 19.2% |
| **Total** | | **2,837** | 100% |

**불일치율**: 41.3% (920 + 252 = 1,172 steps)

### 3.2 Trajectory-level Perfect Match 비교

```
┌─────────────────────────────────────────────────────────────┐
│                    Perfect Match 분석                        │
├─────────────────────┬──────────┬──────────┬─────────────────┤
│ 필터링 방식         │ 총 개수  │ 정답 개수 │ 정답률          │
├─────────────────────┼──────────┼──────────┼─────────────────┤
│ Judge만 (All GOOD)  │ 290      │ 250      │ 86.2%           │
│ RPE만 (All ≥ 0.8)   │ 574      │ 453      │ 78.9%           │
│ Consensus           │ 224      │ 224      │ (intersection)  │
└─────────────────────┴──────────┴──────────┴─────────────────┘
```

### 3.3 Venn Diagram 분석

```
        ┌─────────────────────────────────────┐
        │         RPE All GOOD (574)          │
        │    ┌─────────────────────┐          │
        │    │                     │          │
        │    │    Intersection     │  229개   │
        │    │       (224)         │ RPE only │
        │    │                     │          │
        │    └─────────────────────┘          │
        │              │                       │
        └──────────────┼───────────────────────┘
                       │
               ┌───────┴───────┐
               │    26개       │
               │ Judge only    │
               └───────────────┘

        Judge All GOOD (250) = 224 + 26
        RPE All GOOD (574) = 224 + 229 + 121 (incorrect)
```

---

## 4. 정성적 분석: 불일치 케이스 상세 검토

### 4.1 RPE=BAD, Judge=GOOD (252 cases)

**유형 1: 적절한 검색이지만 결과 실패**

```
Question: "What nationality was James Henry Miller's wife?"
Gold Answer: "American"

Step 2:
- Action: Search[query="nationality of James Henry Miller's wife"]
- Retrieved: 관련 없는 문서들
- RPE: 0.000 (mc_after가 0으로 급락)
- Judge: GOOD ("검색 쿼리는 적절했으나 문서에 답이 없음")
```

**분석**: Judge가 올바름. 검색 **행위**는 합리적이었으나 retrieval 시스템이 실패.

**유형 2: 관련 정보 획득했으나 MC 노이즈**

```
Question: "The Great Outdoors (1988) star's Hollywood Walk of Fame year?"
Gold Answer: "2006"

Step 2:
- Retrieved: The Great Outdoors 출연진 정보 (John Candy, Dan Aykroyd)
- RPE: 0.725 (threshold 0.8 미만)
- Judge: GOOD ("출연진 정보를 성공적으로 검색")
```

**분석**: RPE가 경계값(0.725 vs 0.8) 문제로 BAD 판정. 실제로는 유용한 정보 획득.

### 4.2 RPE=GOOD, Judge=BAD (920 cases) ⚠️ 중요

**유형 1: 증거 없이 정답 맞춤 (Hallucination)**

```
Question: "Human Error is the season finale of what network's show?"
Gold Answer: "Fox"

Step 2:
- Model Answer: "FOX"
- RPE: 0.881 (GOOD - MC 롤아웃도 같은 답 도출)
- Judge: BAD ("FOX는 hallucination, 문서에 증거 없음")
```

**분석**: 모델이 **운 좋게 정답**을 맞췄지만 추론 과정이 잘못됨.
- RPE는 결과만 보고 GOOD 판정
- Judge는 추론 과정의 오류를 감지

**유형 2: 잘못된 추론으로 정답 도달**

```
Question: "Which tennis player won more Grand Slam titles?"
Gold Answer: "Jonathan Stark"

Step 3-4:
- Model: "Leconte가 1988 프랑스 오픈 결승 진출" → "우승했다"로 잘못 추론
- RPE: 1.261 (GOOD - MC 상승)
- Judge: BAD ("결승 진출을 우승으로 잘못 해석")
```

**분석**: MC 롤아웃도 같은 오류를 범해 RPE가 높게 나옴. Judge만 논리적 오류 감지.

---

## 5. RPE의 한계점

### 5.1 Outcome vs Process Supervision

| 관점 | RPE | Judge |
|------|-----|-------|
| 평가 대상 | 결과 (정답 도달 여부) | 과정 (추론의 논리성) |
| Hallucination 감지 | ❌ 불가능 | ✅ 가능 |
| 운 좋은 정답 필터링 | ❌ 불가능 | ✅ 가능 |
| 적절한 실패 인정 | ❌ 불가능 | ✅ 가능 |

### 5.2 MC 롤아웃의 문제점

1. **노이즈**: 롤아웃 샘플링에 따른 분산
2. **경계값 민감도**: 0.79 vs 0.81 차이로 라벨 뒤집힘
3. **오류 전파**: 롤아웃도 같은 hallucination을 하면 감지 불가
4. **계산 비용**: 각 스텝마다 N번의 롤아웃 필요

### 5.3 Threshold 변경 실험

| Threshold | Consensus Trajectories | 변화 |
|-----------|----------------------|------|
| 0.8 | 390 | baseline |
| 0.7 | 389 | -1 |
| 0.6 | 387 | -3 |
| 0.5 | 386 | -4 |

**결론**: Threshold를 낮춰도 consensus 개수가 거의 변하지 않음.
불일치의 원인은 threshold가 아닌 **근본적인 평가 관점 차이**.

---

## 6. 권장사항

### 6.1 학습 데이터 구성

**Option A: Judge Only (권장)**
- 250개 trajectories (All GOOD + Correct)
- 정답률 86.2%
- 장점: 높은 품질, hallucination 필터링
- 단점: 데이터 양 적음

**Option B: RPE Only**
- 453개 trajectories
- 정답률 78.9%
- 장점: 데이터 양 많음
- 단점: hallucination 포함 가능

**Option C: Hybrid**
- Judge로 필터링 + RPE로 보조
- 또는 Judge confidence가 낮은 경우만 RPE 참조

### 6.2 시스템 개선

1. **RPE 제거 고려**:
   - 계산 비용 절감 (롤아웃 불필요)
   - Judge만으로 충분한 품질 확보 가능

2. **Judge 프롬프트 강화**:
   - 문서 인용 필수화
   - Confidence score 활용

3. **데이터 증강**:
   - Judge All GOOD 기준으로 더 많은 데이터 생성
   - 다양한 질문 유형 커버

---

## 7. 결론

RPE와 Judge 라벨링 방식을 비교한 결과:

1. **Judge가 더 신뢰할 수 있음**: 86.2% vs 78.9% 정답률
2. **RPE의 근본적 한계**: Outcome만 보고 Process 오류 감지 불가
3. **Hallucination 문제**: RPE는 운 좋게 맞춘 경우 필터링 불가
4. **실용적 권장**: RPE 제거하고 Judge만 사용 → 속도 향상 + 품질 유지

---

## Appendix: 코드 변경사항

### A.1 새로운 브랜치 구조

```
master ─────┬──── rpe (RPE 코드 보존)
            │
            └──── no-rpe (SimpleTrajectoryGenerator)
```

### A.2 SimpleTrajectoryGenerator

- MC 롤아웃 제거
- Action 기반 흐름 (Search/Finish)
- Backtracking 없음
- Judge 라벨링에 의존

```python
from prmrag.generation import SimpleTrajectoryGenerator

generator = SimpleTrajectoryGenerator(
    policy_model=model,
    retriever=retriever,
    config={'max_steps': 10, 'top_k_passages': 5}
)

trajectory = generator.generate_trajectory(question, gold_answer)
```
