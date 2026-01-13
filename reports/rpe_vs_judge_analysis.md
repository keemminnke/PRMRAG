# RPE vs Judge Label 비교 분석 보고서

**작성일**: 2025-01-13
**데이터셋**: HotpotQA Medium Level (1,043 trajectories, 2,837 steps)

---

## 1. Executive Summary

RPE(Relative Progress Estimation)와 Judge 라벨링 방식을 비교 분석한 결과, **Consensus Filtering (RPE & Judge 둘 다 GOOD)이 가장 높은 정확도**를 보여줍니다.

| 방식 | Perfect Match 개수 | 정답률 |
|------|-------------------|--------|
| **Consensus (RPE & Judge 둘다 GOOD)** | 242 | **93.0%** |
| Judge만 (All GOOD) | 290 | 89.3% |
| RPE만 (All ≥ 0.8) | 574 | 79.6% |

**핵심 발견**: Consensus Filtering이 가장 높은 품질(93.0%)을 제공하지만 데이터 양이 적음. 품질-양 트레이드오프 고려 필요.

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
│ Consensus (둘다GOOD)│ 242      │ 225      │ 93.0% ⭐        │
│ Judge만 (All GOOD)  │ 290      │ 259      │ 89.3%           │
│ RPE만 (All ≥ 0.8)   │ 574      │ 457      │ 79.6%           │
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

**Option A: Consensus Filtering (권장 - 최고 품질)**
- 242개 trajectories (RPE & Judge 둘 다 GOOD)
- 정답률 **93.0%**
- 장점: 가장 높은 품질, hallucination + 운 좋은 정답 둘 다 필터링
- 단점: 데이터 양 가장 적음

**Option B: Judge Only (품질-양 균형)**
- 290개 trajectories (All GOOD)
- 정답률 89.3%
- 장점: 적당한 품질, hallucination 필터링
- 단점: 일부 운 좋은 정답 포함 가능

**Option C: RPE Only (양 우선)**
- 574개 trajectories
- 정답률 79.6%
- 장점: 데이터 양 많음
- 단점: hallucination, 잘못된 추론 포함

### 6.2 시스템 개선

1. **Consensus Filtering 활용**:
   - RPE와 Judge를 함께 사용하여 최고 품질(93.0%) 달성
   - RPE는 계산 비용이 있지만, 품질 향상에 기여

2. **속도 vs 품질 트레이드오프**:
   - 빠른 생성 필요 시: RPE 제거, Judge Only (89.3%)
   - 최고 품질 필요 시: Consensus Filtering 유지 (93.0%)

3. **데이터 증강**:
   - Consensus 기준으로 더 많은 데이터 생성
   - 다양한 질문 유형 커버하여 242개 → 더 많이 확보

---

## 7. 결론

RPE와 Judge 라벨링 방식을 비교한 결과:

1. **Consensus Filtering이 최고 품질**: 93.0% (RPE & Judge 둘 다 GOOD)
2. **RPE와 Judge의 상호보완 효과**:
   - RPE: Outcome 기반 → 결과적으로 맞는지 확인
   - Judge: Process 기반 → 추론 과정이 올바른지 확인
   - 둘 다 GOOD이면 "올바른 과정으로 올바른 결과" 도달
3. **단독 사용 시 한계**:
   - RPE만: 79.6% (hallucination 필터링 불가)
   - Judge만: 89.3% (운 좋은 정답 일부 포함)
4. **실용적 권장**:
   - 최고 품질 필요 시 → Consensus Filtering (93.0%)
   - 데이터 양 필요 시 → Judge Only (89.3%) 또는 RPE Only (79.6%)

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
