# Eval 분석 결과 (2026-03-12)

## Eval 결과 비교표 (전체)

### Agentic RAG (SearchR1Pipeline, temp=0)
| Model | Tag | EM | F1 | Sub_EM | Gap |
|-------|-----|-----|-----|--------|-----|
| Qwen2.5-7B-Instruct (base) | qwen_base_t0 | 3.35% | 14.69% | 44.29% | +40.95% |
| ReasonRAG LoRA (HF) | reasonrag_t0 | 20.62% | 29.01% | 27.86% | +7.24% |
| Our DPO V1 (MCTS, old prompt) t=0.8 | dpo_mcts_v1 | 20.74% | 35.39% | 47.16% | +26.42% |
| Our DPO V1 (MCTS, old prompt) t=0 | dpo_mcts_t0 | 23.07% | 38.06% | 48.41% | +25.35% |
| Our DPO V2 (cleaned) t=0 | dpo_v2_cleaned_t0 | 32.79% | 43.17% | 37.08% | +4.29% |
| Our Step-DPO V1 (base) | step_dpo_v1_base | 33.71% | 44.83% | - | - |
| Our DPO V2 (3000q, ep1 matched) | dpo_v2_ep1_matched | 34.15% | 45.53% | - | - |
| Our DPO V2 (3000q, full) | dpo_v2_3000q_full | 34.38% | 45.85% | - | - |
| Our SFT V1 | sft_v1_eval | 34.92% | 45.97% | - | - |
| Our SFT V2 (3000q) t=0.8 | sft_v2_3000q | 34.54% | 45.68% | - | - |
| Our SFT V2 t=0 | sft_v2_t0 | 36.27% | 47.65% | 41.92% | +5.65% |
| Search-R1 PPO (official) | searchr1_ppo_t0 | 37.93% | 49.48% | 44.11% | +6.18% |
| **Our DPO V1 + new prompt** t=0 | dpo_v1_newprompt_t0 | **38.06%** | **50.36%** | 43.71% | +5.65% |
| Our DPO V3 (hybrid) t=0 | dpo_v3_newprompt_t0 | 32.52% | 43.31% | - | - |

### Standard RAG (SequentialPipeline, temp=0)
| Model | Tag | EM | F1 |
|-------|-----|-----|-----|
| Qwen2.5-7B-Instruct (base) | standard_base_t0 | 31.69% | 41.96% |

### 논문 보고 수치 (참고)
| Model | EM | 비고 |
|-------|-----|------|
| ReasonRAG (논문) | 38.4% | 재현 불가 |
| Qwen Standard RAG (논문) | 29.3% | 우리가 더 높음 (31.69%) |

## 핵심 발견

1. **우리 retriever setup은 문제 없음**
   - Standard RAG: 우리 31.69% vs 논문 29.3% → 우리가 더 높음
   - BGE IVF4096 + nprobe=128 + FAISS GPU = OK

2. **우리 모델 > ReasonRAG LoRA** (같은 환경)
   - Our DPO EM=23.07% vs ReasonRAG EM=20.62%
   - 모델 학습 자체는 문제 아님

3. **논문 38.4% 재현 불가**
   - ReasonRAG LoRA를 우리 setup에서 돌려도 20.62%밖에 안 나옴
   - 논문의 eval 조건이 다른 것으로 추정 (corpus augmentation 등)

4. **프롬프트 수정으로 EM 대폭 상승**
   - DPO V1 old prompt: 23.07% → **new prompt: 38.06%** (+15%)
   - 원인: 답변 장황함 해결 (Sub_EM gap 25% → 5.65%)
   - **Search-R1 PPO (37.93%) 돌파**

5. **V2 cleaned 데이터는 검색 횟수 감소 문제**
   - V1 평균 검색 2.7회 vs V2 2.0회 (57%가 1회만 검색)
   - Cross-type 전부 제거 → "검색하라" 신호 소실

## 답변 장황함 (Verbose Answer) 분석

### 우리 DPO 데이터 vs ReasonRAG 데이터

| 지표 | ReasonRAG | 우리 |
|------|-----------|------|
| Chosen 답변 평균 단어수 | **4.4** | 14.6 |
| ≤5단어 비율 | 79.9% | 35.6% |
| Chosen이 Rejected보다 짧은 비율 | 79.8% | 67.6% |

### 근본 원인 5가지

1. **프롬프트 모호함**: "concise" vs ReasonRAG의 "nouns or short phrases whenever possible"
2. **"So the answer is" 패턴 부재**: ReasonRAG는 이 접두사로 키워드만 유도
3. **DPO chosen에 document 포함**: Search 타입 chosen의 94%가 문서 텍스트 → 학습 신호 오염
4. **역할 분리 없음**: 우리는 1 프롬프트, ReasonRAG는 3 프롬프트 (but 이건 우리 design choice)
5. **선택 기준에 간결성 미반영**: F1+critic만 → 이미 구현됨 (combined_reward)

### 기존 데이터 복구 가능 여부
1
Answer-type 4,275 pairs:
- 이미 짧음 (≤5단어): 1,521개 → 수정 불필요
- Gold가 substring (Cat1): 1,418개 → gold answer로 교체 가능
- F1<0.5 답 틀림 (Cat3): 1,331개 → 제거

## 수정 사항

### 프롬프트 수정 (적용 완료)
```
기존: "...directly provide a concise final answer using <answer> and </answer> without detailed illustrations."
수정: "...directly provide your final answer. Ensure your answer is concise, using nouns or short phrases whenever possible. Conclude with: "So the answer is <answer>answer</answer>"."
```

적용 파일:
- `scripts/build_mcts_dpo.py` SYSTEM_PROMPT
- `scripts/eval_flashrag.py` SYSTEM_PROMPT

### Sub_EM (Cover Exact Match) 분석

Gold answer가 prediction에 **포함**되면 정답 처리 (verbose해도 OK)

| Model | EM | Sub_EM | F1 | Gap (Sub-EM) |
|-------|-----|--------|-----|------|
| Standard RAG (base) | 31.69% | 36.53% | 41.96% | +4.83% |
| Qwen base (SearchR1) | 3.35% | **44.29%** | 14.94% | **+40.95%** |
| Our DPO V1 (t=0.8) | 20.75% | **47.16%** | 36.00% | **+26.42%** |
| Our DPO V1 (t=0) | 23.07% | **48.41%** | 38.68% | **+25.35%** |
| ReasonRAG LoRA | 20.62% | 27.86% | 29.08% | +7.24% |
| Our DPO V2 (cleaned) | 32.79% | 37.08% | 43.17% | +4.29% |
| Our SFT V2 t=0 | 36.27% | 41.92% | 47.65% | +5.65% |
| Search-R1 PPO | 37.93% | 44.11% | 49.48% | +6.18% |
| **Our DPO V1 + new prompt** | **38.06%** | **43.71%** | **50.36%** | **+5.65%** |

**결론**: 프롬프트 수정으로 Gap이 25% → 5.65%로 감소. 답변 포맷 문제 대부분 해결됨.

## DPO Pair 전략 문제점 및 개선

### 현재 전략: All Sibling Combinations
같은 부모 노드의 모든 자식 조합 비교 (nC2)

### 문제점 1: Cross-type 비교 (19.2%)

| Pair Type | 수량 | 비율 | 구분 |
|-----------|------|------|------|
| Search vs Search | 7,242 | 54.9% | SAME |
| Answer vs Answer | 3,410 | 25.9% | SAME |
| **Answer vs Search** | **2,382** | **18.1%** | **CROSS** |
| Answer vs Reason | 108 | 0.8% | CROSS |
| Reason vs Search | 36 | 0.3% | CROSS |
| Reason vs Reason | 5 | 0.0% | SAME |

- Cross-type 2,526개 (19.2%): Search와 Answer는 텍스트 형식이 완전히 다름
- DPO가 "검색할지 답변할지"를 토큰 레벨로 학습 → 노이즈
- **해결: Same-type 비교만 유지**

### 문제점 2: Low Margin (28.8%)
```
Combined Reward 차이 분포:
  Mean: 0.2483, Median: 0.2916
  <0.05: 2,391 (18.1%)  ← 사실상 노이즈
  <0.1:  3,801 (28.8%)
  >=0.2: 7,766 (58.9%)
```

- Margin이 너무 작으면 chosen/rejected 구분이 무의미
- **해결: margin >= 0.1 이상만 유지**

### 데이터셋 버전별 비교

| | V1 (원본) | V2 (cleaned) | V3 (hybrid) |
|---|---|---|---|
| 총 pairs | 13,183 | 7,192 | **8,724** |
| Same-type | 80.8% | 100% | 82.4% |
| Search-chosen cross | 12.2% | 0% | **17.6%** |
| Answer-chosen cross | 6.6% | 0% | 0% |
| Low margin (<0.1) | 28.8% | 0% | 0% |
| Verbose answer 교체 | 없음 | 있음 | 있음 |
| 새 프롬프트 | 없음 | 있음 | 있음 |

V3 전략:
1. V2 기반 (same-type + margin≥0.1) = 7,192
2. **Search-chosen cross-type 복원** (+1,532): "답변보다 검색이 나은 경우" → 검색 독려
3. Answer-chosen cross-type 제거 유지: 검색 억제 노이즈
4. Verbose answer → gold answer 교체
5. 새 프롬프트 전체 적용

파일:
- `outputs/mcts_dpo_5000q.jsonl` — V1 (13,183 pairs)
- `outputs/mcts_dpo_5000q_cleaned.jsonl` — V2 (7,192 pairs)
- `outputs/mcts_dpo_5000q_v3.jsonl` — V3 (8,724 pairs)

## 데이터 양 vs 품질: 양이 압도적으로 중요

### 실험 결과

| 데이터 | Pairs | 필터링 | EM | F1 |
|--------|-------|--------|-----|-----|
| **V1 (원본)** | **13,183** | 없음 | **38.06%** | **50.36%** |
| V2 (cleaned) | 7,192 | cross-type 제거, low margin 제거 | 32.79% | 43.17% |
| V3 (hybrid) | 9,340 | V2 + cross-type 일부 복원 | 32.52% | 43.31% |

**V1 > V3 > V2 순서가 pair 수 순서와 정확히 일치.**

### 원인 분석 (V1 vs V3)

| 지표 | V1 | V3 |
|------|-----|-----|
| 평균 검색 횟수 | 2.03 | 1.93 |
| 1회 검색 비율 | 42.5% | **57.1%** (1회에 치우침) |
| 1회 검색 EM | 41.0% | 36.4% (-4.6%p) |
| 2회 검색 EM | 42.5% | 35.1% (-7.4%p) |
| V1만 정답 | 854개 (11.5%) | - |
| V3만 정답 | - | 444개 (6.0%) |

- V3는 검색 횟수는 비슷하지만, **같은 횟수에서의 정답률이 일관되게 낮음**
- Low margin 제거가 "미묘한 차이를 구분하는 판단력" 학습 신호를 없앤 것으로 추정
- **노이즈로 보였던 데이터가 실제로 유용한 학습 신호였음**

### 결론
- 데이터 **필터링보다 양 확보**가 더 중요
- 프롬프트 수정 + 원본 데이터(V1)가 최적 조합
- **→ Pair 수를 더 늘리는 방향으로 진행 (Regeneration)**

## Data Regeneration 전략

### 현황: 5,000개 질문 중 pair 부족

| 상태 | 질문 수 | 비고 |
|------|---------|------|
| Pair 0개 | 886개 | 대부분 F1=0 (모델이 답 못 맞춤) |
| Pair 1~2개 | 1,923개 | 데이터 부족 |
| Pair 3개+ | 2,191개 | 충분 |
| **Regeneration 대상** | **2,809개** | |

### 886개 pair 없는 질문 분석

최고 F1 분포:
- F1=0 (완전 실패): 665개 (75%)
- F1>0이지만 형제 reward 동일: 221개 (25%)

Critic reasoning 실패 패턴:
- unsupported (근거 없는 답변): 2,975회
- premature_termination (너무 일찍 포기): 1,736회
- incorrect_answer (오답): 1,275회
- contradicts_docs (문서와 모순): 979회

### 재생성 대상 재분류

"탐색 부족"은 MCTS가 UCB score 기반으로 탐색할 가치가 없다고 판단한 것 → 같은 모델로 재시도해도 개선 어려움.
실질적 대상은 **F1=0 (답을 못 맞춘 질문)**만.

| 대상 | 수 | 재생성 가치 |
|------|-----|------------|
| F1=0 (답 못 맞춤) | 1,573개 | **높음** — critic feedback으로 다른 경로 유도 가능 |
| 탐색 부족 (UCB 낮음) | 1,015개 | 낮음 — 모델 능력 문제, feedback만으로 해결 어려움 |
| Reward 동일 | 221개 | 낮음 |

### 채택 전략: Critic Feedback-Guided Regeneration (GenPRM 스타일)

**참고 논문**: GenPRM (AAAI 2026) — PRM의 reasoning을 policy model에 피드백하여 재생성하는 critique refinement 방식

**핵심 아이디어**: Critic PRM의 reasoning(왜 틀렸는지)을 base model에 주입하여, 같은 실수를 반복하지 않도록 유도

**생성 모델은 항상 base Qwen2.5-7B-Instruct** — DPO 모델 사용 X (논문에서 정당화 가능: Critic PRM이 데이터 품질을 가이드하는 것이 contribution)

#### 구현 파이프라인

```
1. 대상 질문 선정: F1=0인 1,573개 질문
2. 기존 trajectory에서 최고 점수 경로 선택
3. Critic score가 떨어지는 지점(실패 지점) 찾기
4. 해당 지점의 critic reasoning 추출
5. Base model 프롬프트에 주입:
   "[이전 시도 피드백] {critic_reasoning}. 이 문제를 피하여 다른 접근을 시도하세요."
6. 실패 지점부터 재생성 (이전 good steps는 유지)
7. Critic으로 다시 scoring → DPO pair 추출
```

#### Round 계획
- **Round 1**: 1,573개 질문 × critic feedback 주입 → 재생성
- **Round 2**: Round 1에서도 실패한 질문 → feedback 갱신하여 재시도 (필요시)

#### 예상 결과
- V1 (13,183 pairs) + 새 pairs → **15,000~17,000 pairs** 예상
- 데이터 양 증가 → V1 대비 성능 향상 기대

### TODO
- [ ] 1,573개 대상 질문 추출 + 실패 지점 critic reasoning 추출
- [ ] 재생성 스크립트 구현 (build_mcts_dpo.py 기반)
- [ ] Round 1 실행
- [ ] 새 pair를 V1에 합쳐서 DPO 재학습 → eval

## Eval Infrastructure

- FlashRAG SearchR1Pipeline: multi-turn agentic RAG
- FlashRAG SequentialPipeline: standard single-turn RAG
- vLLM + LoRA: tensor_parallel=2에서 hang → merged model 사용
- FAISS IVF4096: nprobe=128, faiss_gpu=True, gpu_util=0.75
- FlashRAG retriever.py에 nprobe=128 패치 적용됨

## 모델 경로

- Our DPO V1 merged: `outputs/dpo_policy_mcts_v1/merged_model/`
- Our DPO V2 cleaned merged: `outputs/dpo_policy_mcts_v2_cleaned/merged_model/`
- Our SFT V2 merged: `outputs/sft_policy_v2/merged_model/`
- ReasonRAG merged: `outputs/reasonrag_merged_model/`
- ReasonRAG LoRA (HF): `reasonrag/Qwen2.5-7B-Instruct-RAG-Lora`
  - adapter_config의 base_model은 `Qwen/Qwen2.5-7B` (not Instruct)
  - 우리는 Instruct에 merge함
- Search-R1 PPO: `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo`
- Our DPO V3 merged: `outputs/dpo_policy_mcts_v3/merged_model/`

## DPO 데이터 파일
- `outputs/mcts_dpo_5000q.jsonl` — V1 원본 (13,183 pairs) ← **현재 최고 성능**
- `outputs/mcts_dpo_5000q_cleaned.jsonl` — V2 cleaned (7,192 pairs)
- `outputs/mcts_dpo_5000q_v3.jsonl` — V3 hybrid (9,340 pairs)
- `outputs/mcts_dpo_5000q_tree.jsonl` — MCTS tree 원본 (5,000 질문, 전체 노드)
