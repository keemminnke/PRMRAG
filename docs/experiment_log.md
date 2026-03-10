# PRMRAG 실험 기록

## 모델 구성
- **Base model**: `Qwen/Qwen2.5-7B-Instruct` (7B)
- **Critic model**: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` + LoRA
- **Retrieval**: BGE-M3 (cached) + BM25 + E5-large-v2
- **평가 데이터**: HotpotQA validation 7,405 questions
- **평가 방법**: FlashRAG SearchR1Pipeline (top-k=3, temp=0.8)

---

## 실험 결과 요약

| # | Model | EM | F1 | 데이터 | DPO Epoch | 비고 |
|---|-------|-----|-----|--------|-----------|------|
| 0 | Qwen Base (no training) | 29.7% | 40.8% | - | - | 기준선 |
| 0.5 | DPO only (SFT 없이) | 29.5% | 41.2% | 2,000q, 13,467 pairs | 3 | 효과 없음 |
| 1 | SFT v1 | 34.9% | 46.0% | 2,000q | - | SFT만 |
| 2 | SFT v1 + DPO v1 (ep3) | 34.8% | 46.1% | 2,000q, 13,467 pairs | 3 | DPO 효과 없음 |
| 3 | SFT v2 | 34.5% | 45.7% | 3,000q | - | 데이터 늘려도 개선 없음 |
| 4 | SFT v2 + DPO v2 (ep3) | 34.4% | 45.9% | 3,000q, 19,209 pairs | 3 | DPO 효과 없음, 과적합 의심 |
| 5 | SFT v2 + DPO v2 (ep1) | - | - | 3,000q, 19,209 pairs | 1 | 중단 (prompt 불일치 발견) |
| 6 | SFT v2 + DPO v2 (ep1, prompt 통일) | 34.2% | 45.5% | 3,000q, 19,209 pairs | 1 | DPO 효과 없음 |
| 7 | DPO only (base, 5,000q) | - | - | 5,000q, 27,323 pairs | 1 | 중단 (loss 미감소) |
| 8 | SFT v2 + traj-DPO (5,000q) | - | - | 5,000q, 27,323 pairs | 1 | 중단 → step-level DPO로 전환 |
| 9 | Step-level DPO (base) | 33.7% | 44.8%| 5,000q, 41,007 pairs | 1 | Base + step-level, 기준선과 비슷 |

---

## 실험 상세

### Exp 0: Qwen Base (기준선)
- **결과**: EM 29.7%, F1 40.8%
- **설명**: 학습 없이 Qwen2.5-7B-Instruct 그대로 평가
- **평가 경로**: `outputs/eval/flashrag_qwen_base_full/`

### Exp 1: SFT v1 (2,000q)
- **결과**: EM 34.9%, F1 46.0%
- **설명**: 2,000개 질문에서 생성한 trajectory 중 all-GOOD만 SFT
- **학습 데이터**: 2,000q 기반 all-GOOD trajectories
- **모델 경로**: `outputs/sft_policy_v1/merged_model`
- **평가 경로**: `outputs/flashrag_sft_v1_eval/`

### Exp 2: SFT v1 + DPO v1 (2,000q, epoch 3)
- **결과**: EM 34.8%, F1 46.1%
- **설명**: SFT v1 위에 DPO 학습 (Best-vs-All)
- **DPO 데이터**: 13,467 pairs (messages format)
- **하이퍼파라미터**: beta=0.1, lr=5e-7, LoRA r=16, epoch=3
- **모델 경로**: `outputs/dpo_policy_v1/final_model`
- **평가 경로**: `outputs/eval/flashrag_dpo_bva_full/`
- **분석**: SFT 대비 DPO 추가 효과 거의 없음 (EM -0.1%p)

### Exp 3: SFT v2 (3,000q)
- **결과**: EM 34.5%, F1 45.7%
- **설명**: 3,000개 질문 (기존 2,000 + 신규 1,000)으로 데이터 확장 후 SFT
- **학습 데이터**: 53,976 trajectories (원본 47,728 + regen 6,248) 중 all-GOOD만
- **Critic**: v9 (DeepSeek-R1-0528-Qwen3-8B + LoRA, 3,000q 학습)
- **모델 경로**: `outputs/sft_policy_v2/merged_model`
- **평가 경로**: `outputs/flashrag_sft_v2_3000q/`
- **분석**: 2,000q → 3,000q 확장해도 성능 개선 없음. SFT 포화 의심

### Exp 4: SFT v2 + DPO v2 (3,000q, epoch 3)
- **결과**: EM 34.4%, F1 45.9%
- **DPO 데이터**: 19,209 pairs (Best-vs-All, messages format)
- **하이퍼파라미터**: beta=0.1, lr=5e-7, LoRA r=16, epoch=3
- **최종 학습 지표**:
  - Loss: 0.423 (초기 0.697)
  - Rewards accuracy: 82.8%
  - Rewards margin: 1.085
- **모델 경로**: `outputs/dpo_policy_v2/final_model`
- **평가 경로**: `outputs/flashrag_dpo_v2_3000q_full/`
- **분석**: DPO 과적합 (rewards acc 82.8%). SFT 대비 개선 없음

### Exp 5: SFT v2 + DPO v2 (3,000q, epoch 1) — 중단
- **결과**: 중단
- **사유**: System prompt 불일치 발견으로 중단
- **DPO system prompt**: `"You are a helpful assistant..."` (SFT와 다름)

### Exp 6: SFT v2 + DPO v2 (3,000q, epoch 1, prompt 통일)
- **결과**: EM 34.2%, F1 45.5%
- **목적**: SFT와 동일한 system prompt로 DPO 학습 + epoch 1로 과적합 방지
- **핵심 변경**: DPO system prompt를 SFT와 동일한 `"You are an advanced AI agent..."` 로 통일
- **DPO 데이터**: 19,209 pairs (Best-vs-All, messages format)
- **하이퍼파라미터**: beta=0.1, lr=5e-7, LoRA r=16, **epoch=1**
- **분석**: prompt 통일 + epoch 1로도 DPO 효과 없음. **trajectory-level DPO 자체가 한계**

### Exp 7: DPO only (base, 5,000q) — 중단
- **중단 사유**: SFT 없이 base model에서 DPO → loss가 0.693에서 0.658까지밖에 안 내려감
- **Rewards accuracy**: 57~61% (거의 랜덤 수준)
- **결론**: SFT 없이 DPO는 task distribution을 모르는 상태에서 preference를 학습하기 어려움

### Exp 8: SFT v2 + trajectory-level DPO (5,000q) — 중단
- **중단 사유**: Step-level DPO로 전환
- **DPO 데이터**: 27,323 pairs (5,000q, Best-vs-All)

### Exp 9: Step-level DPO on Base (5,000q, epoch 1)
- **결과**: EM 33.7%, F1 44.8%
- **Base model**: `Qwen/Qwen2.5-7B-Instruct` (SFT 없이 base 위에 직접 DPO)
- **핵심 변경**: Trajectory-level → Step-level DPO (GOOD vs BAD critic labels)
- **DPO 데이터**: 41,007 pairs (Source 1 only, 기존 데이터 step-level 추출)
  - Step 1: 1,410 pairs
  - Step 2: 29,495 pairs (주력)
  - Step 3: 9,676 pairs
  - Step 4-6: 426 pairs
  - 3,430 unique questions
- **학습 지표** (최종):
  - Loss: 0.693 → 0.583
  - Rewards accuracy: 41% → 70%
  - Rewards margin: 0.003 → 0.345
- **하이퍼파라미터**: beta=0.1, lr=5e-7, LoRA r=16, epoch=1
- **모델 경로**: `outputs/dpo_step_level_v1_base/merged_model`
- **평가 경로**: `outputs/flashrag_step_dpo_v1_base/`
- **분석**: 학습 지표는 크게 개선 (기존 traj-DPO 대비 loss, accuracy 모두 양호), 하지만 평가 성능은 base(29.7%)보다 +4%p, SFT(34.9%)보다 -1.2%p. SFT 없이 base 위에서 DPO만으로는 SFT 수준 도달 어려움. → **SFT 위에 step-level DPO 실험 필요**

---

## Step-level DPO (ReasonRAG 방식 적용)

### 문제: 왜 Trajectory-level DPO가 안 되는가?
1. **신호 희석**: 5~8 step 전체를 비교하면 어디가 좋고 나쁜지 학습 신호가 약해짐
2. **Reward gap 부족**: chosen/rejected trajectory가 비슷한 경우 많음
3. **ReasonRAG 논문**: step-level에서 `p(y_t^w|x,y_{<t})` vs `p(y_t^l|x,y_{<t})` 비교 → 명확한 분기점 학습

### 방법: 기존 데이터에서 Step-level pairs 추출

**핵심 원리**: 같은 question의 trajectory들은 특정 step까지 동일한 prefix를 공유할 수 있음

#### Source 1: 기존 데이터 (regeneration 없이)
같은 question에서 동일 query sequence를 가진 trajectory 중 correct vs incorrect 비교

| Step | 동일 prefix에서 분기하는 pairs |
|------|------|
| Step 1 | 67,014 (prefix = question, 모든 trajectory 공유) |
| Step 2 | 17,445 (동일 query1 공유) |
| Step 3 | 5,061 (동일 query1,2 공유) |
| Step 4~5 | 185 |
| **소계** | **89,705** |

#### Source 2: Regeneration (K=3)
BAD step 지점에서 prefix를 고정하고 대안 step을 3번 재생성

| BAD step 위치 | Unique prefixes | 추정 pairs (K=3, 성공률 50%) |
|------|------|------|
| Step 2 | 11,165 | ~16,748 |
| Step 3 | 10,678 | ~16,017 |
| Step 4 | 3,571 | ~5,357 |
| 기타 | 1,543 | ~2,315 |
| **소계** | **26,957** | **~40,437** |

#### 합계
| 소스 | Pairs | 비고 |
|------|-------|------|
| 기존 데이터 step-level | 89,705 | 추가 비용 없음 |
| Regen step-level (K=3) | ~40,437 | vLLM ~2시간 |
| **합계** | **~130,142** | trajectory-level 27,323의 **약 5배** |

### 추가 이점
- **1,473개 question** (모든 trajectory 오답) → regen으로 1,373개 추가 커버 가능
- 모든 pair가 **물리적으로 동일한 prefix**에서의 1-step 차이 → 학습 신호 명확
- ReasonRAG과 동일한 step-level DPO loss 적용

### 구현 계획
1. 기존 5,000q scored 데이터에서 step-level pairs 추출 (Source 1)
2. BAD step prefix에서 K=3 재생성 + critic scoring (Source 2)
3. Step-level DPO 학습 (SFT v2 base, epoch 1)
4. 평가

---

## 핵심 관찰 및 분석

### 1. SFT만으로 충분한 성능
- SFT v1 (34.9%) ≈ DPO v1 (34.8%) ≈ SFT v2 (34.5%) ≈ DPO v2 (34.4%)
- SFT가 base 대비 +5%p EM 개선, DPO는 추가 개선 없음
- **결론**: 현재 DPO 데이터/설정으로는 SFT 이상의 학습 signal을 제공하지 못함

### 2. 데이터 확장 효과 없음
- 2,000q → 3,000q로 50% 증가시켜도 성능 정체
- **가능 원인**:
  - SFT 학습이 이미 포화 (all-GOOD trajectory 패턴을 충분히 학습)
  - 추가 데이터의 다양성 부족 (같은 HotpotQA 분포)
  - Critic의 판별력 한계 (all-GOOD 판정 기준이 너무 관대하거나 엄격)

### 3. System Prompt 불일치 (Exp 1~5 공통 문제)
- SFT: `"You are an advanced AI agent capable of Adaptive RAG..."` (긴 상세 prompt)
- DPO: `"You are a helpful assistant..."` (짧은 1줄 prompt)
- 평가: `"You are a multi-step reasoning assistant..."` (중간 길이)
- **3곳 모두 다른 prompt 사용** → DPO 학습 효과가 평가에 반영 안 됨
- **Exp 6에서 SFT/DPO prompt 통일하여 재실험 중**

### 4. DPO 과적합 문제
- Epoch 3 기준 rewards accuracy 82.8%, margin 1.085
- Train 데이터를 너무 잘 구분하지만, 일반화 실패
- **대응**: epoch 1로 줄여서 실험 중 (Exp 5)

### 4. Regen 품질 이슈
- Regen all-GOOD 비율: v1 5.9%, v2 4.6%
- Regen으로 생성된 데이터의 품질이 낮아 DPO chosen 후보로서 한계

---

## Round 2+3 실험 계획

### 데이터 생성 (한번에)
- **Round 2**: MusiQue +1,000q (3,000q → 4,000q)
- **Round 3**: 2WikiMultihopQA +1,000q (4,000q → 5,000q)
- 스크립트: `scripts/run_round2_3_data.sh`
- 불량 trajectory 필터링 적용 (no answer, max_step>=10)

### 학습 실험 (데이터 고정, 학습만 변경)
스크립트: `scripts/run_training_experiments.sh`

| Exp | 설정 | 데이터 | DPO | 목적 |
|-----|------|--------|-----|------|
| 7 | SFT only | 4,000q | - | 데이터 스케일링 효과 |
| 8 | SFT only | 5,000q | - | 데이터 스케일링 효과 |
| 9 | SFT + DPO ep1 | 5,000q | 1 | DPO + prompt 통일 효과 |
| 10 | SFT + DPO ep3 | 5,000q | 3 | DPO epoch 비교 |
| 11 | SFT + DPO ep1 | 4,000q | 1 | 데이터 × DPO 조합 |

### 핵심 변경사항 (Round 1 교훈)
- **System prompt 통일**: SFT/DPO/eval 모두 동일한 prompt 사용
- **불량 trajectory 필터링**: no answer, max_step>=10 제외
- **eval_flashrag.py**: system prompt를 SFT와 동일하게 수정 완료

---

## 학습 설정 참고

### SFT 공통
- LoRA: r=16, alpha=32
- LR: 2e-5
- Batch: 2 × 8 grad_accum × 2 GPU = 32 effective
- Loss masking: prompt + `<documents>` 구간 제외, think/search/answer만 학습
- 데이터: all-GOOD trajectories만 사용

### DPO 공통
- LoRA: r=16, alpha=32
- LR: 5e-7
- Beta: 0.1
- Loss: sigmoid
- Max length: 4096
- Pairing: Best-vs-All
- Format: messages (list[dict]) — text format 사용 시 tokenization mismatch 발생

### 평가
- FlashRAG SearchR1Pipeline
- top-k=3, temperature=0.8
- 전체 7,405 questions
- Metrics: EM, F1
