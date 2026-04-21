# PRO-Step Ablation & Hyperparameter Tuning Plan (v2 — v8 Critic)

## 0. Critic 선정 근거

### V8 vs V9 비교 결과 (동일 테스트셋, 500q × 128 traj)

| Dataset | Method | V8 (2000q 학습) | V9 (3000q 학습) |
|---------|--------|-----------------|-----------------|
| HotpotQA | WMV min | **51.0%** | 50.2% |
| 2Wiki | WMV min | **53.0%** | 52.2% |
| PopQA | WMV min | **53.6%** | 52.8% |

Step-level accuracy (unseen 2000q):
- V8: Acc=82.62%, BAD Recall=57.5%
- V9: Acc=82.49%, BAD Recall=56.5%

**결론: V8이 3개 벤치마크에서 일관되게 +0.8%p 우세. 데이터 양(3000q)보다 품질(2000q)이 중요.**

### 선정 Critic: **V8 (2000q, r=16, α=16)**
- Base: DeepSeek-R1-0528-Qwen3-8B
- LoRA: r=16, α=16
- Path: `outputs/critic_model_v8_2000q/final_model`

### Critic v8 vs VersaPRM vs MathPRM — BoN Scaling (500q × 128 traj)

**결론: Critic v8이 3개 벤치마크 모두에서 우세. avg aggregation이 min보다 robust.**

#### HotpotQA (BoN, EM%)

| k | Majority | **Critic v8 avg** | Critic v8 min | VersaPRM avg | VersaPRM min | MathPRM avg | MathPRM min |
|---|----------|-------------------|---------------|--------------|--------------|-------------|-------------|
| 1 | 48.3 | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 | 48.0 |
| 8 | 51.2 | **54.4** | 53.6 | 52.2 | 52.2 | 50.0 | 48.4 |
| 32 | 52.2 | **58.8** | 56.2 | 52.2 | 52.8 | 50.4 | 51.0 |
| 128 | 53.6 | **58.4** | 59.0 | 50.8 | 51.8 | 48.8 | 48.4 |

#### 2WikiMultiHopQA (BoN, EM%)

| k | Majority | **Critic v8 avg** | Critic v8 min | VersaPRM avg | VersaPRM min | MathPRM avg | MathPRM min |
|---|----------|-------------------|---------------|--------------|--------------|-------------|-------------|
| 1 | 44.3 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 | 44.2 |
| 8 | 47.8 | **50.4** | 49.8 | 48.4 | 47.2 | 48.8 | 47.6 |
| 32 | **49.8** | 49.8 | 49.2 | 46.4 | 43.6 | 45.6 | 44.8 |
| 128 | **50.8** | 48.6 | 47.8 | 41.4 | 40.0 | 41.2 | 42.2 |

#### PopQA (BoN, EM%)

| k | Majority | **Critic v8 avg** | Critic v8 min | VersaPRM avg | VersaPRM min | MathPRM avg | MathPRM min |
|---|----------|-------------------|---------------|--------------|--------------|-------------|-------------|
| 1 | 51.3 | 51.2 | 51.2 | 51.2 | 51.2 | 51.2 | 51.2 |
| 8 | 50.8 | **55.2** | 54.4 | 53.2 | 53.6 | 48.8 | 50.2 |
| 32 | **52.4** | 54.4 | 52.6 | 50.4 | 49.8 | 45.2 | 47.4 |
| 128 | **52.2** | 52.6 | 50.4 | 46.0 | 43.8 | 34.8 | 40.6 |

**핵심 관찰:**
1. **HotpotQA**: Critic v8이 모든 k에서 1등. k=128에서 58~59% (다른 PRM은 50% 수준)
2. **2Wiki / PopQA**: k≥32부터 Majority가 PRM들보다 좋아짐 (BoN은 k 클수록 약점)
3. **avg > min**: Critic v8 k=128에서 avg=58.4 vs min=59.0 (비슷), 하지만 2Wiki/PopQA에서 avg가 더 robust
4. **MathPRM이 가장 빠르게 하락**: PopQA k=128에서 34.8%까지 급락
5. **Critic v8은 VersaPRM 대비 모든 구간에서 우위**

**결론: BoN으로 쓸 경우 k=8~32 구간이 sweet spot. k≥64에서는 WMV가 더 안전.**

---

## 1. 기존 성능 기준선

### 1.1 실험 결과 요약 (HotpotQA EM, **BGE retriever + fixed pipeline** 2026-04-19)

**Eval pipeline 수정**: `faiss_gpu=False` + `nprobe=128` (vLLM V1 엔진 + FAISS GPU shard 충돌 버그 회피).

| Version | Tree/Critic | Filter | α | DPO β | Pairs | EM | F1 | vs V1 |
|---|---|---|---|---|---|---|---|---|
| **V1 (baseline)** | K=2, V9 critic | δ=0.01 | 0.3 | 0.1 | 13,183 | 0.3830 | 0.5044 | — |
| SearchR1 PPO (ref) | — | — | — | — | N/A | 0.3838 | 0.4998 | +0.1%p |
| K=3 raw | K=3, V8 critic | δ=0.01 | 0.3 | 0.1 | 39,578 | 0.3762 | 0.5058 | -0.7% |
| K=3 sampled13k | K=3, V8 critic | δ=0.01 | 0.3 | 0.1 | 13,183 | 0.378 | — | -0.5% |
| K=3 deduped | K=3, V8 critic | prompt dedup | 0.3 | 0.1 | 19,044 | 0.3745 | 0.4966 | -0.8% |
| K=3 δ0.4 | K=3, V8 critic | δ=0.4 | 0.3 | 0.3 | 10,287 | 0.3773 | 0.5012 | -0.6% |
| K=3 F4 | K=3, V8 critic | cho_f1 ≥ 0.5 | 0.3 | 0.1 | 13,234 | 0.3581 | 0.4578 | -2.5% ❌ |
| **🎯 K=3 F6+α0.3 (best)** | K=3, V8 critic | **cho≥0.2 AND Δ≥0.2** | **0.3** | 0.1 | 15,877 | **0.3873** | **0.5163** | **+0.4%p (+1.2%p F1)** |
| K=3 F6+α0.5 | K=3, V8 critic | cho≥0.2 AND Δ≥0.2 | 0.5 | 0.1 | 15,596 | 0.3495 | 0.4753 | -3.3% |
| K=3 F6+α1.0 | K=3, V8 critic | cho≥0.2 AND Δ≥0.2 | 1.0 | 0.1 | 15,059 | 0.3797 | 0.5060 | -0.3% |
| K=3 F6+α0 (ablation) | K=3, V8 critic | cho≥0.2 AND Δ≥0.2 | 0.0 | 0.1 | 16,065 | 0.3674 | 0.4931 | -1.6% |
| K=3 F6tight | K=3, V8 critic | cho≥**0.3** AND Δ≥0.2 | 0.3 | 0.1 | 13,944 | 0.3598 | 0.4728 | -2.3% ❌ |
| V1-tree α=0 (ablation) | K=2, V9 critic | δ=0.01 | 0.0 | 0.1 | 9,551 | 0.2933 | 0.4080 | -9.0% |

**확정 방향**: **K=3 F6 filter가 새 baseline** — V1(0.383) 및 SearchR1 PPO(0.384) 모두 초과.

**핵심 관찰**:
1. **F6+α0.3이 V1 초과 (0.383 → 0.387)** — 2-axis filter (chosen floor + margin floor)
2. **F6 성공은 대부분 Yes/No 질문에서 (+4.2%p)** — Binary decision에 clean pair가 critical
3. **F4 실패 (cho_f1≥0.5만)** — chosen floor 만으로는 overfit (질문 coverage 손실)
4. **V1-tree α=0 → EM -9%p** — PRM 가치 증명 (핵심 ablation)
5. **F6+α0 vs F6+α0.3 = -2.0%p** — F6 하에서도 α 기여 유효 (97% overlap pair임에도)
6. **α × F6 상호보완**: α=0 no-F6 (0.293) < F6+α0 (0.367) < F6+α0.3 (0.387) → 둘 다 필요
7. **V1(13k) vs K=3 raw(39k)**: pair 수 scaling 자체는 무효. 필터 설계가 진짜 신호.
8. **α sweep non-monotonic**: α ∈ {0, 0.3, 0.5, 1.0} → {0.367, 0.387, 0.350, 0.380}. α=0.3 sweet spot.
9. **Chosen floor 엄격화 실패**: F6tight (cho≥0.3) 0.360 — F4 패턴 재확인.

**주의: 모든 비교는 BGE retriever + fixed pipeline 기준**

### 1.3 Phase A α sweep (2026-04-21 완료)

F6 filter 고정 + α extraction 값만 변경:

| α | Pairs | EM | F1 | vs V1 |
|---|---|---|---|---|
| 0.0 (F1 only) | 16,065 | 0.3674 | 0.4931 | -1.6%p |
| **0.3 ⭐** | 15,877 | **0.3873** | **0.5163** | **+0.4%p** |
| 0.5 | 15,596 | 0.3495 | 0.4753 | -3.3%p |
| 1.0 | 15,059 | 0.3797 | 0.5060 | -0.3%p |

**결론**: α=0.3이 global optimum. α=0.5에서 큰 dip (critic 가중치 과해서 F1 ordering 방해). α=1.0 (pure critic)은 α=0.3 근접하지만 도달 못함.

### 1.4 Phase B F6 threshold sweep (진행 중)

F6 base = (cho_floor=0.2, Δ_floor=0.2). α=0.3 고정.

| Config | cho_floor | Δ_floor | Pairs | EM | F1 | vs base |
|---|---|---|---|---|---|---|
| **F6 base ⭐** | 0.2 | 0.2 | 15,877 | **0.3873** | **0.5163** | — |
| F6tight | 0.3 | 0.2 | 13,944 | 0.3598 | 0.4728 | -2.8%p ❌ |
| F6margin | 0.2 | 0.3 | ~13,000 | 🔄 학습 중 | — | — |
| F6loose | 0.1 | 0.2 | ~19,000 | 대기 | — | — |

**중간 결론**: chosen_floor 올리는 건 F4와 동일 패턴 — 질문 coverage 손실로 overfit.

### 1.5 Benchmark 일반화 (F6+α0.3, 5 datasets)

SearchR1/ReasonRAG와 비교 (모두 BGE retriever, 동일 fixed pipeline):

| Benchmark | F6+α0.3 (ours) | SearchR1 PPO | ReasonRAG |
|---|---|---|---|
| HotpotQA | **EM 0.387 / F1 0.516** ⭐ | 0.379 / 0.496 | 0.364 / 0.475 |
| PopQA | 0.404 / **0.474** | **0.407** / 0.468 | 0.378 / 0.449 |
| 2Wiki | **0.441 / 0.515** ⭐ | 0.349 / 0.425 | 0.398 / 0.463 |
| Bamboogle | 0.368 / **0.471** | 0.336 / 0.436 | **0.384** / 0.469 |
| Musique | 0.120 / 0.219 | **0.130** / 0.212 | 0.106 / 0.192 |

**결과**: 
- **3 대승**: HotpotQA, 2Wiki, Bamboogle F1
- **2Wiki 압도**: EM +9.2%p vs SearchR1, +4.3%p vs ReasonRAG
- **1 박빙 패**: PopQA (SearchR1 -0.3%p EM, F1 우위)
- **1 근소 패**: Musique (multi-hop 어려움, 전부 낮음)

### 1.2 v1 DPO 데이터 품질

| Metric | v1 | newprompt |
|--------|-----|-----------|
| Pairs | 13,183 | 10,672 |
| Noise (F1 diff < 0.01) | 27.6% | 35.6% |
| Ideal (cho=GOOD, rej=BAD) | 39.4% | 45.6% |
| Reversed (cho=BAD, rej=GOOD) | 0.8% | 2.1% |

---

## 2. Prompt Alignment 분석

### 2.1 Judge → Critic → MCTS Prompt 불일치 발견

| | Judge (라벨링) | Critic Training | MCTS Inference (v1) |
|---|---|---|---|
| Gold Answer | O (참조용) | **X** | **O** ← mismatch |
| 상세 Criteria | O (R1~R6) | X (간단) | X (간단) |
| 평가 단위 | 전체 trajectory | step 1개씩 | step 1개씩 |

### 2.2 VersaPRM 논문 분석 결과
- VersaPRM은 classification head 방식 → prompt alignment 불필요
- 우리는 생성형 SFT → prompt가 직접 영향
- **조치**: MCTS critic prompt에서 Gold Answer 제거하여 training과 일치시킴

### 2.3 수정 내역
- `build_mcts_dpo.py:391`: 초기에 `Gold Answer: {tree.gold_answer}` 제거 시도 → **복구**
- F1 reward는 여전히 gold answer로 계산 (terminal node scoring)

### 2.4 Gold Answer 복구 (2026-04-14)
- `mcts_tree_v8_gold_k3.jsonl` (Apr 14 빌드): 파일명의 "gold" 는 **critic prompt에 gold answer 재포함**을 의미
- 현재 `build_mcts_dpo.py:391` 코드: `f"Gold Answer: {tree.gold_answer}"` **포함 상태**
- 이유: gold answer 제거했을 때 critic이 모호한 pair를 못 구분 → pair 품질 저하

---

## 3. MCTS 재빌드 (V8 Critic, Gold Answer 제거)

### 3.1 빌드 파라미터

| Parameter | 값 | 비고 |
|-----------|-----|------|
| policy_model | Qwen2.5-7B-Instruct | v1과 동일 |
| critic_model | **v8 (2000q)** | v9 → v8 변경 |
| critic prompt | **gold answer 제거** | training과 일치 |
| max_children (K) | 2 | v1과 동일 |
| max_depth | 7 | v1과 동일 |
| max_rollouts | 64 | v1과 동일 |
| f1_discount (β) | 0.9 | v1과 동일 |
| critic_alpha | 0.3 | v1과 동일 |
| critic_interval | 4 | v1과 동일 |
| top_k | 3 | v1과 동일 |
| retriever | BGE (FAISS IVF) | v1과 동일 |

### 3.2 빌드 결과

| | v1 (v9, gold O) | v8 (no gold) |
|---|---|---|
| Total nodes | 66,520 | 65,270 |
| Terminal nodes | 25,011 | 23,477 |
| Critic scored (inline) | 61,520 | 36,793 |
| 2-children nodes | 27,263 | 26,975 |

**Sibling reward diff 분포 변화:**
- v1: no diff (< 0.01) = 51.3%
- v8: no diff (< 0.01) = **60.4%** (+9.1%p)
- → gold answer 제거로 critic 판별력 저하, pair 수 감소

---

## 4. (α, δ) Sweep 결과

### 4.1 Pair Extraction Sweep (v8 tree)

| α | δ | Pairs | Noise% | Ideal% | Rev% | Avg F1 diff |
|---|---|-------|--------|--------|------|-------------|
| 0.0 | 0.005 | 7,010 | 0.7% | 9.8% | 7.8% | 0.355 |
| 0.0 | 0.010 | 6,960 | 0.0% | 9.9% | 7.8% | 0.357 |
| 0.3 | 0.010 | 10,619 | 34.6% | 43.2% | 2.8% | 0.234 |
| 0.5 | 0.010 | 10,631 | 34.6% | 44.1% | 2.0% | 0.234 |
| 0.5 | 0.050 | 10,096 | 36.4% | 46.3% | 2.0% | 0.243 |
| 1.0 | 0.010 | 10,634 | 34.5% | 46.1% | 0.0% | 0.234 |
| 1.0 | 0.050 | 10,124 | 36.3% | 48.4% | 0.0% | 0.244 |
| 1.0 | 0.100 | 9,498 | 38.7% | 51.5% | 0.0% | 0.254 |

**관찰:**
- α=0.0 (F1 only): pair 적지만(7k) noise 거의 없음(0.7%)
- α 올리면: pair 수 증가(10k+), 하지만 noise 35%
- α=1.0: reversed 0%로 가장 일관적

### 4.2 DPO Training + HotpotQA Eval (β=0.1, **BGE**, 7,405q 전체)

| # | α | δ | Pairs | EM (BGE) | vs v1 (0.381) |
|---|---|---|-------|----------|---------------|
| **1** | **0.0** | **0.005** | **7,010** | **0.450** | **+6.9%p** |
| **2** | **0.3** | **0.01** | **10,619** | **0.447** | **+6.6%p** |
| 5 | 1.0 | 0.05 | 10,124 | 0.424 | +4.3%p |
| 4 | 0.5 | 0.05 | 10,096 | 0.416 | +3.5%p |
| 3 | 1.0 | 0.01 | 10,634 | 0.406 | +2.5%p |

**핵심 발견 (K=2, eval_policy.py 내부 EM — FlashRAG EM과 다름, 참고용):**
- eval_policy.py 내부 EM 기준으로는 v1(0.381) 대비 개선으로 보였으나
- FlashRAG 표준 EM으로 재측정 시 전부 v1보다 하락 (아래 Section 4.3 참조)
- eval_policy.py의 `_check_answer`가 substring match라 EM이 부풀려짐

### 4.3 K=3 MCTS 실험 — DPO Training + FlashRAG Eval

#### K=3 MCTS Tree 결과 (v8 critic, gold answer O, 64 rollouts)

| | v1 (K=2) | K=3 |
|---|---|---|
| Nodes | 66,520 | 133,942 |
| Sibling comparisons | 27,263 | 107,757 |
| 3-children parents | 0 | ~27,000 |

#### K=3 DPO Training + HotpotQA FlashRAG EM (BGE, 전체 7405q)

| # | α | δ | Pairs | Steps | Loss final | EM (old broken) | EM (fixed 2026-04-19) | vs v1 |
|---|---|---|-------|-------|-----------|-----|-----|-------|
| **v1 baseline (K=2)** | 0.3 | 0.01 | **13,183** | **823** | **0.613** | ~0.381 | **0.383** | — |
| k3_a0.3_d0.01 (39k raw) | 0.3 | 0.01 | 39,578 | 2,474 | ~0.44 | 0.163 (broken) | **0.376** | -0.7%p |
| k3_a0.3_d0.01_deduped (19k) | 0.3 | 0.01 | 19,044 | 1,191 | ~0.50 | — | **0.375** | -0.8%p |
| k3_a0.3_d0.01_sampled13k | 0.3 | 0.01 | 13,183 | 824 | — | — | **0.378** | -0.5%p |
| k3_a0.3_d0.4_b0.3 | 0.3 | 0.4 | 10,287 | 643 | ~0.30 | — | **0.377** | -0.6%p |

**재측정 핵심 발견 (2026-04-19)**: 이전 "실패" 기록(0.163, 0.210, 0.242)은 전부 **eval pipeline 버그** (vLLM V1 + FAISS GPU shard 충돌)였음. 수정된 pipeline으로 재평가 시 모두 0.375~0.378 범위로 복원. **KL 폭발 / prompt 중복 이론 무효**.

#### 실패가 아니라 eval 버그였다 — 학습은 정상

**1. 학습 loss는 정상 수렴**
- k3_a0.3_d0.01 (39k): final loss 0.41~0.48, margin 1.3~1.7, acc 0.75~0.79 (문제 없음)

**2. "Prompt 중복 문제"**는 실제로는 성능에 큰 영향 없음:
- 39k raw (중복 51.9%): EM 0.376
- 19k dedup (중복 0%): EM 0.375
- 거의 동일 → 중복이 문제가 아니었음

**3. V1(K=2) 우위의 진짜 이유**
- V9 critic(V1) vs V8 critic(K=3) — DPO pair 품질 차이
- V1 데이터는 chosen_f1 분포가 **moderate 영역(0.1~0.3)** 26%, K=3는 양극화 (0-0.1 or 0.7-0.9)
- V1의 step 4+ pair 비중 10.8%, K=3는 2~4% (깊은 reasoning 부족)

#### 결론 & 대응

- K=3 tree로 pair 수를 3배 늘려도 **V1(V9 critic) 성능 못 넘음**
- Pair quantity가 아니라 **critic quality + chosen_f1 moderate 분포**가 핵심
- V8 critic은 step-level accuracy 우세했지만 DPO pair 생성엔 V9이 유리

#### K=3 Terminal Node 분석

| 답변 길이 | v1 (K=2) | K=3 |
|-----------|----------|-----|
| 짧은 (<30자) | 7,531 (39.0% 정답) | **49,113** (44.2% 정답) |
| 중간 (30-100자) | 6,888 (1.5% 정답) | 4,360 (13.9% 정답) |
| 긴 (100+자) | 10,592 (0.1% 정답) | 900 (1.4% 정답) |

→ K=3가 짧은 답변 7x 많이 생성, 정답률도 높음. **Tree 품질은 K=3가 우수. 학습 설정이 문제.**

### 4.4 K=3 δ+β 조정 실험

K=3 tree에서 δ(min_reward_diff)를 높여 pair 수를 v1 수준으로 맞추고, β도 올려 KL penalty 강화:

| Setting | δ | β | Pairs | Steps | Noise% | Ideal% |
|---------|---|---|-------|-------|--------|--------|
| v1 baseline | 0.01 | 0.1 | 13,183 | 823 | 27.6% | 39.4% |
| **K=3 new** | **0.4** | **0.3** | **10,287** | **642** | **0.0%** | **26.8%** |

**δ=0.4**: F1 차이가 큰 pair만 남김 → noise 0%, 강한 대비
**β=0.3**: KL penalty 3배 강화 → reference model에서 크게 벗어나지 않음

FlashRAG eval 결과 (fixed pipeline, 2026-04-19): **EM 0.3773, F1 0.5012** — V1(0.383)에 -0.6%p

---

## 4.5 F6 Filter — 최종 확정 (2026-04-20)

### 4.5.1 F4 vs F6 비교 실험

K=3 raw pair (39,578) 기반 filter 실험:

| Filter | 정의 | Pairs | Qs | cho_f1 평균 | ideal% | EM | F1 | vs V1 |
|---|---|---|---|---|---|---|---|---|
| Baseline (no filter) | α=0.3 δ=0.01 | 39,578 | 4,344 | 0.321 | 27.0% | 0.376 | 0.506 | -0.7% |
| **F4** | cho_f1 ≥ 0.5 | 13,234 | 2,419 | 0.712 | 80.6% | **0.358** | 0.458 | **-2.5% ❌** |
| **🎯 F6** | **cho_f1 ≥ 0.2 AND ΔF1 ≥ 0.2** | 15,877 | 2,866 | 0.590 | 63.7% | **0.3873** | **0.5163** | **+0.4%p / +1.2%p F1 ✅** |

**F4 실패 원인**:
- chosen_f1 floor만 → "쉬운 질문" (MCTS가 완벽 답 찾은 것)만 학습
- 2,419 질문으로 질문 coverage 줄어듦 → 어려운 eval 질문 overfit
- chosen_f1=0 pair 35.8% 제거 = negative coverage 상실

**F6 성공 원인**:
- 2축 동시 제약 (chosen floor + margin floor)
- Moderate chosen (0.2~0.9) + clear 대비 → balanced signal
- 2,866 질문 coverage 유지 + chosen 품질 보장

### 4.5.2 F6 Per-Question 분석

| 질문 유형 | # | V1 EM | F6 EM | Δ |
|---|---|---|---|---|
| **Yes/No** | 458 | 0.786 | **0.828** | **+4.2%p** ⭐ |
| Entity | 6,947 | 0.357 | 0.358 | +0.17%p |

**F6 개선은 대부분 Yes/No 질문에서 발생**. Entity 질문은 retrieval에 더 의존해서 DPO 효과 제한적.

### 4.5.3 α × F6 2×2 Ablation — 상호보완적 효과 확인

**Pair overlap 분석 (data-level)**:

| α (extraction) | F6 pairs | α=0.0와 overlap |
|---|---|---|
| α=0.0 | 16,065 | — |
| α=0.3 | 15,877 | **97%** |
| α=1.0 | 15,059 | 94% |

Pair set은 97% 공유하지만 나머지 3% 차이 (ordering + critic-only pair)가 **학습 결과에 큰 영향**.

**학습 결과 (F6 × α 2×2)**:

| | α=0 (F1-only) | α=0.3 (+critic) | α 효과 |
|---|---|---|---|
| **no F6 filter** | **0.293** (V1-tree α=0) | **0.383** (V1 baseline) | **+9.0%p** |
| **F6 filter** | **0.367** (F6+α0) | **0.387** 🎯 (F6+α0.3) | **+2.0%p** |
| F6 효과 | +7.4%p | +0.4%p | — |

**주요 발견**:
1. **α=0 + F6 (0.367) > α=0 w/o F6 (0.293)**: F6 filter가 α 부재 부분 보상 (+7.4%p)
2. **α=0.3 + F6 (0.387) > α=0.3 w/o F6 (0.383)**: F6 filter가 α 있어도 +0.4%p 추가
3. **α=0.3 + F6 > α=0 + F6**: F6 필터 하에서도 α가 여전히 +2.0%p 기여 (97% overlap pair임에도)
4. **두 효과는 **상호보완적****: F6가 α를 완전 대체하지 않음

### 4.5.4 논문적 함의

**Contribution 1 — α 필요성** (V1-tree ablation):
- α=0 vs α=0.3 (no F6): **-9.0%p** — PRM signal이 DPO pair 학습에 critical
- 이 효과 대부분은 critic-only pair (ΔF1=0)를 살리는 것에서 옴

**Contribution 2 — F6 filter 효과**:
- No-F6 → F6: α=0에서 +7.4%p, α=0.3에서 +0.4%p
- F6 filter가 특히 "α 없는 설정"에서 큰 효과 → filter가 critic signal 부분 대체 가능

**Contribution 3 — 상호보완성**:
- α=0.3 + F6 (0.387) > 단독 최고 (0.383)
- 두 기법은 **독립 축**: α는 implicit critic mixing, F6은 explicit F1-based filter
- Best: 두 기법 **조합** → +0.4%p over V1 + +2.0%p over F6-only

**논문 claim 최종**:
1. "PRM signal is essential: removing α=0.3 drops 9%p (V1-tree ablation)"
2. "Explicit 2-axis F1 filter (F6) provides additional signal beyond α: +0.4%p over α=0.3 baseline"
3. "F6 and α are **complementary** — F6+α0.3 achieves best (0.387 EM, 0.516 F1), outperforming V1 and SearchR1 PPO"

---

## 4.6 Eval Pipeline Bug 발견 & 수정 (2026-04-18/19)

### 발견 경위
- ABLATION_PLAN Section 4.3의 K=3 "실패" 모델들 재평가 시도 → SearchR1 baseline으로 sanity check
- SearchR1 baseline이 March 0.379 → 현재 eval에서 0.139 (-24%p) → **파이프라인 regression 확정**

### Root cause
- **vLLM V1 엔진 (0.14.0, Apr 6 설치) + FAISS GPU shard 동시 사용 시 IVF kernel state 오염**
- 증상: 모든 query가 동일한 무관한 문서 반환 (예: "Scott Derrickson" → Indian village articles)
- vLLM 단독 or FAISS GPU 단독일 땐 정상

### 수정
```python
# scripts/eval_flashrag.py
config_dict["faiss_gpu"] = False  # CPU FAISS (vLLM과 분리)
pipeline.retriever.index.nprobe = 128  # default nprobe=1 → 128로 (FlashRAG가 설정 안 함)
```

### 검증
- SearchR1 재평가: 0.139 → **0.384** (March 0.379과 일치)
- V1 DPO 재평가: broken → **0.383** (March 0.381과 일치)

### 파급효과
- Section 4.3의 "K=3 KL 폭발" 분석 → 전부 무효
- 학습은 정상이었고 eval만 망가졌던 것. 모든 K=3 실험 재평가 결과 V1과 ±1%p 이내.

---

## 5. 디렉토리 구조

```
outputs/
├── mcts_tree_v8critic.jsonl              # V8 critic MCTS tree (no gold answer)
├── mcts_dpo_v8critic.jsonl               # Default pairs (α=0.3, δ=0.01)
├── mcts_dpo_v8_a0.0_d0.005.jsonl         # Sweep pair datasets
├── mcts_dpo_v8_a0.3_d0.01.jsonl
├── mcts_dpo_v8_a1.0_d0.01.jsonl
├── mcts_dpo_v8_a0.5_d0.05.jsonl
├── mcts_dpo_v8_a1.0_d0.05.jsonl
├── dpo_v8_a{α}_d{δ}_b0.1/               # Trained models
│   └── final_model/
├── eval_v8_a{α}_d{δ}_b0.1_hotpotqa.jsonl # BGE eval results
├── ablation_v8_alpha_delta_sweep.json     # Sweep summary
└── critic_eval_v8_unseen_2000q.jsonl      # Critic step-level eval
```

## 6. W&B Tracking

- Project: `prmrag-ablation`
- Runs: `dpo_v8_{α}_{δ}_b0.1`
- URL: https://wandb.ai/dataana/prmrag-ablation

---

## 7. Next Steps (2026-04-20 업데이트)

### 완료
- [x] Eval pipeline bug 수정 (faiss_gpu=False + nprobe=128)
- [x] V1 재현 확인 (0.383)
- [x] K=3 모든 변형 재평가 (raw 39k, deduped 19k, sampled13k, δ=0.4)
- [x] V1-tree α=0 ablation (PRM 효과 증명, -9%p)
- [x] 데이터 품질 심층 분석 (chosen_f1, ΔF1, step 분포, critic 동의)
- [x] **F4 실패 분석 (cho_f1≥0.5만 → 0.358, -2.5%p)** — chosen floor 단일 축 위험
- [x] **F6 성공 (cho≥0.2 AND Δ≥0.2 → 0.387)** — 🎯 V1 baseline 초과
- [x] **F6 vs V1 per-question 분석**: F6 우세는 대부분 Yes/No(+4.2%p)에서, Entity는 거의 무차이
- [x] **α sweep under F6 분석**: F6 하에서 α=0.0 vs α=0.3 = **98.8% 동일 pair set** → α/δ는 실질 튜닝공간 없음

### 확정 방향
**최종 모델: K=3 F6 (cho_f1≥0.2 AND ΔF1≥0.2, 15,877 pairs, β=0.1)**
- HotpotQA pass@1 EM: **0.3873** (V1 +0.4%p, SearchR1 PPO +0.3%p)
- F1: **0.5163** (V1 +1.2%p)

### 논문 구조 (2026-04-20 재정립)

**Narrative (α → F6 정당화)**:
1. **Motivation — α 필요성 증명**: V1-tree에서 α=0 vs α=0.3 비교 → EM -9%p → PRM signal 필수
2. **Problem with implicit α mixing**: α가 pair ordering에 영향 주지만 explicit하지 않음
3. **Our solution — F6 filter**: explicit 2-axis F1 filter (chosen floor + margin floor)가 α implicit mixing을 대체 + 개선
4. **Ablation (F4 vs F6)**: chosen floor 단일 축은 실패 (overfit); chosen floor + margin floor 조합이 중요
5. **Analysis**: F6 개선이 binary decisions (Yes/No)에서 집중 → F6 filter creates cleaner binary learning signal

### 단기 플랜 (이번 주, pass@1 전용 — 논문 본체)

**Phase 0 — 다른 벤치마크 평가 (진행 중)**:
- [x] HotpotQA (7,405q) — F6+α0.3 = 0.3873
- [ ] popqa (14,267q) — 🔄 실행 중
- [ ] 2wikimultihopqa (12,576q)
- [ ] bamboogle (125q)
- [ ] musique (2,417q)
- **Baseline**: SearchR1 기존 결과 사용 (추가 학습 불필요)
- **예상**: ~3시간 남음

**Phase A — α sweep on F6 (2 runs, 8h 예상)**:
이미 완료: α=0 (0.367), α=0.3 (0.387)
- [ ] α=0.5 + F6 (base: mcts_dpo_v8_k3_a0.5_d0.05.jsonl → F6 filter)
- [ ] α=1.0 + F6 (base: mcts_dpo_v8_k3_a1.0_d0.05.jsonl → F6 filter)
- **목표**: 논문 2×4 ablation table 완성 (α × F6)

**Phase B — F6 threshold sweep (3 runs, ~12h 예상)**:
- [ ] F6-tight: (cho≥0.3, Δ≥0.2) — 순도↑, pair ~11k
- [ ] F6-margin: (cho≥0.2, Δ≥0.3) — 대비↑, pair ~13k
- [ ] F6-loose: (cho≥0.1, Δ≥0.2) — coverage↑, pair ~19k
- **목표**: 2-axis filter 최적값 + robustness 증명

**Phase C — δ 검증 1회 (~4h)**:
- [ ] δ=0.05 + α=0.3 + F6 → 0.387 ± 0.5%p 재현 검증만
- **목표**: F6 하에서 δ가 no-op임을 실증 (논문 appendix)

**Phase D — β sweep (best setup 확정 후, 2 runs, ~8h)**:
- [ ] β=0.05 on best
- [ ] β=0.15 on best
- **목표**: DPO temperature 최적값

**총 예상 (HotpotQA only)**: ~32h (Phase A+B+C+D) + 최종 best 모델 5 benchmark eval (~3h)

### 중기 플랜 (다음 주~, refine step 도입)

**Refine step 추가** — retrieval 성공(41%)이지만 answer 틀린 경우 해결 목표:

1. **Phase 1 — Prototype & Critic 검증 (1~2일)**:
   - [ ] Refine-enabled pipeline 프로토타입 (SearchR1Pipeline 수정, `<refine>` stop token 추가)
   - [ ] 100q에 refine step 포함 trajectory 생성
   - [ ] Judge LLM으로 refine step GOOD/BAD 라벨링
   - [ ] **V8 critic agreement 측정**:
     - \>75%: V8 그대로 사용 (전략 A)
     - 60~75%: LoRA 증분 학습 (전략 C, +1일)
     - <60%: Critic 전체 재학습 (전략 B, +2~3일)

2. **Phase 2 — MCTS 재빌드 (3일)**:
   - [ ] Refine step 포함 MCTS tree 생성 (5000q × 64 rollouts)
   - [ ] Refine step의 F1 reward 정의 (직접 답 생성 X → 간접 signal)
   - [ ] Pair 추출 (F6 filter 동일 적용)

3. **Phase 3 — DPO 학습 + eval (2일)**:
   - [ ] V1 하이퍼로 DPO 학습
   - [ ] 5개 벤치마크 평가
   - [ ] F6 단독 vs F6+refine 비교

**총 예상 시간**: 
- 전략 A (V8 그대로): 5~6일
- 전략 C (LoRA 증분): 6~7일 ⭐ 추천
- 전략 B (전체 재학습): 8~10일

### 병행 수정 (우선순위 중)

- [ ] **Critic validation (500q × 128 traj) F1/EM 버그 수정** — ABLATION_PLAN Section 0의 BoN 스케일링 실험에 f1/em 계산 버그 존재. 재측정 필요. (이후 critic 비교표 모두 재검증)

### 장기 플랜 (논문 submission 후)

- [ ] **Retrieval bottleneck 개선** (59% nobody correct 원인):
  - Query rewriting / reranker / 더 많은 top_k
  - **paper future work 섹션**
- [ ] **2Wiki/PopQA trajectory 재생성** — 답변 길이 131~136자 → F1 부당 (deferred)
- [ ] **Iterative DPO** — F6 모델로 MCTS 재생성 → v2 self-improvement
- [ ] **레거시 prompt 정리** — Variant B 제거

## 8. Prompt Consistency Audit (2026-04-15)

### Variant A ("So the answer is <answer>answer</answer>") — 현재 메인
- ✅ `src/prmrag/models/policy_model_vllm.py` (inference)
- ✅ `src/prmrag/training/kto_trainer.py`
- ✅ `scripts/build_mcts_dpo.py` (MCTS build)
- ✅ `scripts/eval_flashrag.py`, `scripts/eval_all_datasets.py`
- ✅ `scripts/eval_policy.py` (policy_model_vllm 경유)

### Variant B ("directly provide a concise final answer using `<answer>`... without detailed illustrations") — 레거시
- ⚠️ `src/prmrag/regeneration/regenerator.py`
- ⚠️ `scripts/build_step_level_dpo.py`
- ⚠️ `scripts/build_dpo_dataset.py`

**현재 pipeline (MCTS → DPO → eval)는 Variant A로 통일됨.** 레거시 Variant B는 현재 실험에 사용되지 않음.

### PRM Scaling Trajectory Format 이슈
- HotpotQA: avg 12자 (96% <30자) — policy가 prompt 지시 잘 따름
- 2Wiki: avg 131자 (54% >100자) — policy가 `<answer>` 태그 안에 장문 설명 출력
- PopQA: avg 136자 (49% >100자) — 동일 문제

**영향**: F1 계산 시 gold와 토큰 겹침이 적어서 2Wiki/PopQA F1이 비정상적으로 낮음. **Trajectory 재생성 필요** (deferred task).
