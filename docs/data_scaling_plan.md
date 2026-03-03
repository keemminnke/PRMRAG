# Data Scaling Experiment Plan

## 목표
데이터를 점진적으로 추가하면서 성능 변화를 관찰하여 최적의 데이터 구성을 찾는다.

## 현재 상태 (Baseline)

| 항목 | 값 |
|------|-----|
| 학습 데이터 | HotpotQA 1,000q + MusiQue 1,000q = **2,000q** |
| Trajectory | 질문당 16개 path (총 32,000 trajectories) |
| Critic model | `outputs/critic_model_v8_2000q` (2,000q로 학습) |
| SFT data | all-GOOD trajectories 10,871개 |
| DPO pairs | 13,467 pairs (1,543 questions, regen 126q 포함) |
| **평가 결과** | DPO v1: EM 34.83%, F1 46.10%, Cover EM 57.46% |
| **Base 대비** | EM +5.17%, F1 +5.27%, Cover EM +5.32% |

### 현재 파일 경로 (정리 후)

```
# Trajectory 데이터 (judge label 포함)
outputs/trajectories/hotpotqa_1000q_v1.jsonl   (1,000q × 16 = 15,728 trajs, SFT model 생성)
outputs/trajectories/musique_1000q_v1.jsonl    (1,000q × 16 = 16,000 trajs, SFT model 생성)

# Critic model
outputs/critic_model_v8_2000q/

# DPO 학습 데이터
outputs/training_data/dpo_dataset_v1_bva_messages.jsonl  (13,467 pairs)
outputs/training_data/dpo_regen_scored.jsonl              (regen 데이터)

# Policy models
outputs/sft_policy_v1/merged_model      (SFT LoRA merged)
outputs/dpo_policy_v1/final_model       (DPO LoRA on SFT)

# Raw questions
data/raw/questions/hotpotqa_train.jsonl      (90,447q)  ← 1,000q 사용됨
data/raw/questions/musique_train.jsonl       (19,938q)  ← 1,000q 사용됨

# 평가 결과
outputs/eval/flashrag_dpo_bva_full/     (DPO v1 full eval)
outputs/eval/flashrag_qwen_base_full/   (Base full eval)
outputs/eval/flashrag_full_comparison.json
```

---

## 실험 계획

### Round 1: HotpotQA +1,000q (2,000q → 3,000q)

기존 2,000q에 HotpotQA 1,000q를 추가. 동일 도메인 데이터 증가 효과 확인.

### Round 2: MusiQue +1,000q (3,000q → 4,000q)

다른 도메인 데이터 추가 효과 확인.

### Round 3: 2WikiMultihopQA +1,000q (4,000q → 5,000q)

새로운 도메인 데이터 추가. (데이터 준비 필요)

---

## Round 1 상세 파이프라인

### Step 1. 새 질문 1,000개 샘플링

기존 HotpotQA 1,000q와 **겹치지 않는** 새 1,000q를 `hotpotqa_train.jsonl`에서 추출.

```bash
python scripts/sample_new_questions.py \
    --source data/raw/questions/hotpotqa_train.jsonl \
    --existing outputs/trajectories/hotpotqa_1000q_v1.jsonl \
    --output data/raw/questions/hotpotqa_train_round2.jsonl \
    --num 1000 \
    --seed 42
```

**출력**: `data/raw/questions/hotpotqa_train_round2.jsonl` (1,000q)

### Step 2. Trajectory 생성 (16 paths per question)

**Qwen2.5-7B-Instruct** base model로 16개 trajectory 생성. **BGE-M3 + Reranker** 사용.
(SFT 모델 아님 — Qwen base에서 직접 생성)

```bash
python scripts/generate_trajectories.py \
    --data_path data/raw/questions/hotpotqa_train_round2.jsonl \
    --output_path outputs/hotpotqa_round2_trajectories.jsonl \
    --num_samples 16 \
    --retriever bge \
    --batch_size 500 \
    --policy-model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.85 \
    --gpu-num 2
```

**출력**: `outputs/hotpotqa_round2_trajectories.jsonl` (~16,000 trajs)
**예상 시간**: ~4-6시간

### Step 3. QwQ-32B Judge Labeling

**QwQ-32B** 모델로 새 trajectories의 각 step에 대해 **judge_label (GOOD/BAD)** + **judge_reasoning** 생성.
이것이 critic model 학습의 **ground truth** 데이터가 된다.

```bash
python scripts/judge_label_qwq.py \
    --input outputs/hotpotqa_round2_trajectories.jsonl \
    --output outputs/hotpotqa_round2_judge_labeled.jsonl \
    --batch-size 500 \
    --gpu-num 2 \
    --resume
```

**출력**: `outputs/hotpotqa_round2_judge_labeled.jsonl`
- 각 trajectory의 각 step에 `judge_label` (GOOD/BAD) + `judge_reasoning` 포함
- QwQ-32B가 step의 reasoning 품질, 검색 쿼리 적절성, 답변 정확성 등을 평가

**예상 시간**: ~6-8시간 (QwQ-32B은 32B 모델이라 느림)

### Step 4. Critic Model 재학습 (3,000q)

기존 2,000q judge 데이터 + 새 1,000q judge 데이터 = **3,000q**로 critic 재학습.
Critic model은 QwQ-32B의 judge 판단을 distill하여 빠르게 step-level 평가하는 경량 모델.

**학습 데이터 구성:**
- `outputs/training_data/judge_labels_2000q_v1.jsonl` — 기존 2,000q (HotpotQA 1,000q + MusiQue 1,000q, 31,728 trajs, judge_label+judge_reasoning 포함)
- `outputs/hotpotqa_round2_judge_labeled.jsonl` — **새로운** HotpotQA 1,000q (Step 3에서 QwQ-32B로 라벨링)

```bash
torchrun --nproc_per_node=2 scripts/train_critic_model.py \
    --train-data outputs/training_data/judge_labels_2000q_v1.jsonl \
    --train-data outputs/hotpotqa_round2_judge_labeled.jsonl \
    --output-dir outputs/critic_model_v9_3000q \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --num-epochs 1
```

**출력**: `outputs/critic_model_v9_3000q/`
**예상 시간**: ~3-4시간

### Step 5. 새 Critic으로 전체 Re-scoring → 통합 파일 생성

재학습된 critic (v9)으로 전체 3,000q trajectories 재평가 → **하나의 통합 파일**로 출력.
이후 Step 6~8 모두 이 통합 파일 하나만 사용.

```bash
python scripts/evaluate_with_critic.py \
    --trajectories outputs/trajectories/hotpotqa_1000q_v1.jsonl \
    --trajectories outputs/trajectories/musique_1000q_v1.jsonl \
    --trajectories outputs/hotpotqa_round2_trajectories.jsonl \
    --critic-model outputs/critic_model_v9_3000q \
    --output outputs/all_3000q_critic_v9_scored.jsonl \
    --gpu-num 2
```

**출력**: `outputs/all_3000q_critic_v9_scored.jsonl` (3,000q 전체 통합, critic step labels 포함)
**예상 시간**: ~4-5시간

### Step 6. Regeneration + DPO/SFT 데이터 구축

통합 파일에서 all-GOOD trajectory 없는 질문 → Regenerate → DPO pairs + SFT 데이터 구축.

```bash
python scripts/build_dpo_dataset.py \
    --input outputs/all_3000q_critic_v9_scored.jsonl \
    --regen-out outputs/dpo_regen_v2_trajectories.jsonl \
    --scored-out outputs/all_3000q_with_regen.jsonl \
    --dpo-out outputs/dpo_dataset_v2_bva_messages.jsonl \
    --policy-model Qwen/Qwen2.5-7B-Instruct \
    --gpu-num 2
```

**출력**:
- `outputs/all_3000q_with_regen.jsonl` — 통합 scored 데이터 (regen 포함)
- `outputs/dpo_dataset_v2_bva_messages.jsonl` — DPO pairs (best-vs-all, ~20,000+ 예상)
- SFT data: all-GOOD trajectories (~15,000+ 예상)
**예상 시간**: ~3-4시간

### Step 7. SFT 학습

Qwen2.5-7B-Instruct 위에 SFT LoRA 학습.

```bash
torchrun --nproc_per_node=2 scripts/train_sft_policy.py \
    --input outputs/all_3000q_with_regen.jsonl \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-dir outputs/sft_policy_v2 \
    --epoch 1 \
    --wandb-run-name sft_v2_3000q
```

**출력**: `outputs/sft_policy_v2/merged_model`
**예상 시간**: ~2-3시간

### Step 8. DPO 학습

SFT v2 위에 DPO LoRA 학습.

```bash
torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
    --dpo-dataset outputs/dpo_dataset_v2_bva_messages.jsonl \
    --model-name outputs/sft_policy_v2/merged_model \
    --output-dir outputs/dpo_policy_v2 \
    --wandb-run-name dpo_v2_3000q \
    --epoch 3
```

**출력**: `outputs/dpo_policy_v2/final_model`
**예상 시간**: ~13시간

### Step 9. 평가

FlashRAG로 HotpotQA 전체 7,405q 평가. Base / DPO v1 / DPO v2 비교.

```bash
python scripts/eval_flashrag.py \
    --model outputs/sft_policy_v2/merged_model \
    --lora outputs/dpo_policy_v2/final_model \
    --tag dpo_v2_3000q_full \
    --top-k 3 --limit 7405 --temperature 0.8
```

**출력**: `outputs/eval/flashrag_dpo_v2_3000q_full/`

---

## 예상 소요 시간

| Step | 작업 | 예상 시간 |
|------|------|----------|
| 1 | 질문 샘플링 | ~1분 |
| 2 | Trajectory 생성 (1,000q × 16, GPU×2) | 4-6시간 |
| 3 | QwQ-32B Judge Labeling (GPU×2) | 6-8시간 |
| 4 | Critic 재학습 (3,000q, GPU×2) | 3-4시간 |
| 5 | 전체 Re-scoring → 통합 파일 (GPU×2) | 4-5시간 |
| 6 | Regen + 데이터 구축 (GPU×2) | 3-4시간 |
| 7 | SFT 학습 (GPU×2) | 2-3시간 |
| 8 | DPO 학습 (GPU×2) | ~13시간 |
| 9 | 평가 (GPU×2) | ~1시간 |
| **Total** | | **~36-44시간** |

---

## 기대 결과

| Model | EM | F1 | Cover EM |
|-------|---:|---:|--------:|
| Qwen Base | 29.66% | 40.83% | 52.14% |
| DPO v1 (2,000q) | 34.83% | 46.10% | 57.46% |
| **DPO v2 (3,000q)** | **?** | **?** | **?** |

핵심 가설:
1. 동일 도메인(HotpotQA) 데이터 증가 → critic 품질 향상 → 더 좋은 SFT/DPO 데이터
2. Regeneration 대상 질문 증가 → chosen pool 확대 → DPO pair 수 증가
3. SFT를 Qwen base 위에서 제대로 학습 → DPO도 SFT 위에서 학습 → on-policy 효과

---

## 주요 원칙

- **SFT는 반드시 `Qwen/Qwen2.5-7B-Instruct` 위에 학습** (잘못된 SFT 모델 사용 금지)
- **DPO는 반드시 SFT merged model 위에 학습** (on-policy 보장)
- **GPU 2개 항상 사용** (vLLM TP=2, DDP nproc=2)
- **평가: temp=0.8, gpu_util=0.85, e5-base, top_k=3**
- **겹치지 않는 질문 사용** (question ID 기반 dedup)

---

## 파일 네이밍 규칙

```
Round 1 (3,000q):
  data/raw/questions/hotpotqa_train_round2.jsonl
  outputs/hotpotqa_round2_trajectories.jsonl
  outputs/hotpotqa_round2_critic_scored.jsonl
  outputs/critic_model_v9_3000q/
  outputs/sft_policy_v2/
  outputs/dpo_policy_v2/

Round 2 (4,000q):
  data/raw/questions/musique_train_round2.jsonl
  outputs/musique_round2_trajectories.jsonl
  outputs/critic_model_v10_4000q/
  outputs/sft_policy_v3/
  outputs/dpo_policy_v3/

Round 3 (5,000q):
  data/raw/questions/2wikimultihop_train.jsonl
  outputs/2wiki_round1_trajectories.jsonl
  outputs/critic_model_v11_5000q/
  outputs/sft_policy_v4/
  outputs/dpo_policy_v4/
```
