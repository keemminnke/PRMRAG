# PRMRAG Project Progress Report

## Last Updated: 2026-02-21

## Current Status: BAD Step Regeneration + KTO Pipeline

Critic model(v8)이 BAD으로 판정한 step을 재생성하고, critic으로 재평가한 뒤, 전체 데이터를 합쳐 KTO로 policy model을 fine-tuning하는 파이프라인.

---

## Completed Tasks

### 1. Critic Model Scoring & Reasoning (v8)
- [x] 15,728 trajectories (1,000 questions x ~16 samples) critic v8으로 스코어링 완료
- [x] `outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl` 생성
  - step별 `critic_label` (0=BAD, 1=GOOD), `critic_score`, `critic_reasoning` 포함
  - 47,032 steps 전부 critic_reasoning 보유
- [x] 데이터 분포: All-GOOD 9,336 (59.4%) / Has BAD 6,391 (40.6%)

### 2. Self-Correction Distillation (Pilot)
- [x] `scripts/generate_self_correction_data.py` 개발
- [x] 500 samples 파일럿 완료 (`outputs/self_correction_hotpot_500.jsonl`)

### 3. DPO Trainer 구현 (보류)
- [x] `src/prmrag/training/dpo_trainer.py` — document masking DPO 구현
- [x] `scripts/train_dpo_policy.py` — CLI 스크립트
- [ ] **보류**: All-BAD trajectory가 18개(0.1%)뿐이라 DPO 쌍 생성 불가. KTO로 전환

---

## In-Progress Pipeline

### Step 1: BAD Step 재생성 (Priority: High)
- [ ] `scripts/regenerate_bad_steps.py` 구현 및 실행
- **입력**: `outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl`
- **대상**: BAD step이 있는 6,391 trajectories
- **방법**: 첫 BAD step에서 truncate → critic_reasoning 활용 → GenPRM multi-turn 재생성
  - 기존 `TrajectoryTruncator` + `CriticGuidedRegenerator` 재사용
  - Policy model (Qwen2.5-7B) + BGE retriever + KILT corpus
- **출력**: `outputs/regenerated_trajectories.jsonl`

### Step 2: Critic 재평가 (After Step 1)
- [ ] `scripts/score_regenerated.py` 구현 및 실행
- **입력**: `outputs/regenerated_trajectories.jsonl`
- **방법**: Critic v8 model로 재생성된 trajectory의 각 step 재평가
  - step별 `critic_label`, `critic_score`, `critic_reasoning` 부여
- **출력**: `outputs/regenerated_critic_scored.jsonl`

### Step 3: KTO Fine-tuning (After Step 2)
- [ ] KTO 학습 데이터 구성 및 학습
- **학습 데이터 (3가지 합산)**:

| Source | Count | Step Labels |
|--------|-------|-------------|
| All-GOOD 원본 trajectories | 9,336 | 전부 GOOD |
| 재생성 + critic 재평가 trajectories | ~6,391 | critic 평가 결과 반영 |
| 원본 BAD trajectories | 6,391 | 기존 GOOD/BAD 혼합 |
| **합계** | **~22,000** | |

- **방법**: `StepLevelKTOTrainer` (document masking + dynamic lambda_U)
- **출력**: `outputs/kto_policy_v2/final_model`

### Step 4: 평가
- [ ] Fine-tuned model로 RAG trajectory 생성
- [ ] Base model 대비 accuracy 비교

---

## Key Data Files

| File | Description |
|------|-------------|
| `outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl` | 원본 15,728 trajectories + critic scores + reasoning |
| `outputs/regenerated_trajectories.jsonl` | Step 1 출력 (재생성된 trajectories) |
| `outputs/regenerated_critic_scored.jsonl` | Step 2 출력 (재평가된 trajectories) |
| `outputs/kto_policy_v2/final_model` | Step 3 출력 (fine-tuned LoRA adapter) |

---

## How to Run

```bash
# Step 1: BAD step 재생성
python scripts/regenerate_bad_steps.py \
    --input outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
    --output outputs/regenerated_trajectories.jsonl

# Step 2: Critic 재평가
python scripts/score_regenerated.py \
    --input outputs/regenerated_trajectories.jsonl \
    --output outputs/regenerated_critic_scored.jsonl

# Step 3: KTO 학습 (combined format: --input can be repeated)
python scripts/train_kto_policy.py \
    --input outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
    --input outputs/regenerated_critic_scored.jsonl \
    --output-dir outputs/kto_policy_v2 \
    --truncate-after-bad

# Step 4: 평가
python scripts/generate_trajectories.py \
    --lora_adapter outputs/kto_policy_v2/final_model ...
```
