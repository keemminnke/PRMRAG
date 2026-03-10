# Data Scaling Experiment Plan

## 목표
데이터를 점진적으로 추가하면서 성능 변화를 관찰하여 최적의 데이터 구성을 찾는다.

---

## 주요 원칙

- **Policy base**: 항상 `Qwen/Qwen2.5-7B-Instruct` 사용
- **Critic base**: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` + LoRA
- **System prompt 통일**: SFT / DPO / eval 모두 동일한 prompt 사용 (kto_trainer.py의 SYSTEM_PROMPT)
- **SFT**: Qwen base 위에 LoRA 학습 → merge → DPO base로 사용
- **DPO**: SFT merged model 위에 LoRA 학습, **messages format (list[dict])**, **Best-vs-All** pairing
- **SFT 입력**: 원본 scored + regen scored 합친 전체 데이터 (all-GOOD만 학습)
- **불량 trajectory 필터링**: final answer 없는 것, max_step(10) 도달하여 잘린 것은 제외
- **GPU 2개 항상 사용** (vLLM TP=2 또는 DP=2, torchrun nproc=2)
- **평가**: FlashRAG, temp=0.8, gpu_util=0.85, top_k=3, limit=7405
- **Re-scoring**: `compare_voting_methods.py` 사용 (`evaluate_with_critic.py` 사용 금지 — 535시간)

---

## 실험 결과 요약

| # | Model | EM | F1 | 데이터 | 비고 |
|---|-------|-----|-----|--------|------|
| 0 | Qwen Base | 29.7% | 40.8% | - | 기준선 |
| 0.5 | DPO only (SFT 없이) | 29.5% | 41.2% | 2,000q | 효과 없음 |
| 1 | SFT v1 | 34.9% | 46.0% | 2,000q | SFT만 |
| 2 | SFT v1 + DPO v1 (ep3) | 34.8% | 46.1% | 2,000q | DPO 효과 없음 |
| 3 | SFT v2 | 34.5% | 45.7% | 3,000q | 데이터 늘려도 개선 없음 |
| 4 | SFT v2 + DPO v2 (ep3) | 34.4% | 45.9% | 3,000q | DPO 과적합 (acc 82.8%) |
| 5 | SFT v2 + DPO v2 (ep1, prompt 불일치) | - | - | 3,000q | 중단 |
| 6 | SFT v2 + DPO v2 (ep1, prompt 통일) | 34.2% | 45.5% | 3,000q | DPO 효과 없음 |

### 핵심 관찰
- **SFT만으로 base 대비 +5%p EM 개선** (v1, v2 모두)
- **DPO는 어떤 설정에서도 추가 효과 없음** (ep1/ep3, prompt 통일/불일치 무관)
- **데이터 2,000q → 3,000q 확장해도 성능 정체**
- **System prompt 불일치**: Round 1에서 SFT/DPO/eval 각각 다른 prompt 사용 → Round 2+3에서 통일

---

## 실험 계획

### Round 1: HotpotQA +1,000q (2,000q → 3,000q) — ✅ 완료

기존 2,000q에 HotpotQA 1,000q를 추가.

### Round 2+3: 데이터 한번에 생성, 학습만 다르게 실험 — 진행 중

- **Round 2**: MusiQue +1,000q (3,000q → 4,000q)
- **Round 3**: 2WikiMultihopQA +1,000q (4,000q → 5,000q)

**데이터 생성**: `scripts/run_round2_3_data.sh`
**학습 실험**: `scripts/run_training_experiments.sh`

#### 데이터 생성 파이프라인 (한번에)
```
Step 1: 질문 샘플링 (MusiQue 1,000q + 2Wiki 1,000q)
Step 2: Trajectory 생성 (각 1,000q × 16)
Step 3: QwQ-32B Judge Labeling
Step 4: Critic 재학습 (5,000q 전체)
Step 5: 전체 Re-scoring (5,000q, vLLM batch)
Step 5.5: 불량 trajectory 필터링 (no answer, max_step>=10)
Step 6: Regeneration + DPO 데이터 구축
Step 6.5: SFT 입력 준비 (4,000q / 5,000q)
Step 6.6: 4,000q DPO 데이터 (2Wiki 제외)
```

#### 학습 실험 (데이터 고정)
| Exp | 설정 | 데이터 | DPO | 목적 |
|-----|------|--------|-----|------|
| 7 | SFT only | 4,000q | - | 데이터 스케일링 |
| 8 | SFT only | 5,000q | - | 데이터 스케일링 |
| 9 | SFT + DPO ep1 | 5,000q | 1 | DPO + prompt 통일 |
| 10 | SFT + DPO ep3 | 5,000q | 3 | DPO epoch 비교 |
| 11 | SFT + DPO ep1 | 4,000q | 1 | 데이터 × DPO 조합 |

---

## 파이프라인 아키텍처

```
Step 1: 질문 샘플링
Step 2: Trajectory 생성 (Qwen base, BGE+Reranker, 16 paths)
Step 3: QwQ-32B Judge Labeling (critic 학습용 ground truth)
Step 4: Critic Model 재학습 (기존 + 새 judge labels)
Step 5: 전체 Re-scoring (vLLM batch, compare_voting_methods.py)
Step 5.5: 불량 trajectory 필터링
Step 6: Regeneration + DPO 데이터 구축 (Best-vs-All)
Step 7: SFT 학습 + LoRA merge
Step 8: DPO 학습 (SFT merged model 위에)
Step 9: FlashRAG 평가
```

- 각 Step마다 `.done` 마커 파일로 체크포인트
- 중간에 중단되어도 완료된 Step은 SKIP하고 재개

### 자동화 스크립트

```bash
# Round 1 (완료)
nohup bash scripts/run_round1_pipeline.sh > logs/round1_pipeline.log 2>&1 &

# Round 2+3 데이터 생성
nohup bash scripts/run_round2_3_data.sh > logs/round2_3_data.log 2>&1 &

# 학습 실험 (개별 또는 전체)
bash scripts/run_training_experiments.sh exp7
bash scripts/run_training_experiments.sh all
```

---

## Round 2+3 예상 시간

| Step | 작업 | 시간 |
|------|------|------|
| 1 | 질문 샘플링 (2,000q) | ~1분 |
| 2 | Trajectory 생성 (2,000q × 16) | ~8시간 |
| 3 | QwQ-32B Judge Labeling (2,000q) | ~20시간 |
| 4 | Critic 재학습 (5,000q) | ~5시간 |
| 5 | 전체 Re-scoring (5,000q, ~80K trajs) | ~8시간 |
| 5.5 | 불량 trajectory 필터링 | ~1분 |
| 6 | Regen + DPO 데이터 구축 | ~6시간 |
| **Total (데이터)** | | **~47시간** |
| 7~9 | 학습 + 평가 (per experiment) | ~6시간 |

---

## 데이터 통계

| 항목 | v1 (2,000q) | v2 (3,000q) | v3 (4,000q) | v4 (5,000q) |
|------|-------------|-------------|-------------|-------------|
| 질문 구성 | HotpotQA 1K + MusiQue 1K | + HotpotQA 1K | + MusiQue 1K | + 2Wiki 1K |
| 원본 trajectories | 31,728 | 47,728 | ~63,728 | ~79,728 |
| DPO pairs (Best-vs-All) | 13,467 | 19,209 | ? | ? |

---

## 파일 경로 정리

### 공통 데이터

```
# Raw questions
data/raw/questions/hotpotqa_train.jsonl          (90,447q)
data/raw/questions/musique_train.jsonl           (19,938q)
data/raw/questions/2wikimultihop_train.jsonl     (167,454q)

# Baseline trajectories (judge label 포함)
outputs/trajectories/hotpotqa_1000q_v1.jsonl     (15,728 trajs)
outputs/trajectories/musique_1000q_v1.jsonl      (16,000 trajs)

# Baseline judge labels (critic 학습용)
outputs/training_data/judge_labels_2000q_v1.jsonl  (31,728 trajs)

# HF cache
/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface

# Retriever
data/embeddings/kilt_wikipedia_bge_m3.npy        (12GB)
data/indexes/kilt_wikipedia_bge_m3.faiss         (23GB)
data/kilt/kilt_knowledgesource_parsed.pkl        (5.9M docs)
```

### Round 1 (3,000q) — 완료

```
data/raw/questions/hotpotqa_train_round2.jsonl   (1,000q)
outputs/hotpotqa_round2_trajectories.jsonl       (16,000 trajs)
outputs/hotpotqa_round2_judge_labeled.jsonl      (16,000 trajs)
outputs/training_data/judge_labels_3000q_merged.jsonl  (47,728 trajs)
outputs/critic_model_v9_3000q/final_model
outputs/all_3000q_critic_v9_scored.jsonl         (47,728 trajs)
outputs/dpo_dataset_v2_bva_messages.jsonl        (19,209 pairs)
outputs/dpo_dataset_v2_bva_matched.jsonl         (19,209 pairs, prompt 통일)
outputs/sft_policy_v2/merged_model
outputs/dpo_policy_v2/final_model                (DPO ep3)
outputs/dpo_policy_v2_ep1/final_model            (DPO ep1, prompt 통일)
```

### Round 2 (4,000q) — 진행 중

```
data/raw/questions/musique_train_round2.jsonl    (1,000q)
outputs/musique_round2_trajectories.jsonl
outputs/musique_round2_judge_labeled.jsonl
outputs/all_4000q_critic_scored.jsonl
outputs/all_4000q_sft_input.jsonl
outputs/dpo_dataset_v3_4000q_bva.jsonl
outputs/sft_policy_v3_4000q/
outputs/dpo_policy_v3_4000q_ep1/
```

### Round 3 (5,000q) — 진행 중

```
data/raw/questions/2wikimultihop_train_round1.jsonl  (1,000q)
outputs/2wiki_round1_trajectories.jsonl
outputs/2wiki_round1_judge_labeled.jsonl
outputs/training_data/judge_labels_5000q_merged.jsonl
outputs/critic_model_v11_5000q/
outputs/all_5000q_merged_trajectories.jsonl
outputs/all_5000q_critic_v11_scored.jsonl
outputs/dpo_regen_v4_trajectories.jsonl
outputs/all_5000q_with_regen.jsonl
outputs/dpo_dataset_v4_5000q_bva.jsonl
outputs/all_5000q_sft_input.jsonl
outputs/sft_policy_v4_5000q/
outputs/dpo_policy_v4_5000q_ep1/
```

---

## 삽질 기록 (Lessons Learned)

### System Prompt 불일치 (중요!)
- Round 1에서 SFT/DPO/eval 각각 다른 system prompt 사용 → DPO 효과 측정 불가
- SFT: `"You are an advanced AI agent capable of Adaptive RAG..."` (kto_trainer.py)
- DPO: `"You are a helpful assistant..."` (build_dpo_dataset.py) ← **수정 완료**
- eval: `"You are a multi-step reasoning assistant..."` (eval_flashrag.py) ← **수정 완료**
- **Round 2+3부터 전부 kto_trainer.py의 SYSTEM_PROMPT로 통일**

### 스크립트 인자
- `generate_trajectories.py`: underscore args (`--policy_model`), `--limit` 기본값 500
- `train_critic_model.py`: `--train-data`는 **단일 파일만** → `cat`으로 합치기
- `train_dpo_policy.py`: 상대 경로 → `os.path.abspath()` 추가됨
- `build_dpo_dataset.py`: `--hotpotqa`/`--musique` args, 사용 안 하는 쪽은 `/dev/null`
- `sample_new_questions.py`: `--existing` 필수 — 기존 데이터 없으면 Python 직접 샘플링

### 학습
- `device_map='auto'`는 torchrun DDP와 호환 안 됨 → 제거
- SFT 출력은 `final_model` (LoRA) → merge 별도 필요
- DPO 데이터: **messages format (list[dict])** 필수 — text format은 tokenization mismatch
- DPO epoch 3 → rewards accuracy 82.8%, 과적합 → epoch 1 권장
- DPO 어떤 설정에서도 SFT 이상 성능 개선 없음 (현재까지)

### Re-scoring
- `evaluate_with_critic.py`: 535시간 → **사용 금지**
- `compare_voting_methods.py`: vLLM batch, ~5시간 → 이것 사용

### 데이터 흐름
- `scored_out`: **regen scored만** 저장 (원본 미포함)
- SFT 입력: 반드시 `$SCORED_ALL` + `$SCORED_WITH_REGEN` cat
- eval_flashrag.py 출력: `outputs/flashrag_{tag}/` (NOT `outputs/eval/`)

### vLLM 프로세스
- 메인 프로세스 kill 후에도 EngineCore/Worker 남을 수 있음
- `pkill -f "vllm"` 또는 `/proc/*/fd/*`에서 nvidia device 검색

