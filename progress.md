# PRMRAG Training — Progress Log

Last updated: 2026-02-27

---

## 현재 상태 (브랜치: no-rpe)

### 전략 변경: KTO → DPO

KTO z0 self-reference 버그 수정 완료했으나, 학습 방향을 **DPO**로 전환.
기존 N=16 critic-scored 궤적을 활용해 DPO 데이터셋 구성.

### 핵심 구현 완료
- **KTO z0 버그 수정**: `src/prmrag/training/kto_trainer.py` (lagged EMA로 교체)
- **DPO trainer**: `src/prmrag/training/dpo_trainer.py`
- **DPO 데이터셋 빌더**: `scripts/build_dpo_dataset.py` ← 신규
- **Eval 궤적 재생성 파이프라인**: `scripts/regenerate_val_trajectories.py`

### 현재 실행 중 (2026-02-27)
```bash
# DPO 재생성 파이프라인 (PID 483102)
python scripts/build_dpo_dataset.py --regen-only \
    --policy-model outputs/sft_policy_v1/merged_model \
    --critic-gpu 0.80 --policy-gpu 0.80
# 로그: logs/build_dpo.log
```

### 다음 할 일
1. **재생성 완료 대기** → `outputs/dpo_regen_scored.jsonl`
2. **DPO 데이터셋 최종 빌드**:
   ```bash
   python scripts/build_dpo_dataset.py --build-only
   ```
3. **DPO 학습 실행**: `scripts/train_kto_policy.py` → DPO 학습 스크립트 작성
4. 평가

---

## DPO 데이터셋 구성 전략

### 페어링 방식
- **Chosen**: all-GOOD 궤적 (모든 critic_step_label == 1)
- **Rejected**: has-BAD 궤적 (any critic_step_label == 0)
- **전략**: all-vs-all (M chosen × N rejected 모든 조합)

### 데이터 통계 (재생성 전)

| 데이터셋 | 궤적 수 | all-GOOD | has-BAD | DPO 가능 질문 |
|---|---|---|---|---|
| hotpotqa | 15,728 | 9,336 (59.4%) | 6,392 (40.6%) | 707 / 1000 |
| musique  | 16,000 | 6,124 (38.3%) | 9,876 (61.7%) | 710 / 1000 |
| **합계** | 31,728 | 15,460 (48.7%) | 16,268 (51.3%) | **1,417 / 2,000** |

- 기존 all-vs-all 페어: **57,052개**
- DPO 불가 583개 질문 중: 302개는 all-GOOD 없음 (재생성 대상), 281개는 has-BAD 없음

### 재생성으로 Chosen 풀 확장
0 all-GOOD인 302개 질문의 has-BAD 궤적 4,811개를 재생성:
```
has-BAD: [step1✓, step2✗, step3✗]
    → truncate at first BAD (step2)
    → regenerate with critic_reasoning feedback
    → (if all-GOOD) new chosen trajectory
```

**기대 효과**: DPO 가능 질문 1,417 → 최대 1,719개로 확대

---

## 구현 세부 내용

### kto_trainer.py 주요 클래스

| 클래스 | 역할 |
|---|---|
| `KTODataPreparer` | trajectory + critic scores → Dataset |
| `StepLevelKTOTrainer` | step-level KTO loss (Trainer 서브클래스) |
| `KTODebugCallback` | 학습 중 loss/KL 출력 |
| `EarlyCollapseCallback` | collapse 감지 → 자동 조기 종료 ← 신규 |

### EarlyCollapseCallback 기준
- `bad_loss < 0.05` (step 30 이후) → collapse
- `KL < -5.0` (step 30 이후) → collapse
- → `control.should_training_stop = True` 자동 설정

### train_kto_policy.py 새 옵션
- `--max-steps N`: N 옵티마이저 스텝 후 종료 (sweep probe용)
- `--sweep-result-file PATH`: 결과 JSON 저장 후 모델 저장 생략

### sweep_kto.py 동작
- Grid: `beta=[0.05, 0.1, 0.2]` × `lambda0=[1.0, 1.5, 2.0, 3.0]` = 12 조합
- 각 probe: 50 step, 500 trajectories, collapse 감지 시 즉시 종료
- 결과 테이블 출력 후 best config로 풀 학습 가능 (`--run-winner`)
- 중간 결과: `outputs/kto_sweep_tmp/sweep_YYYYMMDD_HHMMSS/sweep_results.json`

---

## KTO Loss 수식 (step-level)

```
GOOD step: loss = 1 - σ(β * (logratio - z0))   × desirable_weight
BAD  step: loss = 1 - σ(β * (z0 - logratio))   × lambda0

logratio = mean(log π_θ(t) - log π_ref(t))  per token, document 제외
logratio = clamp(logratio, -10, 10)          collapse 방지

z0 = max(0, mean(batch logratios))           논문 수식, 음수 방지
```

---

## 데이터 통계

| 파일 | trajectories | GOOD steps | BAD steps | nD/nU |
|---|---|---|---|---|
| `hotpotqa_critic_results_v8_per_trajectory.jsonl` | 15,728 | 36,422 | 10,610 | 3.43 |
| `original_filtered_for_kto.jsonl` | 11,261 | 27,067 | 2,582 | 10.49 |

**현재 학습 데이터**: `hotpotqa_critic_results_v8_per_trajectory.jsonl`
**critic_score 특성**: 거의 이진 (GOOD≈1.0, BAD≈0.0), 중간값 없음

---

## 실패한 실험 기록

| 실험 | beta | lambda0 | 결과 | 원인 |
|---|---|---|---|---|
| KTO v1 | 0.3 | 3.43 | collapse (step ~200) | BAD loss 과다 지배 |
| KTO v2 | 0.1 | 1.0 | 학습 안됨 (loss 고착) | BAD 신호 너무 약함 |
| KTO v3 | 0.3 | 10.9 | collapse (step ~240) | KL=-12.67, logratio 발산 |

**Logratio clipping ±10** 추가 후 아직 재시도 안 함 → sweep으로 탐색 예정

---

## GPU / 환경

- GPU: A100 SXM 79GB x2
- batch=2, grad_accum=16 (effective batch=32)
- OOM: batch=4 (logits ~9.28GB)
- 모델: Qwen/Qwen2.5-7B-Instruct, bfloat16, Flash Attention 2
- LoRA: r=16~32, alpha=16~64
- HF cache: `/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface`

---

## 파일 구조 (학습 관련)

```
scripts/
  train_kto_policy.py              # KTO 학습 진입점
  sweep_kto.py                     # KTO 자동 하이퍼파라미터 탐색
  build_dpo_dataset.py             # DPO 데이터셋 빌더 (regen 포함) ← 신규
  regenerate_val_trajectories.py   # Eval 궤적 재생성 파이프라인

src/prmrag/training/
  kto_trainer.py           # KTO 핵심 구현 (z0 lagged EMA 수정됨)
  dpo_trainer.py           # DPO 핵심 구현

outputs/
  hotpotqa_critic_results_v8_per_trajectory.jsonl  # hotpotqa 학습 데이터 (critic 채점 완료)
  musique_critic_results_v8_per_trajectory.jsonl   # musique 학습 데이터 (critic 채점 완료)
  sft_policy_v1/merged_model                       # SFT 정책 모델 (재생성용)
  dpo_regen_trajectories.jsonl                     # 재생성된 궤적 (생성 중)
  dpo_regen_scored.jsonl                           # 재생성 궤적 critic 채점 결과
  dpo_dataset.jsonl                                # 최종 DPO 데이터셋

logs/
  build_dpo.log            # 현재 실행 중인 재생성 로그
```
