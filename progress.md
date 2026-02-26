# PRMRAG KTO Training — Progress Log

Last updated: 2026-02-26

---

## 현재 상태 (브랜치: no-rpe)

### 핵심 구현 완료
- **Step-level KTO trainer**: `src/prmrag/training/kto_trainer.py`
- **학습 스크립트**: `scripts/train_kto_policy.py`
- **자동 Sweep 스크립트**: `scripts/sweep_kto.py` ← 신규

### 다음 할 일
1. **Hyperparameter sweep 실행** → best (beta, lambda0) 자동 탐색
   ```bash
   python scripts/sweep_kto.py --run-winner
   ```
2. sweep winner로 풀 학습
3. `scripts/run_train_and_eval.sh`로 평가

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
  train_kto_policy.py      # 학습 진입점
  sweep_kto.py             # 자동 하이퍼파라미터 탐색
  run_train_and_eval.sh    # 학습 + 평가 통합 (v3 설정 하드코딩됨, 업데이트 필요)

src/prmrag/training/
  kto_trainer.py           # KTO 핵심 구현

outputs/
  hotpotqa_critic_results_v8_per_trajectory.jsonl  # 학습 데이터
  kto_sweep_tmp/           # sweep 임시 결과
  kto_policy_*/            # 학습된 모델
```
