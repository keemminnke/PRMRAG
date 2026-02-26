#!/bin/bash
# KTO v3 학습 + 평가 통합 스크립트
# Usage: nohup bash scripts/run_train_and_eval.sh > logs/kto_v3.log 2>&1 &

set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG
mkdir -p logs

# ============================================================
# 설정
# ============================================================
OUTPUT_DIR="outputs/kto_policy_v3"
FINAL_MODEL="$OUTPUT_DIR/final_model"
EVAL_OUTPUT_KTO="outputs/eval_kto_policy_v3_hotpotqa_val.jsonl"
EVAL_OUTPUT_BASE="outputs/eval_base_hotpotqa_val.jsonl"
MAX_RETRIES=5

TRAIN_CMD="python scripts/train_kto_policy.py \
    --input outputs/original_filtered_for_kto.jsonl \
    --output-dir $OUTPUT_DIR \
    --beta 0.3 \
    --lambda0 10.9 \
    --desirable-weight 1.0 \
    --learning-rate 5e-6 \
    --batch-size 2 \
    --gradient-accumulation 16 \
    --num-epochs 1 \
    --lora-r 32 \
    --lora-alpha 64 \
    --max-length 4096 \
    --truncate-after-bad \
    --save-steps 100 \
    --wandb-run-name kto_v3"

# ============================================================
# Phase 1: KTO 학습 (자동 재시작)
# ============================================================
echo "$(date) ============================================================"
echo "$(date) Phase 1: KTO Training (v3)"
echo "$(date) ============================================================"
echo "  Output dir : $OUTPUT_DIR"
echo "  beta=0.3, lambda0=10.9, lora_r=32, epoch=1"
echo "  Data: outputs/original_filtered_for_kto.jsonl"

RETRY_COUNT=0
while [ ! -d "$FINAL_MODEL" ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # 이미 실행 중인 프로세스 대기
    EXISTING_PID=$(pgrep -f "train_kto_policy" || true)
    if [ -n "$EXISTING_PID" ]; then
        echo "$(date) Training already running (PID: $EXISTING_PID), waiting..."
        while kill -0 "$EXISTING_PID" 2>/dev/null; do sleep 30; done
        echo "$(date) Process $EXISTING_PID finished."
    fi

    if [ -d "$FINAL_MODEL" ]; then
        echo "$(date) final_model exists, skipping training."
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "$(date) Starting training attempt $RETRY_COUNT/$MAX_RETRIES ..."
    $TRAIN_CMD || true
    sleep 5
done

if [ ! -d "$FINAL_MODEL" ]; then
    echo "$(date) ERROR: Training failed after $MAX_RETRIES attempts!"
    exit 1
fi

echo "$(date) Phase 1 Complete: $FINAL_MODEL"

# ============================================================
# Phase 2: Base 모델 평가 (LoRA 없음)
# ============================================================
echo ""
echo "$(date) ============================================================"
echo "$(date) Phase 2: Evaluation (Base model)"
echo "$(date) ============================================================"

python scripts/generate_trajectories.py \
    --data_path data/raw/questions/hotpotqa_validation.jsonl \
    --output_path "$EVAL_OUTPUT_BASE" \
    --num_samples 1 \
    --batch_size 1000 \
    --limit 9999 \
    --temperature 0.0 \
    --no_rerank \
    --resume

echo "$(date) Phase 2 Complete: $EVAL_OUTPUT_BASE"

# ============================================================
# Phase 3: KTO 모델 평가
# ============================================================
echo ""
echo "$(date) ============================================================"
echo "$(date) Phase 3: Evaluation (KTO v3 model)"
echo "$(date) ============================================================"

python scripts/generate_trajectories.py \
    --data_path data/raw/questions/hotpotqa_validation.jsonl \
    --output_path "$EVAL_OUTPUT_KTO" \
    --lora_adapter "$FINAL_MODEL" \
    --num_samples 1 \
    --batch_size 1000 \
    --limit 9999 \
    --temperature 0.0 \
    --no_rerank \
    --resume

echo "$(date) Phase 3 Complete: $EVAL_OUTPUT_KTO"

# ============================================================
# Phase 4: 성능 비교
# ============================================================
echo ""
echo "$(date) ============================================================"
echo "$(date) Phase 4: Performance Comparison"
echo "$(date) ============================================================"

python3 -c "
import json

def load_pass1(path, id_field='trajectory_id'):
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tid = r.get(id_field, '')
            qid = tid.rsplit('_sample_', 1)[0] if '_sample_' in tid else tid
            if qid not in results:
                results[qid] = r.get('is_correct', False)
    return results

# KTO v3 결과
kto = load_pass1('$EVAL_OUTPUT_KTO')
kto_acc = sum(kto.values()) / len(kto) * 100 if kto else 0
print(f'KTO v3  : {sum(kto.values())}/{len(kto)} = {kto_acc:.2f}%')

# Base 결과 (sample_0만 pass@1)
base = load_pass1('$EVAL_OUTPUT_BASE')
base_acc = sum(base.values()) / len(base) * 100 if base else 0
print(f'Base    : {sum(base.values())}/{len(base)} = {base_acc:.2f}%')

# 공통 질문 비교
common = set(kto.keys()) & set(base.keys())
if common:
    kto_common = sum(kto[q] for q in common) / len(common) * 100
    base_common = sum(base[q] for q in common) / len(common) * 100
    print()
    print(f'=== 공통 {len(common)}개 질문 비교 ===')
    print(f'Base    : {base_common:.2f}%')
    print(f'KTO v3  : {kto_common:.2f}%')
    print(f'Diff    : {kto_common - base_common:+.2f}%p')
else:
    print('(공통 질문 없음 - 다른 데이터셋 사용)')
"

echo ""
echo "$(date) ============================================================"
echo "$(date) All Done!"
echo "$(date) ============================================================"
