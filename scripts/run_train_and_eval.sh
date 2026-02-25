#!/bin/bash
# Wrapper: KTO 학습 감시 + 자동 재시작 + 완료 후 평가
# Usage: nohup bash scripts/run_train_and_eval.sh > logs/train_and_eval.log 2>&1 &

set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

TRAIN_CMD="python scripts/train_kto_policy.py \
    --input outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
    --input outputs/regenerated_critic_scored_filtered.jsonl \
    --output-dir outputs/kto_policy_v2 \
    --truncate-after-bad \
    --filter-no-gold \
    --save-steps 100 \
    --wandb-run-name kto_v2_filtered"

FINAL_MODEL="outputs/kto_policy_v2/final_model"
MAX_RETRIES=5
RETRY_COUNT=0

# ============================================================
# Phase 1: KTO 학습 (자동 재시작)
# ============================================================
echo "$(date) === Phase 1: KTO Training ==="

while [ ! -d "$FINAL_MODEL" ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # 이미 학습 중인 프로세스가 있으면 기다리기
    EXISTING_PID=$(pgrep -f "train_kto_policy" || true)
    if [ -n "$EXISTING_PID" ]; then
        echo "$(date) Training already running (PID: $EXISTING_PID), waiting..."
        while kill -0 "$EXISTING_PID" 2>/dev/null; do
            sleep 30
        done
        echo "$(date) Process $EXISTING_PID finished."
    fi

    # final_model이 있으면 학습 완료
    if [ -d "$FINAL_MODEL" ]; then
        echo "$(date) Training completed! final_model exists."
        break
    fi

    # 재시작 (checkpoint resume 자동)
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "$(date) Starting training (attempt $RETRY_COUNT/$MAX_RETRIES)..."
    $TRAIN_CMD || true

    sleep 5
done

if [ ! -d "$FINAL_MODEL" ]; then
    echo "$(date) ERROR: Training failed after $MAX_RETRIES attempts!"
    exit 1
fi

echo "$(date) === Phase 1 Complete: Model at $FINAL_MODEL ==="

# ============================================================
# Phase 2: Validation 평가
# ============================================================
echo ""
echo "$(date) === Phase 2: Evaluation on HotpotQA Validation ==="

EVAL_OUTPUT="outputs/eval_kto_v2_validation.jsonl"

python scripts/generate_trajectories.py \
    --data_path data/raw/questions/hotpotqa_validation.jsonl \
    --output_path "$EVAL_OUTPUT" \
    --lora_adapter "$FINAL_MODEL" \
    --num_samples 1 \
    --batch_size 50 \
    --limit 9999 \
    --temperature 0.0 \
    --resume

echo ""
echo "$(date) === Phase 2 Complete: Results at $EVAL_OUTPUT ==="

# 간단한 정확도 계산
python3 -c "
import json, re

def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

correct = 0
total = 0
with open('$EVAL_OUTPUT') as f:
    for line in f:
        row = json.loads(line)
        gold = normalize(row.get('gold_answer', ''))
        pred = normalize(row.get('predicted_answer', ''))
        if gold and pred:
            total += 1
            if gold in pred or pred in gold:
                correct += 1

print(f'=== KTO v2 Evaluation Results ===')
print(f'Total: {total}')
print(f'Correct: {correct} ({100*correct/total:.1f}%)')
"

echo "$(date) === All Done ==="
