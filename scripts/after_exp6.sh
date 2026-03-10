#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NUMEXPR_MAX_THREADS=64

cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

echo "=== Waiting for Exp 6 DPO to finish ==="
while ps aux | grep "dpo_v2_3000q_ep1_matched" | grep -v grep > /dev/null 2>&1; do
    sleep 60
done
echo "=== Exp 6 DPO 완료 ==="

# Exp 6 평가
echo "=== Exp 6 평가 시작 ==="
python scripts/eval_flashrag.py \
    --model outputs/sft_policy_v2/merged_model \
    --lora outputs/dpo_policy_v2_ep1/final_model \
    --tag dpo_v2_ep1_matched \
    --top-k 3 \
    --limit 7405 \
    --temperature 0.8 \
    --gpu-util 0.85

echo "=== Exp 6 평가 결과 ==="
cat outputs/flashrag_dpo_v2_ep1_matched/hotpotqa_*/metric_score.txt 2>/dev/null

# Round 2+3 데이터 생성 시작
echo ""
echo "=== Round 2+3 데이터 생성 시작 ==="
bash scripts/run_round2_3_data.sh
