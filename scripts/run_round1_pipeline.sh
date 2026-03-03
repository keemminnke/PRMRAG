#!/bin/bash
# =============================================================================
# Round 1 Pipeline: HotpotQA +1,000q (2,000q → 3,000q)
#
# Step 1: 질문 샘플링 (1,000q)
# Step 2: Trajectory 생성 (Qwen base, BGE+Reranker, 16 paths)
# Step 3: QwQ-32B Judge Labeling
# Step 4: Critic Model 재학습 (3,000q)
# Step 5: 전체 Re-scoring → 통합 파일
# Step 6: Regeneration + DPO/SFT 데이터 구축
# Step 7: SFT 학습
# Step 8: DPO 학습
# Step 9: 평가
#
# Usage:
#   nohup bash scripts/run_round1_pipeline.sh > logs/round1_pipeline.log 2>&1 &
#   tail -f logs/round1_pipeline.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CRITIC_BASE="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# 기존 데이터
EXISTING_TRAJS="outputs/trajectories/hotpotqa_1000q_v1.jsonl"
JUDGE_LABELS_2000Q="outputs/training_data/judge_labels_2000q_v1.jsonl"
HOTPOTQA_TRAJS_V1="outputs/trajectories/hotpotqa_1000q_v1.jsonl"
MUSIQUE_TRAJS_V1="outputs/trajectories/musique_1000q_v1.jsonl"

# Round 1 출력
NEW_QUESTIONS="data/raw/questions/hotpotqa_train_round2.jsonl"
NEW_TRAJECTORIES="outputs/hotpotqa_round2_trajectories.jsonl"
JUDGE_LABELED="outputs/hotpotqa_round2_judge_labeled.jsonl"
CRITIC_MODEL="outputs/critic_model_v9_3000q"
MERGED_TRAJS="outputs/all_3000q_merged_trajectories.jsonl"
SCORED_ALL="outputs/all_3000q_critic_v9_scored.jsonl"
SCORED_WITH_REGEN="outputs/all_3000q_with_regen.jsonl"
REGEN_TRAJS="outputs/dpo_regen_v2_trajectories.jsonl"
DPO_DATASET="outputs/dpo_dataset_v2_bva_messages.jsonl"
SFT_OUTPUT="outputs/sft_policy_v2"
DPO_OUTPUT="outputs/dpo_policy_v2"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p logs outputs

echo "============================================================"
echo "ROUND 1 PIPELINE — HotpotQA +1,000q (2,000q → 3,000q)"
echo "============================================================"
echo "Started at $(date)"
echo ""

# ---------------------------------------------------------------
# [Step 1/9] 새 질문 1,000개 샘플링
# ---------------------------------------------------------------
STEP1_DONE="${NEW_QUESTIONS}.done"
if [ -f "$STEP1_DONE" ]; then
    echo "[Step 1/9] 질문 샘플링 — SKIP (이미 완료)"
else
    echo "[Step 1/9] 질문 샘플링 시작..."
    python scripts/sample_new_questions.py \
        --source data/raw/questions/hotpotqa_train.jsonl \
        --existing "$EXISTING_TRAJS" \
        --output "$NEW_QUESTIONS" \
        --num 1000 \
        --seed 42

    touch "$STEP1_DONE"
    echo "[Step 1/9] 완료: $(wc -l < "$NEW_QUESTIONS") questions → $NEW_QUESTIONS"
fi
echo ""

# ---------------------------------------------------------------
# [Step 2/9] Trajectory 생성 (Qwen base + BGE + Reranker)
# ---------------------------------------------------------------
STEP2_DONE="${NEW_TRAJECTORIES}.done"
if [ -f "$STEP2_DONE" ]; then
    echo "[Step 2/9] Trajectory 생성 — SKIP (이미 완료)"
else
    echo "[Step 2/9] Trajectory 생성 시작 (Qwen base, BGE+Reranker, 16 paths)..."
    echo "  예상 시간: 4-6시간"

    python scripts/generate_trajectories.py \
        --data_path "$NEW_QUESTIONS" \
        --output_path "$NEW_TRAJECTORIES" \
        --num_samples 16 \
        --retriever bge \
        --batch_size 500 \
        --limit 1000 \
        --policy_model "$BASE_MODEL" \
        --gpu_memory_utilization 0.85 \
        --resume

    touch "$STEP2_DONE"
    echo "[Step 2/9] 완료: $(wc -l < "$NEW_TRAJECTORIES") trajectories → $NEW_TRAJECTORIES"
fi
echo ""

# ---------------------------------------------------------------
# [Step 3/9] QwQ-32B Judge Labeling
# ---------------------------------------------------------------
STEP3_DONE="${JUDGE_LABELED}.done"
if [ -f "$STEP3_DONE" ]; then
    echo "[Step 3/9] Judge Labeling — SKIP (이미 완료)"
else
    echo "[Step 3/9] QwQ-32B Judge Labeling 시작..."
    echo "  예상 시간: 6-8시간"

    python scripts/judge_label_qwq.py \
        --input "$NEW_TRAJECTORIES" \
        --output "$JUDGE_LABELED" \
        --batch-size 500 \
        --resume

    touch "$STEP3_DONE"
    echo "[Step 3/9] 완료: $(wc -l < "$JUDGE_LABELED") trajectories labeled → $JUDGE_LABELED"
fi
echo ""

# ---------------------------------------------------------------
# [Step 4/9] Critic Model 재학습 (3,000q)
# ---------------------------------------------------------------
MERGED_JUDGE="outputs/training_data/judge_labels_3000q_merged.jsonl"
STEP4_DONE="${CRITIC_MODEL}/.done"
if [ -f "$STEP4_DONE" ]; then
    echo "[Step 4/9] Critic 재학습 — SKIP (이미 완료)"
else
    echo "[Step 4/9] Critic Model 재학습 (3,000q)..."
    echo "  기존: $JUDGE_LABELS_2000Q ($(wc -l < "$JUDGE_LABELS_2000Q") trajs)"
    echo "  추가: $JUDGE_LABELED ($(wc -l < "$JUDGE_LABELED") trajs)"
    echo "  예상 시간: 3-4시간"

    # 두 judge label 파일을 하나로 합침
    cat "$JUDGE_LABELS_2000Q" "$JUDGE_LABELED" > "$MERGED_JUDGE"
    echo "  Merged: $(wc -l < "$MERGED_JUDGE") trajectories → $MERGED_JUDGE"

    torchrun --nproc_per_node=2 scripts/train_critic_model.py \
        --train-data "$MERGED_JUDGE" \
        --output-dir "$CRITIC_MODEL" \
        --model-name "$CRITIC_BASE" \
        --num-epochs 1

    touch "$STEP4_DONE"
    echo "[Step 4/9] 완료: $CRITIC_MODEL"
fi
echo ""

# ---------------------------------------------------------------
# [Step 5/9] 전체 Re-scoring → 통합 파일
# ---------------------------------------------------------------
STEP5_DONE="${SCORED_ALL}.done"
if [ -f "$STEP5_DONE" ]; then
    echo "[Step 5/9] Re-scoring — SKIP (이미 완료)"
else
    echo "[Step 5/9] 전체 3,000q Re-scoring..."
    echo "  예상 시간: 4-5시간"

    # 3개 trajectory 파일을 하나로 합침
    echo "  Merging trajectory files..."
    cat "$HOTPOTQA_TRAJS_V1" "$MUSIQUE_TRAJS_V1" "$NEW_TRAJECTORIES" > "$MERGED_TRAJS"
    echo "  Merged: $(wc -l < "$MERGED_TRAJS") trajectories"

    python scripts/evaluate_with_critic.py \
        --trajectories "$MERGED_TRAJS" \
        --critic-model "${CRITIC_MODEL}/final_model" \
        --base-model "$CRITIC_BASE" \
        --output "$SCORED_ALL"

    touch "$STEP5_DONE"
    echo "[Step 5/9] 완료: $(wc -l < "$SCORED_ALL") trajectories → $SCORED_ALL"
fi
echo ""

# ---------------------------------------------------------------
# [Step 6/9] Regeneration + DPO/SFT 데이터 구축
# ---------------------------------------------------------------
STEP6_DONE="${DPO_DATASET}.done"
if [ -f "$STEP6_DONE" ]; then
    echo "[Step 6/9] Regen + 데이터 구축 — SKIP (이미 완료)"
else
    echo "[Step 6/9] Regeneration + DPO/SFT 데이터 구축..."
    echo "  예상 시간: 3-4시간"

    # 통합 파일을 --hotpotqa로 전달, --musique는 사용 안함
    python scripts/build_dpo_dataset.py \
        --hotpotqa "$SCORED_ALL" \
        --musique /dev/null \
        --regen-out "$REGEN_TRAJS" \
        --scored-out "$SCORED_WITH_REGEN" \
        --dpo-out "$DPO_DATASET" \
        --policy-model "$BASE_MODEL" \
        --critic-model "${CRITIC_MODEL}/final_model" \
        --critic-base "$CRITIC_BASE" \
        --policy-gpu 0.45 \
        --critic-gpu 0.45

    touch "$STEP6_DONE"
    echo "[Step 6/9] 완료:"
    echo "  DPO pairs: $(wc -l < "$DPO_DATASET")"
    echo "  Scored+Regen: $(wc -l < "$SCORED_WITH_REGEN")"
fi
echo ""

# ---------------------------------------------------------------
# [Step 7/9] SFT 학습
# ---------------------------------------------------------------
STEP7_DONE="${SFT_OUTPUT}/.done"
if [ -f "$STEP7_DONE" ]; then
    echo "[Step 7/9] SFT 학습 — SKIP (이미 완료)"
else
    echo "[Step 7/9] SFT 학습 (Qwen base, 3,000q)..."
    echo "  예상 시간: 2-3시간"

    torchrun --nproc_per_node=2 scripts/train_sft_policy.py \
        --input "$SCORED_WITH_REGEN" \
        --model-name "$BASE_MODEL" \
        --output-dir "$SFT_OUTPUT" \
        --num-epochs 1 \
        --wandb-run-name sft_v2_3000q

    touch "$STEP7_DONE"
    echo "[Step 7/9] 완료: ${SFT_OUTPUT}/merged_model"
fi
echo ""

# ---------------------------------------------------------------
# [Step 8/9] DPO 학습
# ---------------------------------------------------------------
STEP8_DONE="${DPO_OUTPUT}/.done"
if [ -f "$STEP8_DONE" ]; then
    echo "[Step 8/9] DPO 학습 — SKIP (이미 완료)"
else
    echo "[Step 8/9] DPO 학습 (SFT v2 위에)..."
    echo "  예상 시간: ~13시간"

    torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
        --dpo-dataset "$DPO_DATASET" \
        --model-name "${SFT_OUTPUT}/merged_model" \
        --output-dir "$DPO_OUTPUT" \
        --num-epochs 3 \
        --wandb-run-name dpo_v2_3000q

    touch "$STEP8_DONE"
    echo "[Step 8/9] 완료: ${DPO_OUTPUT}/final_model"
fi
echo ""

# ---------------------------------------------------------------
# [Step 9/9] 평가
# ---------------------------------------------------------------
EVAL_TAG="dpo_v2_3000q_full"
STEP9_DONE="outputs/eval/flashrag_${EVAL_TAG}/.done"
if [ -f "$STEP9_DONE" ]; then
    echo "[Step 9/9] 평가 — SKIP (이미 완료)"
else
    echo "[Step 9/9] FlashRAG 평가 (7,405q)..."
    echo "  예상 시간: ~1시간"

    python scripts/eval_flashrag.py \
        --model "${SFT_OUTPUT}/merged_model" \
        --lora "${DPO_OUTPUT}/final_model" \
        --tag "$EVAL_TAG" \
        --top-k 3 \
        --limit 7405 \
        --temperature 0.8 \
        --gpu-util 0.85

    touch "$STEP9_DONE"
    echo "[Step 9/9] 완료: outputs/eval/flashrag_${EVAL_TAG}/"
fi

# ---------------------------------------------------------------
# 최종 결과 비교
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "ROUND 1 COMPLETE — Final Results"
echo "============================================================"
echo "Finished at $(date)"
echo ""
echo "=== 모니터링 명령어 ==="
echo "  tail -f logs/round1_pipeline.log"
echo "  nvidia-smi"
echo "  ps aux | grep python"
