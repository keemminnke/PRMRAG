#!/bin/bash
# =============================================================================
# Round 2+3 데이터 생성 파이프라인 (학습 제외)
#
# Round 2: MusiQue +1,000q      (3,000q → 4,000q)
# Round 3: 2WikiMultihopQA +1,000q (4,000q → 5,000q)
#
# 데이터만 한번에 뽑고, 학습은 별도로 실험
#
# 수정사항 (Round 1 교훈 반영):
#   - System prompt 통일 (SFT/DPO/eval 모두 동일)
#   - 불량 trajectory 필터링 (no answer, max_step=10)
#   - compare_voting_methods.py로 scoring (evaluate_with_critic.py 금지)
#   - SFT input = scored_all + scored_with_regen (regen만 쓰면 안됨)
#   - DPO: messages format, Best-vs-All pairing
#
# Usage:
#   nohup bash scripts/run_round2_3_data.sh > logs/round2_3_data.log 2>&1 &
#   tail -f logs/round2_3_data.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CRITIC_BASE="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NUMEXPR_MAX_THREADS=64

# ---------------------------------------------------------------
# 기존 데이터 경로
# ---------------------------------------------------------------
# Round 1까지의 trajectory 파일들
HOTPOTQA_V1="outputs/trajectories/hotpotqa_1000q_v1.jsonl"
MUSIQUE_V1="outputs/trajectories/musique_1000q_v1.jsonl"
HOTPOTQA_R2="outputs/hotpotqa_round2_trajectories.jsonl"

# Round 1까지의 judge labels
JUDGE_LABELS_3000Q="outputs/training_data/judge_labels_3000q_merged.jsonl"

# ---------------------------------------------------------------
# Round 2 출력 (MusiQue +1,000q)
# ---------------------------------------------------------------
R2_QUESTIONS="data/raw/questions/musique_train_round2.jsonl"
R2_TRAJECTORIES="outputs/musique_round2_trajectories.jsonl"
R2_JUDGE_LABELED="outputs/musique_round2_judge_labeled.jsonl"

# ---------------------------------------------------------------
# Round 3 출력 (2WikiMultihopQA +1,000q)
# ---------------------------------------------------------------
R3_QUESTIONS="data/raw/questions/2wikimultihop_train_round1.jsonl"
R3_TRAJECTORIES="outputs/2wiki_round1_trajectories.jsonl"
R3_JUDGE_LABELED="outputs/2wiki_round1_judge_labeled.jsonl"

# ---------------------------------------------------------------
# 통합 출력 (5,000q)
# ---------------------------------------------------------------
JUDGE_LABELS_5000Q="outputs/training_data/judge_labels_5000q_merged.jsonl"
CRITIC_MODEL_5000Q="outputs/critic_model_v11_5000q"
MERGED_TRAJS_5000Q="outputs/all_5000q_merged_trajectories.jsonl"
SCORED_ALL_5000Q="outputs/all_5000q_critic_v11_scored.jsonl"
REGEN_TRAJS="outputs/dpo_regen_v4_trajectories.jsonl"
SCORED_WITH_REGEN="outputs/all_5000q_with_regen.jsonl"

# DPO/SFT 데이터 (4,000q / 5,000q 분리는 학습 시 처리)
DPO_DATASET_5000Q="outputs/dpo_dataset_v4_5000q_bva.jsonl"
SFT_INPUT_5000Q="outputs/all_5000q_sft_input.jsonl"

echo "=============================================================="
echo "  ROUND 2+3 DATA GENERATION PIPELINE"
echo "  Round 2: MusiQue +1,000q (→ 4,000q)"
echo "  Round 3: 2WikiMultihopQA +1,000q (→ 5,000q)"
echo "=============================================================="
echo ""

# ===============================================================
# [Step 1] 질문 샘플링 (Round 2 + Round 3 동시)
# ===============================================================

# --- Round 2: MusiQue 1,000q ---
STEP1A_DONE="${R2_QUESTIONS}.done"
if [ -f "$STEP1A_DONE" ]; then
    echo "[Step 1a] MusiQue 질문 샘플링 — SKIP (이미 완료)"
else
    echo "[Step 1a] MusiQue 1,000q 샘플링..."
    python scripts/sample_new_questions.py \
        --source data/raw/questions/musique_train.jsonl \
        --existing outputs/trajectories/musique_1000q_v1.jsonl \
        --output "$R2_QUESTIONS" \
        --num 1000 \
        --seed 43
    touch "$STEP1A_DONE"
    echo "[Step 1a] 완료: $R2_QUESTIONS"
fi

# --- Round 3: 2WikiMultihopQA 1,000q ---
STEP1B_DONE="${R3_QUESTIONS}.done"
if [ -f "$STEP1B_DONE" ]; then
    echo "[Step 1b] 2WikiMultihopQA 질문 샘플링 — SKIP (이미 완료)"
else
    echo "[Step 1b] 2WikiMultihopQA 1,000q 샘플링..."
    # 2Wiki는 처음 사용하므로 기존 데이터 없음 → head로 직접 샘플링
    python3 -c "
import json, random
random.seed(44)
with open('data/raw/questions/2wikimultihop_train.jsonl') as f:
    all_qs = [json.loads(l) for l in f]
sampled = random.sample(all_qs, 1000)
with open('$R3_QUESTIONS', 'w') as f:
    for q in sampled:
        f.write(json.dumps(q, ensure_ascii=False) + '\n')
print(f'Sampled {len(sampled)} questions → $R3_QUESTIONS')
from collections import Counter
types = Counter(q.get('type','') for q in sampled)
print(f'Types: {dict(types)}')
"
    touch "$STEP1B_DONE"
    echo "[Step 1b] 완료: $R3_QUESTIONS"
fi
echo ""

# ===============================================================
# [Step 2] Trajectory 생성 (Round 2 → Round 3 순차)
# ===============================================================

# --- Round 2: MusiQue ---
STEP2A_DONE="${R2_TRAJECTORIES}.done"
if [ -f "$STEP2A_DONE" ]; then
    echo "[Step 2a] MusiQue trajectory 생성 — SKIP (이미 완료)"
else
    echo "[Step 2a] MusiQue trajectory 생성 (1,000q × 16)..."
    echo "  예상 시간: ~4시간"
    python scripts/generate_trajectories.py \
        --data_path "$R2_QUESTIONS" \
        --output_path "$R2_TRAJECTORIES" \
        --num_samples 16 \
        --retriever bge \
        --batch_size 500 \
        --limit 1000 \
        --policy_model "$BASE_MODEL" \
        --gpu_memory_utilization 0.85 \
        --resume
    touch "$STEP2A_DONE"
    echo "[Step 2a] 완료: $R2_TRAJECTORIES"
fi

# --- Round 3: 2WikiMultihopQA ---
STEP2B_DONE="${R3_TRAJECTORIES}.done"
if [ -f "$STEP2B_DONE" ]; then
    echo "[Step 2b] 2WikiMultihopQA trajectory 생성 — SKIP (이미 완료)"
else
    echo "[Step 2b] 2WikiMultihopQA trajectory 생성 (1,000q × 16)..."
    echo "  예상 시간: ~4시간"
    python scripts/generate_trajectories.py \
        --data_path "$R3_QUESTIONS" \
        --output_path "$R3_TRAJECTORIES" \
        --num_samples 16 \
        --retriever bge \
        --batch_size 500 \
        --limit 1000 \
        --policy_model "$BASE_MODEL" \
        --gpu_memory_utilization 0.85 \
        --resume
    touch "$STEP2B_DONE"
    echo "[Step 2b] 완료: $R3_TRAJECTORIES"
fi
echo ""

# ===============================================================
# [Step 3] QwQ-32B Judge Labeling (Round 2 → Round 3 순차)
# ===============================================================

# --- Round 2: MusiQue ---
STEP3A_DONE="${R2_JUDGE_LABELED}.done"
if [ -f "$STEP3A_DONE" ]; then
    echo "[Step 3a] MusiQue judge labeling — SKIP (이미 완료)"
else
    echo "[Step 3a] MusiQue judge labeling..."
    echo "  예상 시간: ~10시간"
    python scripts/judge_label_qwq.py \
        --input "$R2_TRAJECTORIES" \
        --output "$R2_JUDGE_LABELED" \
        --batch-size 500 \
        --resume
    touch "$STEP3A_DONE"
    echo "[Step 3a] 완료: $R2_JUDGE_LABELED"
fi

# --- Round 3: 2WikiMultihopQA ---
STEP3B_DONE="${R3_JUDGE_LABELED}.done"
if [ -f "$STEP3B_DONE" ]; then
    echo "[Step 3b] 2WikiMultihopQA judge labeling — SKIP (이미 완료)"
else
    echo "[Step 3b] 2WikiMultihopQA judge labeling..."
    echo "  예상 시간: ~10시간"
    python scripts/judge_label_qwq.py \
        --input "$R3_TRAJECTORIES" \
        --output "$R3_JUDGE_LABELED" \
        --batch-size 500 \
        --resume
    touch "$STEP3B_DONE"
    echo "[Step 3b] 완료: $R3_JUDGE_LABELED"
fi
echo ""

# ===============================================================
# [Step 4] Critic Model 재학습 (5,000q 전체)
# ===============================================================
STEP4_DONE="${CRITIC_MODEL_5000Q}/.done"
if [ -f "$STEP4_DONE" ]; then
    echo "[Step 4] Critic 재학습 — SKIP (이미 완료)"
else
    echo "[Step 4] Critic 재학습 (5,000q)..."
    echo "  예상 시간: ~5시간"

    # Judge labels 합치기 (3,000q + MusiQue 1,000q + 2Wiki 1,000q)
    cat "$JUDGE_LABELS_3000Q" \
        "$R2_JUDGE_LABELED" \
        "$R3_JUDGE_LABELED" \
        > "$JUDGE_LABELS_5000Q"
    echo "  Judge labels 합침: $(wc -l < "$JUDGE_LABELS_5000Q") trajectories"

    torchrun --nproc_per_node=2 scripts/train_critic_model.py \
        --train-data "$JUDGE_LABELS_5000Q" \
        --output-dir "$CRITIC_MODEL_5000Q" \
        --model-name "$CRITIC_BASE" \
        --num-epochs 1

    touch "$STEP4_DONE"
    echo "[Step 4] 완료: $CRITIC_MODEL_5000Q"
fi
echo ""

# ===============================================================
# [Step 5] 전체 Re-scoring (5,000q, vLLM batch)
# ===============================================================
STEP5_DONE="${SCORED_ALL_5000Q}.done"
if [ -f "$STEP5_DONE" ]; then
    echo "[Step 5] Re-scoring — SKIP (이미 완료)"
else
    echo "[Step 5] 전체 Re-scoring (5,000q)..."
    echo "  예상 시간: ~8시간"

    # 전체 trajectory 합치기 (5,000q)
    cat "$HOTPOTQA_V1" \
        "$MUSIQUE_V1" \
        "$HOTPOTQA_R2" \
        "$R2_TRAJECTORIES" \
        "$R3_TRAJECTORIES" \
        > "$MERGED_TRAJS_5000Q"
    echo "  Merged: $(wc -l < "$MERGED_TRAJS_5000Q") trajectories"

    # vLLM batch scoring
    python scripts/compare_voting_methods.py \
        --trajectories "$MERGED_TRAJS_5000Q" \
        --critic-model "${CRITIC_MODEL_5000Q}/final_model" \
        --critic-base "$CRITIC_BASE" \
        --skip-versaprm \
        --gpu-memory 0.90 \
        --output outputs/critic_v11_voting_results.json

    # 출력 파일명 변경
    mv outputs/critic_v11_voting_results_per_trajectory.jsonl "$SCORED_ALL_5000Q"

    touch "$STEP5_DONE"
    echo "[Step 5] 완료: $(wc -l < "$SCORED_ALL_5000Q") trajectories → $SCORED_ALL_5000Q"
fi
echo ""

# ===============================================================
# [Step 5.5] 불량 Trajectory 필터링
# ===============================================================
FILTERED_5000Q="outputs/all_5000q_filtered.jsonl"
STEP5_5_DONE="${FILTERED_5000Q}.done"
if [ -f "$STEP5_5_DONE" ]; then
    echo "[Step 5.5] 불량 trajectory 필터링 — SKIP (이미 완료)"
else
    echo "[Step 5.5] 불량 trajectory 필터링..."
    python3 -c "
import json, re

input_path = '$SCORED_ALL_5000Q'
output_path = '$FILTERED_5000Q'

total = filtered_no_answer = filtered_max_step = kept = 0
with open(input_path) as fin, open(output_path, 'w') as fout:
    for line in fin:
        total += 1
        d = json.loads(line)
        steps = d.get('steps', [])

        # 필터 1: final answer 없는 trajectory
        has_answer = any(s.get('answer') for s in steps)
        if not has_answer:
            filtered_no_answer += 1
            continue

        # 필터 2: max_step(10) 도달하여 잘린 trajectory
        if len(steps) >= 10:
            filtered_max_step += 1
            continue

        kept += 1
        fout.write(line)

print(f'  전체: {total}')
print(f'  제외 (no answer): {filtered_no_answer}')
print(f'  제외 (max_step>=10): {filtered_max_step}')
print(f'  유지: {kept} ({100*kept/total:.1f}%)')
"
    # 필터링된 파일을 scored로 대체
    mv "$SCORED_ALL_5000Q" "${SCORED_ALL_5000Q}.unfiltered"
    mv "$FILTERED_5000Q" "$SCORED_ALL_5000Q"

    touch "$STEP5_5_DONE"
    echo "[Step 5.5] 완료"
fi
echo ""

# ===============================================================
# [Step 6] Regeneration + DPO 데이터 구축 (5,000q)
# ===============================================================
STEP6_DONE="${DPO_DATASET_5000Q}.done"
if [ -f "$STEP6_DONE" ]; then
    echo "[Step 6] Regen + DPO 데이터 — SKIP (이미 완료)"
else
    echo "[Step 6] Regeneration + DPO 데이터 구축 (5,000q)..."
    echo "  예상 시간: ~6시간"

    python scripts/build_dpo_dataset.py \
        --hotpotqa "$SCORED_ALL_5000Q" \
        --musique /dev/null \
        --regen-out "$REGEN_TRAJS" \
        --scored-out "$SCORED_WITH_REGEN" \
        --dpo-out "$DPO_DATASET_5000Q" \
        --policy-model "$BASE_MODEL" \
        --critic-model "${CRITIC_MODEL_5000Q}/final_model" \
        --critic-base "$CRITIC_BASE" \
        --policy-gpu 0.45 \
        --critic-gpu 0.45

    touch "$STEP6_DONE"
    echo "[Step 6] 완료"
fi
echo ""

# ===============================================================
# [Step 6.5] SFT 입력 데이터 준비 (4,000q / 5,000q)
# ===============================================================
STEP6_5_DONE="${SFT_INPUT_5000Q}.done"
if [ -f "$STEP6_5_DONE" ]; then
    echo "[Step 6.5] SFT 입력 준비 — SKIP (이미 완료)"
else
    echo "[Step 6.5] SFT 입력 데이터 준비..."

    # 5,000q SFT input = scored_all + scored_with_regen
    cat "$SCORED_ALL_5000Q" "$SCORED_WITH_REGEN" > "$SFT_INPUT_5000Q"
    echo "  5,000q SFT input: $(wc -l < "$SFT_INPUT_5000Q") trajectories"

    # 4,000q SFT input (Round 3 데이터 제외)
    # 4,000q = hotpotqa_v1 + musique_v1 + hotpotqa_r2 + musique_r2 (2Wiki 제외)
    python3 -c "
import json

# 2WikiMultihopQA question IDs 수집
wiki2_qids = set()
with open('$R3_QUESTIONS') as f:
    for line in f:
        d = json.loads(line)
        wiki2_qids.add(d['_id'])
print(f'  2Wiki question IDs: {len(wiki2_qids)}')

# 5,000q에서 2Wiki 제외 → 4,000q
kept = 0
total = 0
with open('$SFT_INPUT_5000Q') as fin, open('outputs/all_4000q_sft_input.jsonl', 'w') as fout:
    for line in fin:
        total += 1
        d = json.loads(line)
        tid = d['trajectory_id']
        qid = tid.rsplit('_sample_', 1)[0] if '_sample_' in tid else tid
        if qid not in wiki2_qids:
            fout.write(line)
            kept += 1
print(f'  4,000q SFT input: {kept} trajectories (전체 {total}에서 2Wiki 제외)')
"

    touch "$STEP6_5_DONE"
    echo "[Step 6.5] 완료"
fi
echo ""

# ===============================================================
# [Step 6.6] 4,000q DPO 데이터 (2Wiki 제외)
# ===============================================================
DPO_DATASET_4000Q="outputs/dpo_dataset_v3_4000q_bva.jsonl"
STEP6_6_DONE="${DPO_DATASET_4000Q}.done"
if [ -f "$STEP6_6_DONE" ]; then
    echo "[Step 6.6] 4,000q DPO 데이터 — SKIP (이미 완료)"
else
    echo "[Step 6.6] 4,000q DPO 데이터 구축 (2Wiki 제외)..."

    # 4,000q scored 파일 만들기 (2Wiki 제외)
    python3 -c "
import json
wiki2_qids = set()
with open('$R3_QUESTIONS') as f:
    for line in f:
        d = json.loads(line)
        wiki2_qids.add(d['_id'])

kept = 0
with open('$SCORED_ALL_5000Q') as fin, open('outputs/all_4000q_critic_scored.jsonl', 'w') as fout:
    for line in fin:
        d = json.loads(line)
        tid = d['trajectory_id']
        qid = tid.rsplit('_sample_', 1)[0] if '_sample_' in tid else tid
        if qid not in wiki2_qids:
            fout.write(line)
            kept += 1
print(f'  4,000q scored: {kept} trajectories')
"

    # 4,000q DPO build
    python scripts/build_dpo_dataset.py \
        --hotpotqa outputs/all_4000q_critic_scored.jsonl \
        --musique /dev/null \
        --scored-out "$SCORED_WITH_REGEN" \
        --dpo-out "$DPO_DATASET_4000Q" \
        --build-only

    touch "$STEP6_6_DONE"
    echo "[Step 6.6] 완료: $DPO_DATASET_4000Q"
fi
echo ""

# ===============================================================
echo "=============================================================="
echo "  ROUND 2+3 DATA GENERATION COMPLETE"
echo "=============================================================="
echo ""
echo "생성된 데이터:"
echo "  5,000q scored:     $SCORED_ALL_5000Q"
echo "  5,000q SFT input:  $SFT_INPUT_5000Q"
echo "  5,000q DPO pairs:  $DPO_DATASET_5000Q"
echo "  4,000q SFT input:  outputs/all_4000q_sft_input.jsonl"
echo "  4,000q DPO pairs:  $DPO_DATASET_4000Q"
echo ""
echo "학습은 별도로 실험:"
echo "  scripts/run_training_experiments.sh 참조"
echo "=============================================================="
