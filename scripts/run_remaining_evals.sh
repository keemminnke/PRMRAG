#!/bin/bash
# Run all remaining evals from eval_plan.md
# HiPRAG excluded. ReasonRAG/Search-R1/Standard RAG/V1 already done.
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

export NUMEXPR_MAX_THREADS=64
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface

QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
SELFRAG_MODEL="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface/models--selfrag--selfrag_llama2_13b/snapshots/3d308b05d77bcff0e199b210bece367461173163"
AUTORAG_MODEL="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface/models--ICTNLP--Auto-RAG-Llama-3-8B-Instruct/snapshots/b53199630d98c237634178ee2ac5dcf5d68839c2"

DATASETS="popqa hotpotqa 2wikimultihopqa bamboogle musique"

check_done() {
    local result_dir="$1"
    find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .
}

print_result() {
    local result_dir="$1"
    cat "$result_dir"/*/metric_score.txt 2>/dev/null || cat "$result_dir"/metric_score.txt 2>/dev/null || true
}

# ============================================================
# PHASE 1: Direct (No RAG) — Qwen2.5-7B-Instruct
# ============================================================
echo "=============================================="
echo "PHASE 1: Direct (No RAG)"
echo "=============================================="
for ds in $DATASETS; do
    tag="direct_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] Direct $ds — already done"
        print_result "$result_dir"
        continue
    fi
    echo "[RUN] Direct $ds"
    python scripts/eval_flashrag_direct.py \
        --model "$QWEN_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0
    echo "[DONE] Direct $ds"
    print_result "$result_dir"
done

# ============================================================
# PHASE 2: FLARE — Qwen2.5-7B-Instruct + BGE
# ============================================================
echo ""
echo "=============================================="
echo "PHASE 2: FLARE"
echo "=============================================="
for ds in $DATASETS; do
    tag="flare_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] FLARE $ds — already done"
        print_result "$result_dir"
        continue
    fi
    echo "[RUN] FLARE $ds"
    python scripts/eval_flashrag_builtin.py \
        --model "$QWEN_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --pipeline flare \
        --temperature 0
    echo "[DONE] FLARE $ds"
    print_result "$result_dir"
done

# ============================================================
# PHASE 3: IRCoT — Qwen2.5-7B-Instruct + BGE
# ============================================================
echo ""
echo "=============================================="
echo "PHASE 3: IRCoT"
echo "=============================================="
for ds in $DATASETS; do
    tag="ircot_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] IRCoT $ds — already done"
        print_result "$result_dir"
        continue
    fi
    echo "[RUN] IRCoT $ds"
    python scripts/eval_flashrag_builtin.py \
        --model "$QWEN_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --pipeline ircot \
        --temperature 0
    echo "[DONE] IRCoT $ds"
    print_result "$result_dir"
done

# ============================================================
# PHASE 4: Iter-RetGen — Qwen2.5-7B-Instruct + BGE
# ============================================================
echo ""
echo "=============================================="
echo "PHASE 4: Iter-RetGen"
echo "=============================================="
for ds in $DATASETS; do
    tag="iterretgen_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] Iter-RetGen $ds — already done"
        print_result "$result_dir"
        continue
    fi
    echo "[RUN] Iter-RetGen $ds"
    python scripts/eval_flashrag_builtin.py \
        --model "$QWEN_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --pipeline iterretgen \
        --temperature 0
    echo "[DONE] Iter-RetGen $ds"
    print_result "$result_dir"
done

# ============================================================
# PHASE 5: Self-RAG — selfrag_llama2_13b + BGE
# ============================================================
echo ""
echo "=============================================="
echo "PHASE 5: Self-RAG"
echo "=============================================="
for ds in $DATASETS; do
    tag="selfrag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] Self-RAG $ds — already done"
        print_result "$result_dir"
        continue
    fi
    echo "[RUN] Self-RAG $ds"
    python scripts/eval_flashrag_builtin.py \
        --model "$SELFRAG_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --pipeline selfrag \
        --temperature 0
    echo "[DONE] Self-RAG $ds"
    print_result "$result_dir"
done

# ============================================================
# PHASE 6: AutoRAG — Auto-RAG-Llama-3-8B-Instruct + BGE
# ============================================================
echo ""
echo "=============================================="
echo "PHASE 6: AutoRAG"
echo "=============================================="
for ds in $DATASETS; do
    tag="autorag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] AutoRAG $ds — already done"
        print_result "$result_dir"
        continue
    fi
    echo "[RUN] AutoRAG $ds"
    python scripts/eval_flashrag_builtin.py \
        --model "$AUTORAG_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --pipeline autorag \
        --temperature 0
    echo "[DONE] AutoRAG $ds"
    print_result "$result_dir"
done

echo ""
echo "=============================================="
echo "ALL REMAINING EVALUATIONS COMPLETE"
echo "=============================================="
