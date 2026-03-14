#!/bin/bash
# IRCoT (max_iter=5) + Self-RAG + AutoRAG
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
SELFRAG_MODEL="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface/models--selfrag--selfrag_llama2_13b/snapshots/3d308b05d77bcff0e199b210bece367461173163"
AUTORAG_MODEL="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface/models--ICTNLP--Auto-RAG-Llama-3-8B-Instruct/snapshots/b53199630d98c237634178ee2ac5dcf5d68839c2"

DATASETS="popqa hotpotqa 2wikimultihopqa bamboogle musique"

check_done() {
    find "$1" -name "metric_score.txt" 2>/dev/null | grep -q .
}

# ============================================================
# IRCoT (max_iter=5)
# ============================================================
echo "=============================================="
echo "IRCoT (max_iter=5)"
echo "=============================================="
for ds in $DATASETS; do
    tag="ircot_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] IRCoT $ds — already done"
        cat "$result_dir"/*/metric_score.txt
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
    cat "$result_dir"/*/metric_score.txt
done

# ============================================================
# Self-RAG (Llama2-13B)
# ============================================================
echo ""
echo "=============================================="
echo "Self-RAG"
echo "=============================================="
for ds in $DATASETS; do
    tag="selfrag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] Self-RAG $ds — already done"
        cat "$result_dir"/*/metric_score.txt
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
    cat "$result_dir"/*/metric_score.txt
done

# ============================================================
# AutoRAG (Llama3-8B)
# ============================================================
echo ""
echo "=============================================="
echo "AutoRAG"
echo "=============================================="
for ds in $DATASETS; do
    tag="autorag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if check_done "$result_dir"; then
        echo "[SKIP] AutoRAG $ds — already done"
        cat "$result_dir"/*/metric_score.txt
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
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
