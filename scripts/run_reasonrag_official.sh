#!/bin/bash
# ReasonRAG official pipeline eval (4-prompt multi-stage)
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

MODEL="outputs/reasonrag_instruct_merged"
DATASETS="popqa hotpotqa 2wikimultihopqa bamboogle musique"

echo "=============================================="
echo "ReasonRAG Official Pipeline (BGE retriever)"
echo "=============================================="
for ds in $DATASETS; do
    tag="reasonrag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] ReasonRAG $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] ReasonRAG $ds"
    python scripts/eval_flashrag_reasonrag.py \
        --model "$MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0 \
        --max-iter 8
    echo "[DONE] ReasonRAG $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "ReasonRAG Official Pipeline COMPLETE"
echo "=============================================="
