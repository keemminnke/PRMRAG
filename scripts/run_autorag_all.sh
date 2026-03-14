#!/bin/bash
# Run AutoRAG eval on all 5 datasets
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

export NUMEXPR_MAX_THREADS=64
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface

AUTORAG_MODEL="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface/models--ICTNLP--Auto-RAG-Llama-3-8B-Instruct/snapshots/b53199630d98c237634178ee2ac5dcf5d68839c2"

for ds in popqa hotpotqa 2wikimultihopqa bamboogle musique; do
    tag="autorag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
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
echo "ALL AUTORAG DONE"
