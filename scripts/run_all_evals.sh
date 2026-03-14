#!/bin/bash
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

V1_MODEL="outputs/dpo_policy_mcts_v1/merged_model"
SR1_MODEL="/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface/models--PeterJinGo--SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo/snapshots/74d46a43224daa612b5f27174e1c58b35587ee92"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
REASONRAG_MODEL="outputs/reasonrag_instruct_merged"

DATASETS="popqa hotpotqa 2wikimultihopqa bamboogle musique"

echo "=============================================="
echo "PHASE 1: V1 (PRO-Step) on all datasets"
echo "=============================================="
for ds in $DATASETS; do
    tag="v1_${ds}"
    result_dir="outputs/flashrag_${tag}"
    # Skip if already done
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] V1 $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] V1 $ds"
    python scripts/eval_flashrag.py \
        --model "$V1_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0
    echo "[DONE] V1 $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "PHASE 2: Search-R1 on all datasets"
echo "=============================================="
for ds in $DATASETS; do
    tag="searchr1_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] Search-R1 $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] Search-R1 $ds"
    python scripts/eval_flashrag.py \
        --model "$SR1_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0 \
        --doc-begin "<information>" \
        --doc-end "</information>" \
        --use-default-prompt
    echo "[DONE] Search-R1 $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "PHASE 3: Standard RAG on all datasets"
echo "=============================================="
for ds in $DATASETS; do
    tag="stdrag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] Standard RAG $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] Standard RAG $ds"
    python scripts/eval_flashrag_standard.py \
        --model "$QWEN_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0
    echo "[DONE] Standard RAG $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "PHASE 4: ReasonRAG on all datasets"
echo "=============================================="
REASONRAG_SYSTEM_PROMPT='You are a question-answering assistant with access to a retrieval tool. Your goal is to provide a concise and accurate reasoning process.
Instructions:
* Error Reflection: If errors exist in previous thoughts, identify and correct them. Skip this step if no errors are present.
* Information Sufficiency: Evaluate whether the current information is sufficient to fully and accurately answer the question. If additional retrieval is needed, deconstruct the question and generate the next query. Avoid repeating previous queries. If no meaningful new query can be generated, explain why and provide an answer based on the current information.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Conclusion: If generating an answer: "So the answer is <answer>answer</answer>". If more retrieval is needed: "So the next query is <query>query</query>".'

for ds in $DATASETS; do
    tag="reasonrag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] ReasonRAG $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] ReasonRAG $ds"
    python scripts/eval_flashrag.py \
        --model "$REASONRAG_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0 \
        --query-begin "<query>" \
        --query-end "</query>" \
        --system-prompt "$REASONRAG_SYSTEM_PROMPT"
    echo "[DONE] ReasonRAG $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=============================================="
