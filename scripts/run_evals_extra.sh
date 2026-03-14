#!/bin/bash
# Extra baselines: HiPRAG + Direct (No RAG)
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

HIPRAG_MODEL="qualidea1217/Qwen2.5-7B-Instruct-PPO-HiPRAG"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

DATASETS="popqa hotpotqa 2wikimultihopqa bamboogle musique"

HIPRAG_SYSTEM_PROMPT='Answer user questions by thinking step-by-step. Your entire reasoning process must be encapsulated within a single <think></think> block, which contains one or more <step></step> blocks. Each step must begin with your analysis in <reasoning>. If you identify a knowledge gap, you may use <search>query</search> to query a search engine; search results will then be provided in a <context> tag. Every step must end with a <conclusion> summarizing what you learned in that step. After your thinking process is complete, provide the final, conclusive answer inside an <answer> tag placed immediately after the closing </think> tag. You can use as many steps as you need. Ensure all XML tags are properly formed and nested.

**## Output Format Specification**

Your output must follow this overall structure. The `<think>` block contains all the steps, and the `<answer>` block follows it.

<think>
<step>
    ...
</step>
<step>
    ...
</step>
</think>
<answer>Your final, conclusive answer to the user'"'"'s question.</answer>

**## Step Formats (to be used inside <think>)**

Format 1: Step with a Search

<step>
    <reasoning>Your detailed analysis of what you know and what you need to find out.</reasoning>
    <search>The precise search query you will use.</search>
    <context>[This will be provided by the system after your search]</context>
    <conclusion>Your conclusion or answer to the reasoning and search query at this step.</conclusion>
</step>

Format 2: Step without a Search (Internal Reasoning)

<step>
    <reasoning>Your detailed analysis of what you know and what you need to find out.</reasoning>
    <conclusion>Your conclusion or answer to the reasoning at this step.</conclusion>
</step>'

echo "=============================================="
echo "PHASE 5: HiPRAG (BGE retriever)"
echo "=============================================="
for ds in $DATASETS; do
    tag="hiprag_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] HiPRAG $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] HiPRAG $ds"
    python scripts/eval_flashrag.py \
        --model "$HIPRAG_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0 \
        --query-begin "<search>" \
        --query-end "</search>" \
        --doc-begin "<context>" \
        --doc-end "</context>" \
        --system-prompt "$HIPRAG_SYSTEM_PROMPT"
    echo "[DONE] HiPRAG $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "PHASE 6: Direct (No RAG) — Qwen2.5-7B-Instruct"
echo "=============================================="
for ds in $DATASETS; do
    tag="direct_${ds}"
    result_dir="outputs/flashrag_${tag}"
    if find "$result_dir" -name "metric_score.txt" 2>/dev/null | grep -q .; then
        echo "[SKIP] Direct $ds — already done"
        cat "$result_dir"/*/metric_score.txt
        continue
    fi
    echo "[RUN] Direct $ds"
    python scripts/eval_flashrag_direct.py \
        --model "$QWEN_MODEL" \
        --tag "$tag" \
        --dataset "$ds" \
        --temperature 0
    echo "[DONE] Direct $ds"
    cat "$result_dir"/*/metric_score.txt
done

echo ""
echo "=============================================="
echo "ALL EXTRA EVALUATIONS COMPLETE"
echo "=============================================="
