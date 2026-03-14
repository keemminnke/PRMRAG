#!/bin/bash
# PRM Test-Time Scaling Experiment
# See docs/prm_scaling_experiment.md for full plan
set -e
cd /home/work/.conda/storage/MINKEON_KIM/PRMRAG

POLICY_MODEL="Qwen/Qwen2.5-7B-Instruct"
CRITIC_V8="outputs/critic_model_v8_2000q/final_model"
DATASETS="popqa hotpotqa 2wikimultihopqa"

mkdir -p data/prm_scaling outputs/prm_scaling

# =============================================
# Step 1: Prepare data (500 samples per dataset)
# =============================================
echo "===== Step 1: Prepare data ====="
python scripts/prepare_prm_data.py \
    --datasets $DATASETS \
    --limit 500 \
    --seed 42 \
    --output-dir data/prm_scaling

# =============================================
# Step 2: Generate 128 trajectories per question
# =============================================
echo ""
echo "===== Step 2: Generate trajectories ====="
for ds in $DATASETS; do
    TRAJ_FILE="outputs/prm_scaling/trajectories_${ds}_128.jsonl"
    if [ -f "$TRAJ_FILE" ]; then
        count=$(wc -l < "$TRAJ_FILE")
        expected=$((500 * 128))
        if [ "$count" -ge "$expected" ]; then
            echo "[SKIP] $ds — $count trajectories already generated"
            continue
        fi
        echo "[RESUME] $ds — $count/$expected trajectories, resuming..."
    fi
    echo "[RUN] $ds"
    python scripts/generate_trajectories.py \
        --data_path "data/prm_scaling/${ds}_500.jsonl" \
        --output_path "$TRAJ_FILE" \
        --policy_model "$POLICY_MODEL" \
        --num_samples 128 \
        --batch_size 20 \
        --limit 500 \
        --temperature 0.8 \
        --max_steps 10 \
        --top_k 5 \
        --retriever bge \
        --no_rerank \
        --resume
    echo "[DONE] $ds"
done

# =============================================
# Step 3: Score with all methods
# =============================================
echo ""
echo "===== Step 3: Score trajectories ====="
for ds in $DATASETS; do
    TRAJ="outputs/prm_scaling/trajectories_${ds}_128.jsonl"

    # Critic v8 (2,000q)
    OUT="outputs/prm_scaling/scores_${ds}_critic_v8.json"
    if [ ! -f "$OUT" ]; then
        echo "[RUN] Critic v8 on $ds"
        python scripts/compare_voting_methods.py \
            --trajectories "$TRAJ" \
            --critic-model "$CRITIC_V8" \
            --skip-versaprm \
            --output "$OUT"
    else
        echo "[SKIP] Critic v8 on $ds — already done"
    fi

    # VersaPRM
    OUT="outputs/prm_scaling/scores_${ds}_versaprm.json"
    if [ ! -f "$OUT" ]; then
        echo "[RUN] VersaPRM on $ds"
        python scripts/compare_voting_methods.py \
            --trajectories "$TRAJ" \
            --skip-critic \
            --output "$OUT"
    else
        echo "[SKIP] VersaPRM on $ds — already done"
    fi

    # MathPRM
    OUT="outputs/prm_scaling/scores_${ds}_mathprm.json"
    if [ ! -f "$OUT" ]; then
        echo "[RUN] MathPRM on $ds"
        python scripts/compare_voting_methods.py \
            --trajectories "$TRAJ" \
            --skip-critic \
            --skip-versaprm \
            --use-mathprm \
            --output "$OUT"
    else
        echo "[SKIP] MathPRM on $ds — already done"
    fi
done

# =============================================
# Step 4: Scaling Curve Analysis
# =============================================
echo ""
echo "===== Step 4: Scaling curve analysis ====="
python scripts/analyze_prm_scaling.py \
    --results-dir outputs/prm_scaling/ \
    --datasets $DATASETS \
    --k-values 1 2 4 8 16 32 64 128 \
    --output outputs/prm_scaling/scaling_analysis.json

echo ""
echo "===== ALL PRM SCALING EXPERIMENTS COMPLETE ====="
echo "Results in: outputs/prm_scaling/"
echo "Plots in:   outputs/prm_scaling/plots/"
