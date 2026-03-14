# PRM Test-Time Scaling Experiment

## Goal
128개 trajectory를 생성하고, K=1,2,4,8,16,32,64,128에서 다양한 선택 전략의 **F1** 성능을 비교.
핵심: **우리 PRM이 test-time에 trajectory 선택을 잘 하는가?** (ReasonRAG와의 차별점)

**Metric: Token-level F1** (EM도 함께 기록)

## Datasets
| Dataset | Total Test | Sample | Data Path |
|---------|-----------|--------|-----------|
| PopQA | 14,267 | 500 | `data/flashrag/popqa/test.jsonl` |
| HotpotQA | 7,405 | 500 | `data/flashrag/hotpotqa/test.jsonl` |
| 2WikiMultiHopQA | 12,576 | 500 | `data/flashrag/2wikimultihopqa/test.jsonl` |
| MuSiQue | 2,417 | 500 | `data/flashrag/musique/test.jsonl` |

**Bamboogle 제외** (데이터 소량)

## Models

### Policy Model
- `Qwen/Qwen2.5-7B-Instruct` (base model)
- Temperature: 0.8 (diversity 확보), 128 samples/question

### Critic Models (비교 대상)
| Version | Path | Training Data | Base |
|---------|------|--------------|------|
| v8 (2,000q) | `outputs/critic_model_v8_2000q/final_model` | 109,664 samples | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` + LoRA |
| v9 (3,000q) | `outputs/critic_model_v9_3000q/final_model` | 154,511 samples | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` + LoRA |

### External PRMs
| Model | HuggingFace ID | Type |
|-------|---------------|------|
| VersaPRM | `UW-Madison-Lee-Lab/VersaPRM-Base-8B` | Step-level PRM (Llama-based) |
| MathPRM | `Qwen/Qwen2.5-Math-PRM-7B` | Step-level PRM (Qwen2.5-Math) |

## Scoring Methods (총 5개)
1. **Majority Voting (MV)** — 최다 답변 선택
2. **VersaPRM** — BoN + WMV (avg/min aggregation)
3. **MathPRM** — BoN + WMV (avg/min aggregation)
4. **Our Critic v8** (2,000q) — BoN + WMV (avg/min)
5. **Our Critic v9** (3,000q) — BoN + WMV (avg/min)

## K Values
`K = 1, 2, 4, 8, 16, 32, 64, 128`

K=1은 greedy baseline (첫 번째 trajectory). 각 K에서 처음 K개 trajectory만 사용하여 선택.

## Pipeline

### Step 1: Data Preparation
FlashRAG format → generate_trajectories.py format 변환 필요.
FlashRAG는 `golden_answers` (list), generate_trajectories.py는 `answer` (string) 사용.

```bash
# 각 데이터셋에서 500개 샘플링 (seed=42)
python scripts/prepare_prm_data.py \
    --datasets popqa hotpotqa 2wikimultihopqa musique \
    --limit 500 \
    --seed 42 \
    --output-dir data/prm_scaling
```

### Step 2: Trajectory Generation (128 per question)
각 데이터셋 별로 개별 실행. 약 500 × 128 = 64,000 trajectories/dataset.

```bash
POLICY_MODEL="outputs/dpo_policy_mcts_v1/merged_model"

for ds in popqa hotpotqa 2wikimultihopqa musique; do
    python scripts/generate_trajectories.py \
        --data_path "data/prm_scaling/${ds}_500.jsonl" \
        --output_path "outputs/prm_scaling/trajectories_${ds}_128.jsonl" \
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
done
```

**예상 시간**: 데이터셋당 약 6-8시간 (GPU 2개 기준), 총 24-32시간

### Step 3: Scoring with All Methods
Critic v8, v9를 순차적으로 실행 (vLLM LoRA 교체).

```bash
for ds in popqa hotpotqa 2wikimultihopqa musique; do
    TRAJ="outputs/prm_scaling/trajectories_${ds}_128.jsonl"

    # Critic v9 (3,000q)
    python scripts/compare_voting_methods.py \
        --trajectories "$TRAJ" \
        --critic-model "outputs/critic_model_v9_3000q/final_model" \
        --skip-versaprm \
        --output "outputs/prm_scaling/scores_${ds}_critic_v9.json"

    # Critic v8 (2,000q)
    python scripts/compare_voting_methods.py \
        --trajectories "$TRAJ" \
        --critic-model "outputs/critic_model_v8_2000q/final_model" \
        --skip-versaprm \
        --output "outputs/prm_scaling/scores_${ds}_critic_v8.json"

    # VersaPRM
    python scripts/compare_voting_methods.py \
        --trajectories "$TRAJ" \
        --skip-critic \
        --output "outputs/prm_scaling/scores_${ds}_versaprm.json"

    # MathPRM (새로 추가)
    python scripts/compare_voting_methods.py \
        --trajectories "$TRAJ" \
        --skip-critic \
        --skip-versaprm \
        --use-mathprm \
        --output "outputs/prm_scaling/scores_${ds}_mathprm.json"
done
```

### Step 4: Scaling Curve Analysis
K=1,2,4,8,16,32,64,128에서 EM 계산.

```bash
python scripts/analyze_prm_scaling.py \
    --results-dir outputs/prm_scaling/ \
    --datasets popqa hotpotqa 2wikimultihopqa musique \
    --k-values 1 2 4 8 16 32 64 128 \
    --output outputs/prm_scaling/scaling_analysis.json
```

## Code Modifications Required

### 1. `scripts/compare_voting_methods.py` — MathPRM 추가
- `--use-mathprm` flag 추가
- `--skip-mathprm` flag 추가
- `Qwen/Qwen2.5-Math-PRM-7B` 로드: `AutoModel` + `trust_remote_code=True`
- Step separator: `<extra_0>` 토큰
- Scoring: `softmax(logits)[:, :, 1]` at `<extra_0>` positions
- `mathprm_bon_{avg,min}`, `mathprm_wmv_{avg,min}` 결과 추가

### 2. `scripts/prepare_prm_data.py` — 새로 생성
- FlashRAG format (`golden_answers` list) → raw format (`answer` string) 변환
- Seed-based sampling (500개)
- 출력: `data/prm_scaling/{dataset}_500.jsonl`

### 3. `scripts/analyze_prm_scaling.py` — 새로 생성
- Per-trajectory JSONL에서 K별 subset 추출
- 각 K에서 MV, BoN, WMV 재계산
- Scaling curve 그래프 생성 (matplotlib)
- 결과 테이블 출력

## Expected Output

### Table Format
```
K  | MV    | Critic_v8_BoN | Critic_v8_WMV | Critic_v9_BoN | Critic_v9_WMV | VersaPRM_BoN | MathPRM_BoN
1  | xx.x  | xx.x          | xx.x          | xx.x          | xx.x          | xx.x         | xx.x
2  | xx.x  | ...
...
128| xx.x  | ...
```

### Key Metrics
- **v8 vs v9**: 어떤 critic이 BoN/WMV에서 더 좋은지
- **Scaling slope**: K가 늘어남에 따라 각 method의 성능 증가율
- **Saturation point**: 어느 K에서 성능이 plateau되는지
- **Our PRM vs baselines**: MV 대비 개선폭, VersaPRM/MathPRM 대비 우위

## Critic v8 기존 결과 (HotpotQA)

### 500q × 16s (soft scores)
- Trajectory: `outputs/archive/critic_scores_v8_val500_per_trajectory.jsonl`

| Method | EM | F1 |
|--------|------|------|
| Majority Voting | 45.80% | 54.90% |
| Critic v8 BoN (avg) | 49.40% | 58.23% |
| Critic v8 BoN (min) | 48.00% | 57.34% |
| Critic v8 WMV (avg) | 47.40% | 56.43% |
| **Critic v8 WMV (min)** | **49.00%** | **58.64%** |

### 500q × 128s (soft scores)
- Trajectory: `outputs/archive/voting_hotpotqa_per_trajectory.jsonl`

| Method | EM | F1 |
|--------|------|------|
| Majority Voting | 46.20% | 54.07% |
| Critic v8 BoN (avg) | — | 58.28% |
| Critic v8 BoN (min) | — | 59.07% |
| Critic v8 WMV (avg) | — | 55.89% |
| **Critic v8 WMV (min)** | — | **59.50%** |
| VersaPRM BoN (avg) | 45.20% | 51.77% |
| VersaPRM WMV (avg) | 47.00% | — |

### v9 비교 실행 커맨드
```bash
# 500q × 128s val trajectory에 v9 채점
python scripts/compare_voting_methods.py \
    --trajectories outputs/archive/hotpotqa_val_500q_128s.jsonl \
    --critic-model outputs/critic_model_v9_3000q/final_model \
    --skip-versaprm \
    --output outputs/prm_scaling/hotpotqa_critic_v9_comparison.json
```

**주의: 반드시 validation 데이터만 사용. train 데이터 사용 금지.**

## Notes
- **Metric: F1 (primary), Cover EM (secondary)** — `compare_voting_methods.py`에 F1 추가 완료
- Temperature 0.8은 diversity 확보용 (eval은 temperature 0이지만 trajectory 생성은 다양성 필요)
- Resume 지원: trajectory 생성과 critic 채점 모두 중단 후 재개 가능
- GPU 메모리: critic 채점 시 `--gpu-memory 0.9` 사용
