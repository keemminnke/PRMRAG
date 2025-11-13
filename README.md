# PRMRAG: Consensus-Based Auto-Labeling Pipeline for RAG-CoT

자동 라벨링 파이프라인: MC/RPE 신호(작은 모델) + LLM Judge 신호(큰 모델)를 결합하여 consensus 기반으로 고품질 학습 데이터를 생성합니다.

## 목표

사람 개입 없이:
- 작은 모델의 MC/RPE 신호 + 큰 모델의 judge 신호를 합쳐서
- 서로 동의하는 step만 남기고 (consensus)
- 남은 step들에는 이미 0/1 라벨이 자동으로 붙어 있는 상태까지 가는 파이프라인

**핵심**: "필터링 과정 = 라벨링 과정"

## 파이프라인 구조

```
RAG-CoT Trajectories (HotpotQA)
    ↓
┌─────────────────────────────────────┐
│  MC-based RPE Labeler               │
│  - 작은 모델로 rollout              │
│  - Monte Carlo estimation           │
│  - RPE 계산 및 threshold 기반 라벨  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  LLM Judge Labeler                  │
│  - 큰 모델 (70B+)                   │
│  - VersaPRM 스타일 평가             │
│  - GOOD/BAD 라벨                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Consensus Module                   │
│  - 두 신호 비교                     │
│  - 합의된 step만 유지               │
│  - 자동 라벨 확정 (0/1)             │
│  - 충돌/애매한 step 제거            │
└─────────────────────────────────────┘
    ↓
고품질 라벨 완비 데이터셋
```

## 주요 특징

### 1. MC-based RPE Labeling (작은 모델)
- Prefix 고정 후 K번 rollout
- MC(s_t, a_t) / MC(s_t) 계산
- Binary threshold 기반 자동 라벨링 (GOOD/BAD)

### 2. LLM Judge Labeling (큰 모델)
- VersaPRM 스타일 평가
- Gold answer + supporting facts 활용
- Step별 GOOD/BAD 판정

### 3. Consensus-based Filtering
- **합의된 긍정**: RPE=GOOD + Judge=GOOD → label=1
- **합의된 부정**: RPE=BAD + Judge=BAD → label=0
- **불일치**: RPE와 Judge가 다르면 → 제거 (데이터셋에서 제외)

## 설치

```bash
pip install -e .
```

또는:

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 전체 파이프라인 실행

```bash
python scripts/label_dataset.py \
    --config configs/default.yaml \
    --input data/raw/hotpotqa_trajectories.jsonl \
    --output data/labeled/consensus_labeled.jsonl
```

### 2. RPE 라벨링만

```bash
python scripts/rpe_labeling.py \
    --config configs/rpe_labeling.yaml \
    --input data/raw/trajectories.jsonl \
    --output data/processed/rpe_labels.jsonl
```

### 3. Judge 라벨링만

```bash
python scripts/judge_labeling.py \
    --config configs/judge_labeling.yaml \
    --input data/raw/trajectories.jsonl \
    --output data/processed/judge_labels.jsonl
```

### 4. Consensus 분석

```bash
python scripts/consensus_analysis.py \
    --rpe data/processed/rpe_labels.jsonl \
    --judge data/processed/judge_labels.jsonl \
    --output outputs/metrics/consensus_analysis.json
```

## 설정

`configs/default.yaml`에서 파라미터 조정:

```yaml
labeling:
  rpe:
    model_name: "llama-2-7b"
    num_rollouts: 5
    threshold: 0.5  # Binary: >= 0.5 → GOOD, < 0.5 → BAD

  judge:
    model_name: "llama-2-70b"
    temperature: 0.3
    max_retries: 3

  consensus:
    strategy: "strict"  # Both must agree
    min_agreement: 0.8
    trajectory_level: true  # trajectory 전체 버리기
```

## 프로젝트 구조

```
PRMRAG/
├── src/prmrag/
│   ├── data/           # 데이터 처리 및 스키마
│   ├── models/         # 모델 래퍼
│   ├── labeling/       # 핵심 라벨링 로직
│   │   ├── rpe_labeler.py
│   │   ├── judge_labeler.py
│   │   └── consensus.py
│   ├── evaluation/     # 평가 도구
│   └── utils/          # 유틸리티
├── scripts/            # 실행 스크립트
├── configs/            # 설정 파일
└── tests/              # 테스트
```

## 참고 논문 & 코드

- [VersaPRM](https://github.com/UW-Madison-Lee-Lab/VersaPRM)
- [GenPRM](https://github.com/RyanLiu112/GenPRM)

## 라이선스

MIT
