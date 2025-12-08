# PRMRAG 프로젝트 분석: 논문 작성을 위한 가이드

## 1. 프로젝트 개요

이 프로젝트는 **RAG(Retrieval-Augmented Generation) 시스템의 성능을 향상시키기 위한 프로세스 기반 보상 모델(Process-based Reward Model, PRM)의 학습 데이터를 자동으로 생성**하는 것을 목표로 하는 연구 프레임워크입니다.

핵심 아이디어는 **'합의 기반 자동 라벨링(Consensus-Based Auto-Labeling)'**입니다. 이는 두 개의 서로 다른 AI 모델(하나는 빠르고 작은 모델, 다른 하나는 크고 정확한 모델)을 사용하여 생성된 추론 과정(reasoning trajectory)의 각 단계를 평가하고, 두 모델의 의견이 일치하는 경우에만 해당 데이터를 '고품질' 학습 데이터로 채택하는 방식입니다. 이를 통해 사람의 개입 없이도 신뢰도 높은 PRM 학습 데이터셋을 구축할 수 있습니다.

이 문서는 코드베이스를 분석하여 프로젝트의 핵심 구조, 주요 컴포넌트, 그리고 실행 흐름을 설명함으로써 논문 작성에 필요한 깊이 있는 이해를 돕기 위해 작성되었습니다.

## 2. 핵심 아키텍처

프로젝트의 전체 아키텍처는 다음과 같이 요약할 수 있습니다.

```mermaid
graph TD
    subgraph 1단계: 적응형 궤적 생성
        A[시작 질문] --> B{AdaptiveTrajectoryGenerator};
        B -- RPE 기반 판단 --> C[CoT 추론];
        B -- RPE 기반 판단 --> D[RAG 검색];
        C --> E[추론 단계 생성];
        D -- HybridRetriever<br>(BM25+BGE) --> E;
        E --> F[원시 궤적 데이터<br>(Raw Trajectories)];
    end

    subgraph 2단계: 합의 기반 자동 라벨링
        F --> G[라벨링 파이프라인];
        G --> H{RPELabeler<br>(정량적 평가)};
        G --> I{JudgeLabeler<br>(정성적 평가)};
        H --> J[RPE 점수];
        I --> K[Judge 판정<br>('GOOD'/'BAD')];
        J & K --> L{ConsensusModule};
        L -- 동의하는 샘플 --> M[고품질 라벨링 데이터<br>(Final Labeled Dataset)];
        L -- 불일치하는 샘플 --> N[필터링됨];
    end

    M --> O[PRM 모델 학습];
```

## 3. 주요 파이프라인 (2단계)

### 1단계: 적응형 궤적 생성 (Adaptive Trajectory Generation)

이 단계에서는 최종적으로 라벨링될 '데이터'를 생성합니다.

-   **`AdaptiveTrajectoryGenerator`** (`src/prmrag/generation/adaptive_generator.py`)
    -   이 모듈은 질문에 대한 답변을 생성하기 위해 추론 경로, 즉 '궤적'을 만듭니다.
    -   각 단계에서 순수한 논리적 추론(Chain-of-Thought, CoT)을 계속할지, 아니면 외부 정보 소스를 참조할지(Retrieval-Augmented Generation, RAG)를 **동적으로 결정**합니다.
    -   이 결정은 **RPE (Relative Preference Estimation)** 점수를 실시간으로 계산하여 이루어집니다. 만약 현재 추론의 품질이 특정 임계값 아래로 떨어지면, 시스템은 RAG를 통해 외부 정보를 가져와 추론의 품질을 보강하려고 시도합니다.

-   **`HybridRetriever`** (`src/prmrag/retrieval/hybrid_retriever.py`)
    -   RAG가 트리거될 때 사용되는 검색기입니다.
    -   전통적인 키워드 기반 검색(**BM25**)과 최신 의미론적 검색(**BGE 임베딩**)을 결합한 하이브리드 방식입니다.
    -   두 검색 결과는 **Reciprocal Rank Fusion (RRF)** 알고리즘을 통해 융합되어 더 정확하고 관련성 높은 문서를 찾아냅니다.

### 2단계: 합의 기반 자동 라벨링 (Consensus-Based Auto-Labeling)

이 단계는 이 프로젝트의 핵심적인 기여(novelty)로, 1단계에서 생성된 원시 궤적의 각 단계를 자동으로 평가하고 라벨링합니다.

-   **두 개의 독립적인 평가 신호**
    1.  **`RPELabeler`** (`src/prmrag/labeling/rpe_labeler.py`)
        -   **정량적 평가**를 수행합니다.
        -   비교적 작은 언어 모델을 사용하여 특정 추론 단계를 포함했을 때와 제외했을 때의 성공 확률을 **몬테카를로(Monte Carlo) 롤아웃**을 통해 각각 추정합니다.
        -   이 두 확률의 비율(RPE)을 계산하여 해당 단계가 얼마나 유용한지를 점수로 나타냅니다.

    2.  **`JudgeLabeler`** (`src/prmrag/labeling/judge_labeler.py`)
        -   **정성적 평가**를 수행합니다.
        -   크고 강력한 언어 모델(예: GPT-5)을 '심판(Judge)'으로 활용합니다.
        -   `VersaPRM` 스타일의 상세한 프롬프트를 통해 질문, 전체 문맥, 정답, 그리고 해당 추론 단계를 제공하고, 이 단계가 'GOOD'인지 'BAD'인지 판정하도록 요청합니다.

-   **`ConsensusModule`** (`src/prmrag/labeling/consensus.py`)
    -   이 모듈은 위 두 평가자의 결과를 취합하여 최종 라벨을 결정합니다.
    -   **"필터링이 곧 라벨링이다 (Filtering is Labeling)"**라는 철학을 따릅니다.
    -   `RPELabeler`의 점수와 `JudgeLabeler`의 판정이 **서로 일치하는 경우에만** 해당 데이터 포인트를 최종 데이터셋에 포함시키고, 긍정(1) 또는 부정(0) 라벨을 할당합니다.
    -   두 평가자의 의견이 다를 경우, 해당 데이터는 신뢰도가 낮다고 판단하여 **과감히 폐기**합니다. 이 과정을 통해 데이터의 노이즈를 최소화하고 높은 신뢰도를 확보합니다.

## 4. 핵심 코드 컴포넌트

-   `scripts/label_dataset.py`: 전체 자동 라벨링 파이프라인(2단계)을 실행하는 메인 스크립트.
-   `scripts/generate_adaptive_trajectories.py`: 적응형 궤적 생성(1단계)을 실행하는 스크립트.
-   `src/prmrag/generation/adaptive_generator.py`: CoT와 RAG를 동적으로 전환하며 추론 궤적을 생성하는 핵심 로직.
-   `src/prmrag/labeling/consensus.py`: 두 평가자(RPE, Judge)의 결과를 종합하여 최종 라벨을 결정하는 합의(Consensus) 로직.
-   `src/prmrag/labeling/rpe_labeler.py`: 몬테카를로 롤아웃 기반의 정량적 평가자.
-   `src/prmrag/labeling/judge_labeler.py`: 대형 언어 모델을 활용한 정성적 평가자.
-   `src/prmrag/retrieval/hybrid_retriever.py`: BM25와 BGE를 결합한 하이브리드 검색기.

## 5. 실행 흐름

1.  **1단계: 적응형 궤적 생성**
    ```bash
    python scripts/generate_adaptive_trajectories.py
    ```
    - 이 스크립트를 실행하여 원시 데이터(`raw_trajectories.jsonl`)를 생성합니다.

2.  **2단계: 합의 기반 라벨링**
    ```bash
    python scripts/label_dataset.py
    ```
    - 이 스크립트는 생성된 원시 데이터를 입력으로 받아, `RPELabeler`와 `JudgeLabeler`를 차례로 실행한 뒤, `ConsensusModule`을 통해 최종적으로 라벨링된 고품질 데이터셋(`labeled_dataset.jsonl`)을 출력합니다.

## 6. 논문 기여점 (Thesis Contribution)

이 프로젝트의 핵심적인 학술적 기여는 다음과 같이 요약할 수 있습니다.

-   **PRM 학습 데이터 생성의 완전 자동화**: 기존의 PRM 연구들이 수작업(human annotation)에 크게 의존했던 것과 달리, 이 프레임워크는 **모델 간의 합의**라는 새로운 패러다임을 통해 데이터 라벨링 과정을 완전히 자동화했습니다. 이는 데이터 구축 비용을 획기적으로 절감하고 확장성을 높입니다.
-   **높은 신뢰도의 데이터셋 확보**: 서로 다른 강점을 가진 두 종류의 AI 평가자(빠른 정량 평가 + 정확한 정성 평가)를 교차 검증 장치로 활용하고, 의견이 불일치하는 모호한 샘플들을 제거함으로써 데이터의 노이즈를 최소화하고 신뢰도를 극대화했습니다.
-   **RAG 시스템 개선을 위한 독창적 접근**: 고품질의 프로세스 기반 보상 모델을 통해 RAG 시스템의 추론 과정을 단계별로 직접적으로 개선할 수 있는 기반을 마련했습니다. 이는 단순히 최종 답변의 품질만 보는 것보다 훨씬 세밀한 제어를 가능하게 합니다.

결론적으로, 이 프로젝트는 'AI가 AI를 감독하고 학습시키는' 효율적인 부트스트래핑(bootstrapping) 메커니즘을 제안하며, 이는 향후 RAG 및 LLM 에이전트 연구에 중요한 기여가 될 수 있습니다.
