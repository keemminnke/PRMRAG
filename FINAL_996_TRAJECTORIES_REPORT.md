# 996개 궤적 생성 완료 보고서

## 실행 일시
2025-12-11

## 데이터 생성 완료

### 파일 정보
- **최종 파일**: `outputs/hybrid_1000q_from_0/results_hybrid_all_996.jsonl`
- **총 궤적**: 996개
- **파일 크기**: 6.4MB

### 구성
1. **기존 데이터**: 589개 (재파싱 완료)
2. **신규 데이터**: 407개 (재파싱 완료)
3. **총합**: 996개

## 📊 전체 통계

### 정확도
- **Total**: 996 trajectories
- **정답**: 492개 (49.4%)
- **오답**: 504개 (50.6%)

### Retrieval 분석
| 타입 | 개수 | 정답률 |
|------|------|--------|
| **With RAG** | 792 (79.5%) | 293/792 (37.0%) |
| **CoT Only** | 204 (20.5%) | 199/204 (97.5%) |

### 주요 발견

#### ✅ 강점
1. **CoT 성능 우수**: CoT-only 97.5% 정확도
   - 모델의 내재적 추론 능력 매우 강함
   - 검색 없이도 대부분의 질문 해결 가능

2. **Label 분포 균형**: Good 59.6%, Bad 40.4%
   - PRM 학습에 적합한 분포
   - 양성/음성 샘플 모두 충분

#### ⚠️ 개선 필요
1. **RAG 정확도 낮음**: 37.0%
   - 전체 정확도(49.4%)보다 낮음
   - 검색이 오히려 성능 저하 유발
   - **원인 분석 필요**:
     - 검색 쿼리 품질
     - 검색된 문서의 관련성
     - 검색 결과 통합 방식

2. **MC stuck at 0.0**: 45.8% (456개)
   - Rollout에서 정답 추출 실패
   - 또는 정답 비교 로직 문제
   - **확인 필요**: Answer extraction 함수

## 📈 Monte Carlo 분석

### MC Score 분포
| 범위 | 개수 | 비율 |
|------|------|------|
| **High (0.8-1.0)** | 444 | 44.6% |
| **Medium (0.3-0.8)** | 75 | 7.5% |
| **Low (0.0-0.3)** | 477 | 47.9% |

### 특징
- **이분화된 분포**: High vs Low
- Medium 범위가 매우 적음 (7.5%)
- MC=0.0이 가장 큰 비중 (45.8%)

## 🔢 Step 통계

### 전체 Step
- **총 단계**: 4,777개
- **CoT 단계**: 3,641개 (76.2%)
- **RAG 단계**: 1,136개 (23.8%)
- **평균 단계/궤적**: 4.8

### Label 분포
| Label | 개수 | 비율 |
|-------|------|------|
| **Good** | 2,846 | 59.6% |
| **Bad** | 1,931 | 40.4% |

## 💡 핵심 인사이트

### 1. RAG 성능 문제
**현상**: RAG 사용 시 정확도 37.0% (CoT 97.5%보다 훨씬 낮음)

**가능한 원인**:
1. **검색 쿼리 생성 품질**
   - 모델이 적절한 검색 쿼리를 생성하지 못함
   - 너무 구체적이거나 너무 일반적인 쿼리

2. **검색 결과 관련성**
   - Hybrid retrieval (BM25 + Dense)이 관련 문서를 찾지 못함
   - 또는 관련 문서가 KB에 없음

3. **검색 결과 통합**
   - 모델이 검색 결과를 올바르게 해석/활용하지 못함
   - Irrelevant 정보가 추론을 방해

**권장 조치**:
- RAG 단계의 검색 쿼리 샘플 분석
- 검색 결과의 관련성 평가
- RAG가 필요 없는 질문을 CoT로 처리하도록 개선

### 2. MC=0.0 문제
**현상**: 45.8% 궤적에서 MC가 0.0

**가능한 원인**:
1. **Answer Extraction 실패**
   - Rollout에서 생성한 답변을 추출하지 못함
   - `extract_answer_from_text()` 함수 검토 필요

2. **Answer Matching 실패**
   - 정답과 예측을 비교하는 로직 문제
   - `check_answer_match()` 함수 검토 필요

3. **Rollout 품질**
   - Rollout이 실제로 정답을 생성하지 못함
   - Temperature, max_tokens 등 파라미터 조정 필요

**권장 조치**:
- MC=0.0 샘플 수동 검토
- Rollout 결과와 정답 비교 로직 검증
- Rollout 생성 품질 개선

### 3. CoT 우수 성능
**현상**: CoT-only 97.5% 정확도

**의미**:
- 모델(Qwen2.5-7B)의 추론 능력 매우 우수
- 많은 질문이 검색 없이 해결 가능
- 검색이 필요한 질문과 불필요한 질문 구분 필요

**활용 방안**:
- Dynamic K를 더 공격적으로 조정
  - Easy 질문: 더 적은 RAG 호출
  - Hard 질문: 더 많은 RAG 호출
- CoT 우선 시도 후 필요 시 RAG 사용

## 🎯 다음 단계

### 즉시 가능
1. ✅ **데이터 준비 완료**: 996개 궤적
2. ✅ **재파싱 완료**: CoT 필드 구조화
3. ✅ **Label 분포 확인**: 60:40 균형

### Step 2: Consensus Labeling
```bash
python3 scripts/run_consensus_labeling.py \
    outputs/hybrid_1000q_from_0/results_hybrid_all_996.jsonl \
    --output outputs/training_data/consensus_labeled_996.jsonl \
    --rpe-model Qwen/Qwen2.5-7B-Instruct \
    --judge-model Qwen/Qwen2.5-72B-Instruct
```

**예상 결과**:
- 총 단계: ~4,777
- Agreement rate: ~85%
- 유지될 단계: ~4,060 (85%)
- 필터될 단계: ~717 (15%)

### 추가 개선 (선택)
1. **RAG 품질 분석**
   - 검색 쿼리 샘플링 및 평가
   - 검색 결과 관련성 분석
   - RAG vs CoT 성능 비교 심화

2. **MC=0.0 문제 해결**
   - Answer extraction 로직 검토
   - Rollout 샘플 수동 분석
   - 파라미터 튜닝

3. **Dynamic K 최적화**
   - Easy/Medium/Hard 임계값 재조정
   - CoT 우선 전략 실험

## 파일 목록

### 원본 데이터
- `results_hybrid_20251209_072136.jsonl` (589개, 재파싱됨)
- `results_hybrid_20251210_104431.jsonl` (407개, 원본)

### 재파싱 데이터
- `results_hybrid_20251210_104431_reparsed.jsonl` (407개, 재파싱됨)

### 최종 통합 데이터
- **`results_hybrid_all_996.jsonl`** (996개, 최종 데이터) ✅

### 분석 보고서
- `results_hybrid_all_996_quick_summary.txt` (통계 요약)
- `FINAL_996_TRAJECTORIES_REPORT.md` (이 문서)

## 사용 명령어

### 통계 분석
```bash
# 빠른 통계
python3 scripts/quick_stats.py outputs/hybrid_1000q_from_0/results_hybrid_all_996.jsonl

# 상세 분석 (대화형)
python3 scripts/analyze_results.py outputs/hybrid_1000q_from_0/results_hybrid_all_996.jsonl
```

### Consensus Labeling
```bash
python3 scripts/run_consensus_labeling.py \
    outputs/hybrid_1000q_from_0/results_hybrid_all_996.jsonl \
    --output outputs/training_data/consensus_labeled_996.jsonl \
    --limit 10  # 테스트
```

## 결론

✅ **996개 궤적 생성 완료**
- 전체 정확도: 49.4%
- CoT 성능 우수: 97.5%
- Label 분포 균형: 60:40

⚠️ **개선 필요 사항**
- RAG 정확도 낮음 (37.0%)
- MC stuck at 0.0 (45.8%)

🎯 **다음 작업**
- Step 2: Consensus Labeling 실행
- RAG 품질 분석 및 개선
- PRM 모델 학습 준비

데이터는 학습에 사용할 준비가 되었습니다!
