# Hybrid Retrieval Guide (BM25 + BGE-M3)

**날짜**: 2025-12-04
**목적**: DPR corpus에서 검색 품질 향상을 위한 BM25 + Dense 하이브리드 방식

---

## 🎯 왜 Hybrid Retrieval인가?

### DPR Dense-only의 문제점

이전 분석 결과 (batch_test_100q_final):
- **검색 실패율**: 82% (RPE < 0.5)
- **중앙값 RPE**: 0.000 (절반이 쓸모없음)
- **정확도**: RAG 사용 시 50%, 미사용 시 95%

**주요 문제**:
1. **Semantic-only 한계**: "7th largest country" → "7th Infantry Regiment" (숫자만 매칭)
2. **고유명사 실패**: "Oberoi Group" 정확한 키워드 매칭 필요
3. **DPR 100-word 분할**: 같은 문서 반복 (35개 케이스)

### Hybrid의 장점

```
BM25 (Sparse):
✅ 정확한 키워드 매칭
✅ 고유명사, 숫자 처리 우수
✅ 빠름

BGE-M3 (Dense):
✅ 의미적 유사도
✅ 동의어, 패러프레이즈 처리
✅ 문맥 이해

Hybrid = BM25 + BGE-M3:
✅ 정확도 + 의미 이해
✅ 20~40% 검색 향상 예상
```

---

## 📁 구현 완료된 파일들

### 1. Retrievers

```
src/prmrag/retrieval/
├── bm25_retriever.py       # BM25 sparse retriever
├── bge_retriever.py        # BGE-M3 dense retriever
└── hybrid_retriever.py     # Hybrid (BM25 + BGE)
```

### 2. 스크립트

```
scripts/
├── precompute_bm25_index.py    # BM25 인덱스 빌드
├── batch_test_hybrid.py        # 하이브리드 테스트 (새로 작성!)
└── batch_test_adaptive.py      # 기존 Dense-only 테스트
```

### 3. 데이터

```
data/
├── kilt/
│   └── kilt_knowledgesource.json    # KILT corpus (35GB) ✅ 완료
├── embeddings/
│   └── kilt_wikipedia_bge_m3.npy    # BGE embeddings (12GB) ✅ 진행 중
└── indexes/
    └── kilt_wikipedia_bm25.pkl      # BM25 index (빌드 필요)
```

---

## 🚀 사용 방법

### Step 1: BM25 인덱스 빌드

```bash
# 전체 KILT corpus (5.9M documents)
python3 scripts/precompute_bm25_index.py \
    --corpus-file data/kilt/kilt_knowledgesource.json \
    --output-index data/indexes/kilt_wikipedia_bm25.pkl

# 테스트용 (10K documents)
python3 scripts/precompute_bm25_index.py \
    --corpus-file data/kilt/kilt_knowledgesource.json \
    --output-index data/indexes/kilt_wikipedia_bm25_10k.pkl \
    --max-docs 10000
```

**예상 시간**:
- 10K docs: ~1분
- 100K docs: ~10분
- 5.9M docs: ~1-2시간

**메모리**: ~2GB per 1M docs

---

### Step 2: 하이브리드 테스트 실행

```bash
# 기본 테스트 (20 questions)
python3 scripts/batch_test_hybrid.py \
    --num-questions 20 \
    --output-dir outputs/hybrid_test

# 옵션 설명
python3 scripts/batch_test_hybrid.py \
    --num-questions 100 \              # 질문 개수
    --start-idx 0 \                    # 시작 인덱스
    --split train \                    # train/validation/test
    --num-rollouts 8 \                 # MC 샘플링 횟수
    --corpus-limit 100000 \            # corpus 크기 제한 (테스트용)
    --fusion-method rrf \              # rrf or weighted
    --k-sparse 50 \                    # BM25 top-K
    --k-dense 50 \                     # BGE top-K
    --output-dir outputs/hybrid_100q
```

---

## 🔧 Fusion 방법

### 1. RRF (Reciprocal Rank Fusion) - 추천!

**공식**: `score(d) = Σ [1 / (k + rank(d))]`

**장점**:
- Score normalization 불필요
- 간단하고 robust
- 일반적으로 weighted보다 나음

**설정**:
```python
HybridRetriever(
    bm25_retriever=bm25,
    bge_retriever=bge,
    fusion_method="rrf",    # ← RRF 사용
    k_sparse=50,
    k_dense=50,
    rrf_k=60,              # RRF constant (기본값 60)
)
```

### 2. Weighted Sum

**공식**: `score(d) = α × norm(BM25) + (1-α) × norm(BGE)`

**장점**:
- Weight 조절 가능
- 한쪽에 더 비중을 두고 싶을 때

**설정**:
```python
HybridRetriever(
    bm25_retriever=bm25,
    bge_retriever=bge,
    fusion_method="weighted",
    k_sparse=50,
    k_dense=50,
    alpha=0.5,             # BM25 weight (0.5 = 50:50)
)
```

**Alpha 값 가이드**:
- `alpha=0.7`: BM25 70%, BGE 30% (키워드 중요)
- `alpha=0.5`: 균등 (기본값)
- `alpha=0.3`: BM25 30%, BGE 70% (의미 중요)

---

## 📊 예상 성능 향상

### 이전 (Dense-only BGE)

```
전체 정확도: 59% (59/100)
RAG 사용: 50% (40/80)
RAG 미사용: 95% (19/20)

검색 품질:
- 평균 RPE: 2.652
- 중앙값 RPE: 0.000
- RPE < 0.5: 82%
```

### 예상 (Hybrid BM25 + BGE)

```
전체 정확도: 70-75% (추정)
RAG 사용: 65-70% (추정)

검색 품질:
- 평균 RPE: 8-12 (3-4배 향상)
- 중앙값 RPE: 1.5-2.5
- RPE < 0.5: 30-40% (절반 감소)
```

**개선 예상**:
- 검색 정확도: +20~40%
- 전체 정확도: +10~15%
- Citation 사용률: +15~25%

---

## 🔍 하이브리드가 해결하는 문제들

### 문제 1: 숫자/고유명사 매칭 실패

**이전 (Dense-only)**:
```
Query: "seventh largest country"
Results:
  [1] "7th Infantry Regiment" ❌
  [2] "Grand Est" ❌
  [3] "Bing search engine" ❌
```

**하이브리드**:
```
Query: "seventh largest country"
BM25 results: "India" (7번째로 큰 나라) ✅
BGE results: 의미적 유사 문서들
Fused: "India" (정확한 키워드 매칭!) ✅
```

---

### 문제 2: 같은 문서 반복

**이전 (Dense-only)**:
```
Query: "Matt Groening name Milhouse"
Results:
  [1] "Matt Groening" (조각 1) ❌
  [2] "Matt Groening" (조각 2) ❌
  [3] "Matt Groening" (조각 3) ❌
→ 모두 같은 문서의 100-word 조각들
```

**하이브리드**:
```
Query: "Matt Groening name Milhouse"
BM25 results: 키워드 매칭으로 다양한 문서
BGE results: 의미적으로 관련된 문서
Fused: 다양한 소스에서 정보 수집 ✅
```

---

### 문제 3: 추상적 쿼리 실패

**이전 (Dense-only)**:
```
Query: "chemical in which Cadmium Chloride is slightly soluble"
Results:
  [1] "Nitrite" ❌ (화학 물질이라는 것만 매칭)
  [2] "Niter" ❌
  [3] "Trivial name" ❌
```

**하이브리드**:
```
Query: "Cadmium Chloride solubility alcohol"  ← 키워드 개선도 필요!
BM25 results: "Cadmium Chloride" + "alcohol" 정확 매칭
BGE results: 용해도 관련 문서들
Fused: 정답 문서 찾기 ✅
```

---

## 📈 모니터링 및 분석

### 테스트 실행 후 확인할 지표

```bash
# 1. 전체 요약
cat outputs/hybrid_test/summary_hybrid_*.json

# 2. RPE 분포
python3 << 'EOF'
import json
with open('outputs/hybrid_test/results_hybrid_*.jsonl', 'r') as f:
    rpes = []
    for line in f:
        r = json.loads(line)
        for step in r['steps']:
            if step['action'] == 'Search':
                rpes.append(step['rpe'])

    print(f"평균 RPE: {sum(rpes)/len(rpes):.3f}")
    print(f"중앙값 RPE: {sorted(rpes)[len(rpes)//2]:.3f}")
    print(f"RPE < 0.5: {sum(1 for r in rpes if r < 0.5)}/{len(rpes)}")
EOF

# 3. Citation 분석
grep -o '"rag_with_citations": [0-9]*' outputs/hybrid_test/summary_*.json
```

### 비교 분석

```python
# Dense vs Hybrid 비교
import json

# Dense 결과
with open('outputs/batch_test_100q_final/summary_*.json', 'r') as f:
    dense_stats = json.load(f)

# Hybrid 결과
with open('outputs/hybrid_test/summary_hybrid_*.json', 'r') as f:
    hybrid_stats = json.load(f)

print("검색 품질 비교:")
print(f"Dense 정확도: {dense_stats['with_rag_correct']}/{dense_stats['with_rag']}")
print(f"Hybrid 정확도: {hybrid_stats['with_rag_correct']}/{hybrid_stats['with_rag']}")
```

---

## 🎯 다음 단계 최적화

### 1. Top-K 튜닝

```bash
# 실험 1: k_sparse, k_dense 변경
python3 scripts/batch_test_hybrid.py \
    --k-sparse 100 --k-dense 50  # BM25 많이, BGE 적게

python3 scripts/batch_test_hybrid.py \
    --k-sparse 50 --k-dense 100  # BGE 많이, BM25 적게
```

### 2. Alpha 튜닝 (Weighted)

```bash
# BM25 중심
python3 scripts/batch_test_hybrid.py \
    --fusion-method weighted \
    --alpha 0.7

# BGE 중심
python3 scripts/batch_test_hybrid.py \
    --fusion-method weighted \
    --alpha 0.3
```

### 3. 검색 쿼리 개선

현재 문제:
```python
# 나쁜 쿼리 (추상적)
"chemical in which Cadmium Chloride is slightly soluble"

# 좋은 쿼리 (키워드 중심)
"Cadmium Chloride solubility alcohol"
```

개선 방안:
- Prompt에서 키워드 중심 쿼리 생성 강제
- 불필요한 단어 제거
- 고유명사 강조

---

## 🔧 트러블슈팅

### 문제 1: BM25 인덱스 빌드 실패

```bash
# 메모리 부족 시
python3 scripts/precompute_bm25_index.py \
    --max-docs 100000  # 작은 크기로 시작

# 또는 swap 메모리 증가
sudo fallocate -l 16G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 문제 2: GPU 메모리 부족

```
ValueError: Free memory on device (74.56/95.0 GiB) is less than desired (85.5 GiB)
```

**해결책**:
```bash
# vLLM 프로세스 종료
pkill -f vllm

# 또는 GPU utilization 낮추기
# configs/adaptive_generation.yaml에서
gpu_memory_utilization: 0.7  # 기본값 0.9에서 낮춤
```

### 문제 3: KILT corpus 로드 느림

```python
# corpus_limit 사용
python3 scripts/batch_test_hybrid.py \
    --corpus-limit 100000  # 10만 개만 로드

# 또는 캐시된 corpus 사용 (추후 구현)
```

---

## 📝 체크리스트

테스트 실행 전 확인:

- [ ] KILT corpus 다운로드 완료 (35GB)
- [ ] BGE embeddings 생성 완료 (12GB)
- [ ] BM25 인덱스 빌드 완료 (.pkl)
- [ ] GPU 메모리 여유 확인 (>20GB)
- [ ] 시스템 RAM 여유 확인 (>16GB)

테스트 실행:

- [ ] 작은 테스트 (10K corpus, 5 questions)
- [ ] 중간 테스트 (100K corpus, 20 questions)
- [ ] 전체 테스트 (5.9M corpus, 100 questions)

분석:

- [ ] RPE 분포 확인
- [ ] Citation 사용률 확인
- [ ] MC 개선 정도 확인
- [ ] Dense-only와 비교

---

## 🎉 결론

**Hybrid Retrieval = BM25 + BGE-M3**

✅ **구현 완료**:
- BM25Retriever
- BGERetriever
- HybridRetriever (RRF/Weighted fusion)
- batch_test_hybrid.py

✅ **준비 완료**:
- KILT corpus (35GB)
- BGE embeddings (진행 중)

⏳ **다음 단계**:
1. BM25 인덱스 빌드
2. 작은 테스트 실행
3. 결과 분석 및 튜닝
4. 전체 테스트 (100 questions)

**예상 효과**: 검색 품질 2-4배 향상, 전체 정확도 +10~15%
