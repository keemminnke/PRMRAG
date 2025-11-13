# Wikipedia Retriever

## 개요

Wikipedia API 기반 실시간 검색 retriever입니다. HotpotQA가 Wikipedia 기반이므로 완벽하게 호환됩니다.

## 장점

✅ **무료** - API key 불필요
✅ **실시간** - 최신 Wikipedia 데이터
✅ **간편** - Corpus 파일 불필요
✅ **정확** - Wikipedia의 고품질 데이터

## 설치

```bash
pip install wikipedia-api
```

또는:

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용

```python
from prmrag.retrieval import WikipediaRetriever

# 초기화
retriever = WikipediaRetriever(
    lang='en',                  # 언어 (en, ko, etc.)
    user_agent='PRMRAG/1.0',    # User agent
    extract_sentences=5,        # 페이지당 추출할 문장 수
)

# 검색
results = retriever.retrieve("capital of France", top_k=5)

# 결과
for doc in results:
    print(f"Title: {doc['title']}")
    print(f"Text: {doc['text']}")
    print(f"URL: {doc['url']}")
    print(f"Score: {doc['score']}")
```

### Adaptive Trajectory Generation에서 사용

#### 방법 1: Config 파일로 지정

`configs/adaptive_generation.yaml`:
```yaml
retrieval:
  method: "wikipedia"  # bm25 → wikipedia로 변경

  wikipedia:
    lang: "en"
    user_agent: "PRMRAG/1.0"
    extract_sentences: 5
```

실행:
```bash
python scripts/generate_adaptive_trajectories.py \
    --config configs/adaptive_generation.yaml \
    --questions data/raw/hotpotqa_questions.jsonl \
    --output data/generated/adaptive_trajectories.jsonl \
    --num-trajectories 3
```

#### 방법 2: 명령줄 인자로 지정

```bash
python scripts/generate_adaptive_trajectories.py \
    --retrieval-method wikipedia \
    --questions data/raw/hotpotqa_questions.jsonl \
    --output data/generated/adaptive_trajectories.jsonl
```

**주의**: Wikipedia 사용 시 `--corpus` 인자는 필요 없습니다!

### BM25 vs Wikipedia 비교

| 특징 | BM25 (기존) | Wikipedia (새로 추가) |
|------|-------------|----------------------|
| **속도** | ⚡ 매우 빠름 (로컬) | 🌐 네트워크 의존 (약간 느림) |
| **데이터** | 📦 사전 준비 필요 | 🔄 실시간 최신 데이터 |
| **오프라인** | ✅ 가능 | ❌ 인터넷 필요 |
| **설정** | Corpus 파일 로드 | API 호출만 |
| **비용** | 무료 | 무료 |
| **품질** | Corpus 품질에 의존 | Wikipedia 공식 데이터 |

### 전환 방법

#### BM25에서 Wikipedia로

**Before:**
```bash
python scripts/generate_adaptive_trajectories.py \
    --corpus data/raw/hotpotqa_corpus.jsonl \  # ← 필요
    --questions data/raw/questions.jsonl \
    --output output.jsonl
```

**After:**
```bash
python scripts/generate_adaptive_trajectories.py \
    --retrieval-method wikipedia \  # ← 추가
    --questions data/raw/questions.jsonl \
    --output output.jsonl
    # --corpus 불필요!
```

또는 config 수정:
```yaml
# configs/adaptive_generation.yaml
retrieval:
  method: "wikipedia"  # "bm25" → "wikipedia"
```

## 테스트

Wikipedia retriever 테스트:

```bash
python scripts/test_wikipedia_retriever.py
```

출력 예시:
```
Query: Paris France capital
------------------------------------------------------------

  [1] Paris
      Score: 1.00
      Text: Paris is the capital and most populous city of France...
      URL: https://en.wikipedia.org/wiki/Paris

  [2] France
      Score: 0.90
      Text: France, officially the French Republic...
      URL: https://en.wikipedia.org/wiki/France
```

## 고급 사용

### 특정 페이지 가져오기

```python
doc = retriever.retrieve_by_title("Paris")

print(doc['title'])  # "Paris"
print(doc['text'])   # Wikipedia summary
print(doc['url'])    # https://en.wikipedia.org/wiki/Paris
```

### 배치 검색

```python
queries = [
    "capital of France",
    "author of Harry Potter",
    "largest planet",
]

results_batch = retriever.batch_retrieve(queries, top_k=3)

for query, results in zip(queries, results_batch):
    print(f"Query: {query}")
    for doc in results:
        print(f"  - {doc['title']}")
```

### 다른 언어 사용

```python
# 한국어 Wikipedia
retriever_ko = WikipediaRetriever(lang='ko')
results = retriever_ko.retrieve("서울 대한민국", top_k=5)

# 일본어 Wikipedia
retriever_ja = WikipediaRetriever(lang='ja')
results = retriever_ja.retrieve("東京", top_k=5)
```

## 코드에서 직접 통합

```python
from prmrag.generation import AdaptiveTrajectoryGenerator
from prmrag.retrieval import WikipediaRetriever

# Wikipedia retriever 생성
retriever = WikipediaRetriever(lang='en')

# Generator에 전달
generator = AdaptiveTrajectoryGenerator(
    policy_model=model,
    retriever=retriever,  # ← Wikipedia retriever
    config=config
)

# 나머지는 동일
trajectories = generator.generate_batch(questions)
```

## 제한사항 및 해결책

### 1. Rate Limiting

Wikipedia API에는 rate limit이 있습니다.

**해결책:**
```python
import time

# 요청 사이에 짧은 딜레이
for query in queries:
    results = retriever.retrieve(query)
    time.sleep(0.1)  # 100ms 딜레이
```

### 2. 네트워크 오류

인터넷 연결이 불안정할 수 있습니다.

**해결책:** 자동 재시도 (이미 구현됨)
```python
# wikipedia_retriever.py 내부에 error handling 있음
try:
    response = requests.get(url, timeout=10)
except Exception as e:
    print(f"Error: {e}")
    return []
```

### 3. 검색 결과 없음

일부 쿼리는 결과가 없을 수 있습니다.

**해결책:** 쿼리 재구성 또는 fallback
```python
results = retriever.retrieve(query)

if not results:
    # 쿼리 단순화
    simple_query = query.split()[0]
    results = retriever.retrieve(simple_query)
```

## Factory Pattern으로 생성

```python
from prmrag.retrieval.wikipedia_retriever import create_retriever

# Wikipedia
retriever = create_retriever('wikipedia', lang='en')

# BM25
retriever = create_retriever('bm25', corpus=corpus)
```

## 성능 최적화

### Caching

자주 검색되는 쿼리는 캐시:

```python
from functools import lru_cache

class CachedWikipediaRetriever(WikipediaRetriever):
    @lru_cache(maxsize=1000)
    def retrieve(self, query: str, top_k: int = 5):
        return super().retrieve(query, top_k)
```

### Parallel Retrieval

여러 쿼리 병렬 처리:

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_retrieve(retriever, queries, top_k=5):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(
            lambda q: retriever.retrieve(q, top_k),
            queries
        ))
    return results
```

## 문제 해결

### 설치 오류

```bash
pip install --upgrade wikipedia-api
```

### Import 오류

```python
# Wrong
from prmrag.retrieval import wikipedia_retriever

# Correct
from prmrag.retrieval import WikipediaRetriever
```

### 검색이 느림

- 네트워크 속도 확인
- `top_k` 줄이기
- `extract_sentences` 줄이기

### 결과 품질이 낮음

- 쿼리 개선 (더 구체적으로)
- `top_k` 늘리기
- 다른 언어 시도

## 참고

- Wikipedia API 문서: https://wikipedia-api.readthedocs.io/
- MediaWiki API: https://www.mediawiki.org/wiki/API:Main_page
- HotpotQA: https://hotpotqa.github.io/
