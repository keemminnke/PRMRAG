# 데이터 수정 계획

## 현재 상황

**문제**: 996개 중 29.4% (1,405 steps) 쓰레기
- 환각: 1,327개 (27.8%)
- 잘못 분류: 78개 (1.6%)

**결론**: 이 데이터로 Step 2 불가능!

## 해결 방안

### 옵션 1: 빠른 수정 (1-2시간)

**방법**: fix_data_issues.py 실행

```bash
python3 scripts/fix_data_issues.py
```

**수정 내용**:
1. 78개 잘못 분류 → action='Search'로 변경
2. 1,327개 환각 → Observation 제거, Thought만 남김

**결과**:
- 깨끗한 데이터: results_hybrid_all_996_FIXED.jsonl
- 하지만 content 필드 손상
- Step 2 가능하지만 품질 낮음

**장점**: 빠름 (1-2시간)
**단점**: 임시방편, content 정보 손실

---

### 옵션 2: 제대로 다시 생성 (권장!)

**방법**: 생성 코드 수정 후 996개 재생성

#### Step 1: 프롬프트 수정 (10분)

src/prmrag/models/policy_model_vllm.py 수정

기존 문제 부분 제거:
- "Do NOT stop after writing Action"
- "You MUST write Observation"

새로 추가:
- "Write ONLY Thought + Action"
- "STOP after Action: Search"
- "DO NOT write Observation yourself"

#### Step 2: Stop sequence 추가 (5분)

generate_with_chat_template에 stop_sequences 추가:
- "\nObservation:"
- "Observation:"

#### Step 3: num_passages 저장 확인 (5분)

생성 로직에서 모든 스텝에 num_passages 설정 확인

#### Step 4: 테스트 (30분)

```bash
# 10개로 테스트
python3 scripts/batch_test_hybrid.py \
    --num-questions 10 \
    --start-idx 0 \
    --num-rollouts 4 \
    --output-dir outputs/test_fixed
```

#### Step 5: 전체 재생성 (8-12시간)

```bash
# 996개 재생성
python3 scripts/batch_test_hybrid.py \
    --num-questions 996 \
    --start-idx 0 \
    --num-rollouts 4 \
    --output-dir outputs/hybrid_1000q_CLEAN
```

**장점**: 깨끗한 데이터, 환각 없음, 올바른 분류
**단점**: 시간 소요 (8-12시간)

---

## 권장: 옵션 2 (재생성)

**이유**:
1. 29.4% 쓰레기면 수정해도 품질 낮음
2. 프롬프트 수정 간단 (20분)
3. 재생성해도 8-12시간 (하룻밤)
4. 깨끗한 데이터로 PRM 학습

**타임라인**:
- 지금: 프롬프트 수정 (20분)
- 지금+30분: 10개 테스트
- 지금+1시간: 996개 생성 시작
- 내일 아침: 완료, 검증

---

## 즉시 실행 명령어

### 옵션 1 선택 시:
```bash
python3 scripts/fix_data_issues.py
python3 scripts/quick_stats.py outputs/hybrid_1000q_from_0/results_hybrid_all_996_FIXED.jsonl
```

### 옵션 2 선택 시:
```bash
# 1. 프롬프트 수정 (제가 도와드림)
# 2. 10개 테스트
python3 scripts/batch_test_hybrid.py --num-questions 10 --start-idx 0 --num-rollouts 4 --output-dir outputs/test_fixed

# 3. 검증 후 전체 생성
python3 scripts/batch_test_hybrid.py --num-questions 996 --start-idx 0 --num-rollouts 4 --output-dir outputs/hybrid_1000q_CLEAN
```

---

**결정해주세요**: 옵션 1 (빠른 수정) vs 옵션 2 (재생성)?
