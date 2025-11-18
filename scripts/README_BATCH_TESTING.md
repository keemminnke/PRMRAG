# Batch Testing and RAG Analysis

이 가이드는 100개 이상의 질문으로 adaptive pipeline을 테스트하고, RAG가 실제로 reasoning을 개선하는지 분석하는 방법을 설명합니다.

## 📋 Workflow

### 1. 100개 질문 배치 테스트

```bash
# 기본 설정 (100개 질문, 시작 인덱스 0)
python3 scripts/batch_test_adaptive.py

# 커스텀 설정
python3 scripts/batch_test_adaptive.py \
    --num-questions 10 \
    --start-idx 0 \
    --num-rollouts 5 \
    --output-dir outputs/batch_test
```

**출력 파일:**
- `outputs/batch_test/results_TIMESTAMP.jsonl` - 상세 결과 (각 trajectory)
- `outputs/batch_test/summary_TIMESTAMP.json` - 요약 통계

**진행 상황:**
- 각 질문마다 진행 상황과 결과를 실시간으로 출력
- 완료 시 전체 통계 요약 표시

### 2. RAG 효과 분석

```bash
python3 scripts/analyze_rag_impact.py outputs/batch_test/results_TIMESTAMP.jsonl
```

**분석 내용:**
1. **기본 통계**: RAG 사용 빈도, 전체 정확도
2. **정확도 비교**: RAG 사용 시 vs CoT만 사용 시
3. **RAG 개입 분석**:
   - 'good' vs 'bad' label 비율
   - MC 개선 정도
4. **가장 효과적인 RAG 개입**: MC 개선이 가장 큰 경우들
5. **부정적 영향**: RAG가 오히려 MC를 낮춘 경우들
6. **흥미로운 케이스**:
   - RAG가 도움이 된 경우 (정답 맞춤)
   - RAG가 도움이 안 된 경우 (여전히 오답)
   - CoT만으로 성공한 경우

### 3. 개별 Trajectory 상세 확인

```bash
# RAG를 사용한 trajectory만 보기
python3 scripts/view_trajectory.py outputs/batch_test/results_TIMESTAMP.jsonl \
    --filter-rag \
    --show-passages \
    --limit 20

# 틀린 답만 보기 (RAG가 도움이 안 된 경우)
python3 scripts/view_trajectory.py outputs/batch_test/results_TIMESTAMP.jsonl \
    --filter-rag \
    --filter-incorrect \
    --show-passages

# 특정 질문 상세히 보기
python3 scripts/view_trajectory.py outputs/batch_test/results_TIMESTAMP.jsonl \
    --question-id 5a8b57f25542995d1e6f1371 \
    --show-passages
```

**출력 형식:**
```
TRAJECTORY: 5a8b57f25542995d1e6f1371
======================================================================

📝 Question:
   What government position was held by the woman who...

✅ Gold Answer:
   Chief of Protocol

🤖 Predicted Answer: ✅ CORRECT
   chief of protocol

📊 Summary:
   Total steps: 6
   - CoT steps: 4
   - RAG steps: 2

🔍 Step-by-Step Breakdown:
   ------------------------------------------------------------------

   Step 1 🔵 CoT [✓ good]
   MC: 0.200 → 0.400 (RPE: 2.000)
   │ First, I need to identify who played Corliss Archer...

   Step 2 🟢 RAG [✓ good]
   MC: 0.400 → 0.800 (RPE: 2.000)
   │ Based on the retrieved information, Shirley Temple...
   │
   │ Retrieved passages: 2
   │   1. Kiss and Tell (1945 film)
   │   2. Shirley Temple
   ...

💡 RAG Impact Summary:
   Step 2: MC ↑ +0.400 [✓ good]
      Passages: Kiss and Tell (1945 film), Shirley Temple
```

## 📊 결과 파일 형식

### results_TIMESTAMP.jsonl

각 줄은 하나의 trajectory를 나타냅니다:

```json
{
  "question_id": "5a8b57f25542995d1e6f1371",
  "question": "What government position was held by...",
  "gold_answer": "Chief of Protocol",
  "predicted_answer": "chief of protocol",
  "is_correct": true,
  "num_steps": 6,
  "num_cot_steps": 4,
  "num_rag_steps": 2,
  "has_rag": true,
  "steps": [
    {
      "step_num": 1,
      "type": "cot",
      "content": "...",
      "mc_before": 0.200,
      "mc_after": 0.400,
      "rpe": 2.000,
      "label": "good",
      "is_rag": false
    },
    {
      "step_num": 2,
      "type": "rag",
      "content": "...",
      "mc_before": 0.400,
      "mc_after": 0.800,
      "rpe": 2.000,
      "label": "good",
      "is_rag": true,
      "num_passages": 2,
      "passage_titles": ["Kiss and Tell (1945 film)", "Shirley Temple"]
    }
  ],
  "rag_interventions": [
    {
      "step_num": 2,
      "mc_improvement": 0.400,
      "rpe": 2.000,
      "label": "good",
      "passage_titles": ["Kiss and Tell (1945 film)", "Shirley Temple"]
    }
  ]
}
```

## 🔍 분석 가이드

### RAG가 정말 도움이 되는지 확인하는 방법:

1. **MC Improvement 확인**:
   - `rag_interventions[*].mc_improvement > 0` → RAG가 confidence 향상
   - 음수이면 RAG가 오히려 해를 끼침

2. **Step Content 읽기**:
   - RAG step 이전: CoT가 막혔거나 잘못된 방향으로 가고 있었나?
   - RAG step 이후: 올바른 정보를 찾아서 방향을 바꿨나?

3. **Retrieved Passages 확인**:
   - 관련성 있는 passage를 찾았나?
   - Passage가 실제로 답에 필요한 정보를 포함하나?

4. **Label 분석**:
   - 'good' label: RPE ≥ 0.8 (MC가 충분히 개선됨)
   - 'bad' label: RPE < 0.8 (MC가 거의 안 올라가거나 떨어짐)

### 주의할 점:

- **MC = 1.0이 항상 정답은 아님**: Rollout이 우연히 맞았을 수 있음
- **RAG가 사용되었다 = 어려운 문제**: CoT만으로 안 풀려서 RAG 개입
- **Label이 'bad'여도 유지**: 나쁜 step도 학습 데이터로 유용함

## 💡 예상되는 발견 사항

### RAG가 도움이 되는 경우:
- Multi-hop question (여러 단계의 추론 필요)
- 외부 지식이 필요한 경우
- CoT가 잘못된 가정을 했을 때 교정

### RAG가 도움이 안 되는 경우:
- 질문 자체가 모호함
- Corpus에 관련 정보가 없음
- 잘못된 passage를 retrieve했음
- 모델이 passage를 제대로 활용 못함

## 🚀 빠른 시작

```bash
# 1. 10개로 먼저 테스트 (빠르게)
python3 scripts/batch_test_adaptive.py --num-questions 10

# 2. 결과 분석
python3 scripts/analyze_rag_impact.py outputs/batch_test/results_*.jsonl

# 3. RAG 사용한 trajectory 확인
python3 scripts/view_trajectory.py outputs/batch_test/results_*.jsonl \
    --filter-rag --show-passages --limit 5

# 4. 만족스러우면 100개로 확장
python3 scripts/batch_test_adaptive.py --num-questions 100
```

## 📝 수동 검토 팁

결과를 보면서 다음을 체크하세요:

- [ ] RAG step에서 실제로 유용한 정보를 가져왔는가?
- [ ] RAG step 이후 reasoning이 개선되었는가?
- [ ] MC improvement와 실제 reasoning 품질이 일치하는가?
- [ ] 'good' label이 정당한가? (실제로 progress가 있었나?)
- [ ] 'bad' label인데도 최종 답은 맞는 경우가 있나? (false negative)

이렇게 하면 RAG가 정말로 reasoning을 돕는지, 아니면 단순히 정보만 추가하는지 알 수 있습니다!
