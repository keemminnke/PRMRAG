# Step-level DPO 데이터 구축 방법론

## 배경: 왜 Trajectory-level DPO가 실패했는가

### 기존 방식 (Trajectory-level DPO)
```
Prompt: system_prompt + question
Chosen:  전체 trajectory (step 1 ~ step N)  ← 정답 맞춘 trajectory
Rejected: 전체 trajectory (step 1 ~ step N) ← 틀린 trajectory
```

**문제점:**
1. **신호 희석**: 5~8 step 전체를 비교하므로 "어디가 좋고 나쁜지" 학습 신호가 약함
2. **Prefix 불일치**: chosen/rejected trajectory의 step 1부터 이미 다르므로, 모델이 "같은 상황에서의 더 나은 선택"을 학습하기 어려움
3. **실험 결과**: 6번의 실험 (Exp 0~6)에서 DPO가 SFT 대비 개선 효과 없음
   - Loss: 0.693 → 0.42 (잘 내려감) → 하지만 eval 성능은 동일
   - Rewards accuracy 82.8% → 과적합이지 일반화가 아님

### ReasonRAG 논문의 Step-level DPO
```
L(θ) = -E log σ(β · log p(y_t^w | x, y_{<t}) / p(y_t^l | x, y_{<t}))
```
- `y_{<t}`: 동일한 prefix (같은 context)
- `y_t^w`: 더 좋은 다음 step (chosen)
- `y_t^l`: 더 나쁜 다음 step (rejected)
- **핵심**: 동일한 상황에서 1 step의 차이만 비교 → 명확한 학습 신호

---

## Step-level DPO 데이터 구축

### Source 1: 기존 데이터에서 추출 (추가 비용 없음)

#### 원리
같은 question의 trajectory들은 특정 step까지 **동일한 prefix를 공유**할 수 있다.

- **Step 1**: 모든 trajectory가 동일한 prefix 공유 (system_prompt + question)
- **Step 2**: Step 1에서 같은 query를 생성한 trajectory끼리 prefix 공유
- **Step 3**: Step 1, 2 모두 같은 query인 trajectory끼리 prefix 공유
- ...

#### 기준: Critic Step Label (GOOD/BAD)
정답 여부(`is_correct`)가 아닌 **critic model의 step-level label**을 기준으로 chosen/rejected를 결정한다.
- **GOOD (1)**: critic이 해당 step의 reasoning이 적절하다고 판단 → chosen
- **BAD (0)**: critic이 해당 step의 reasoning이 부적절하다고 판단 → rejected

이유: 최종 정답을 맞춘 trajectory도 중간에 BAD step이 있을 수 있고, 틀린 trajectory도 GOOD step이 있을 수 있다. Step-level DPO에서는 각 step의 질을 개별적으로 판단해야 한다.

#### 예시 (Step 1)
```
Question: "Who directed Parasite?"

Trajectory A step 1 (GOOD): <think>영화 기생충의 감독을 찾아야...</think><search>Parasite film director</search>
Trajectory B step 1 (BAD):  <think>기생충이라는 영화가 있는데...</think><search>Parasite movie cast</search>

Step-level DPO pair:
  Prompt:   system_prompt + "Question: Who directed Parasite?"
  Chosen:   step 1 from A (critic: GOOD)
  Rejected: step 1 from B (critic: BAD)
```
→ 적절한 검색 쿼리 (감독 직접 검색) vs 부적절한 쿼리 (캐스트 검색)

#### 예시 (Step 2 — 동일 query1 공유)
```
Trajectory A, B 모두: step 1 query = "Parasite film director" (동일)
                      → 같은 documents 수신

Trajectory A step 2 (GOOD): <think>봉준호라는 정보를 찾았다</think><answer>봉준호</answer>
Trajectory B step 2 (BAD):  <think>더 찾아봐야 할 것 같다</think><search>Parasite 2019 awards</search>

Step-level DPO pair:
  Prompt:   system_prompt + question + [step 1 + documents] (동일 prefix)
  Chosen:   step 2 from A (critic: GOOD)
  Rejected: step 2 from B (critic: BAD)
```
→ 충분한 정보가 있을 때 "바로 답할지 vs 불필요한 검색을 더 할지" 학습

#### 추출 조건
- 같은 question 내에서 **동일 prefix를 공유하는** trajectory 비교
- 분기 step의 critic label이 **GOOD vs BAD**인 경우만 pair 생성
- 분기 step의 텍스트가 동일한 경우 제외 (학습 신호 없음)
- Dedup: 같은 (chosen_text, rejected_text) 쌍 중복 제거

#### 결과
| Step | Pairs | 비고 |
|------|-------|------|
| Step 1 | 86,249 | prefix = question (모든 trajectory 공유) |
| Step 2 | 20,585 | 동일 query1 공유 |
| Step 3 | 5,162 | 동일 query1,2 공유 |
| Step 4 | 151 | |
| Step 5 | 48 | |
| **합계** | **112,195** | |

---

### Source 2: Regeneration (BAD step 지점에서 재생성)

#### 원리
Critic model이 BAD로 판정한 step 지점에서, 동일 prefix를 고정하고 대안 step을 K번 재생성한다.

```
원본 trajectory:
  step 1 (GOOD) → step 2 (GOOD) → step 3 (BAD) → ...
                                    ↑ 여기서 잘라서 재생성

Regen (K=3):
  같은 prefix (step 1~2) 에서:
    → step 3' (attempt 1) → critic scoring → GOOD ✓ → chosen
    → step 3' (attempt 2) → critic scoring → BAD  ✗ → 버림
    → step 3' (attempt 3) → critic scoring → GOOD ✓ → chosen

DPO pair:
  Prompt:   system_prompt + question + [step 1 + docs + step 2 + docs]
  Chosen:   step 3' (regenerated, GOOD)
  Rejected: step 3  (original, BAD)
```

#### 이점
- **1,473개 question** (모든 trajectory가 오답)에서도 pair 생성 가능
- 기존 데이터만으로는 correct trajectory가 없어서 pair을 못 만들던 question 커버
- Prefix가 물리적으로 완전 동일 (같은 텍스트에서 분기)

#### Regen 대상 분포
| BAD step 위치 | Unique prefixes |
|------|------|
| Step 2 | 11,165 |
| Step 3 | 10,678 |
| Step 4 | 3,571 |
| Step 5+ | 1,543 |
| **합계** | **26,957** |

#### 파이프라인
1. Policy model (vLLM, `Qwen/Qwen2.5-7B-Instruct`)로 step 재생성
   - `SamplingParams(temperature=0.7, n=K)` — 한 prompt에서 K개 동시 생성
   - `stop=["<search>", "<answer>"]` — 1 step만 생성 후 중단
2. Critic model (vLLM + LoRA, `critic_v9`)로 재생성된 step 평가
3. GOOD 판정 → chosen, 원본 BAD → rejected 으로 pair 구성

---

## 최종 데이터 형식

```json
{
  "prompt": [
    {"role": "system", "content": "You are an advanced AI agent..."},
    {"role": "user", "content": "Question: ..."},
    {"role": "assistant", "content": "<think>...</think>\n<search>...</search>\n<documents>...</documents>"}
  ],
  "chosen": [
    {"role": "assistant", "content": "<think>...</think>\n<search>good query</search>"}
  ],
  "rejected": [
    {"role": "assistant", "content": "<think>...</think>\n<search>bad query</search>"}
  ],
  "question_id": "...",
  "step_level": 2,
  "source": "existing"
}
```

- `prompt`: system + user + (prefix steps as assistant)
- `chosen` / `rejected`: 분기하는 1 step만 (messages format)
- `step_level`: 몇 번째 step에서 분기하는지
- `source`: `"existing"` (Source 1) 또는 `"regen"` (Source 2)

---

## Trajectory-level DPO와 비교

| | Trajectory-level DPO | Step-level DPO |
|---|---|---|
| 비교 단위 | 전체 trajectory (5~8 steps) | 1 step |
| Prefix 공유 | ✗ (각 trajectory 독립) | ✓ (동일 prefix 보장) |
| 학습 신호 | 희석됨 | 명확함 |
| Pairs 수 | 27,323 | ~130,000+ |
| Question 커버 | 3,541 / 5,000 | 2,407 + regen으로 확장 |
| 근거 | Best-vs-All heuristic | ReasonRAG (EMNLP 2025) |

---

## 스크립트

```bash
# Source 1 only (기존 데이터만)
python scripts/build_step_level_dpo.py \
    --input outputs/all_5000q_critic_v9_filtered.jsonl \
    --output outputs/step_dpo_5000q_source1.jsonl

# Source 1 + Source 2 (regen 포함)
python scripts/build_step_level_dpo.py \
    --input outputs/all_5000q_critic_v9_filtered.jsonl \
    --output outputs/step_dpo_5000q_full.jsonl \
    --regen --regen-k 3 \
    --policy-model Qwen/Qwen2.5-7B-Instruct \
    --critic-model outputs/critic_model_v9_3000q/final_model \
    --gpu-memory 0.88
```
