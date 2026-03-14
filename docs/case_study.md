# Case Study: Process Reward vs Outcome Reward

## Case 1: Entity Confusion (2WikiMultiHopQA)

**Question**: Where was the place of death of the director of film Curse Of The Crimson Altar?
**Gold Answer**: Durban, South Africa

### Search-R1 (Outcome-based) — ❌ Pietermaritzburg (2 steps)

```
Step 1: <search>Vernon Sewell place of death</search>
→ Retrieved: George Sewell (politician), Cecil Sewell, John Sewell (cricketer)
  ※ Vernon Sewell 문서 검색 실패, 다른 Sewell들만 반환

Step 2: (추론 후 즉시 답변)
"The information about Vernon Sewell's place of death is not directly available.
 However, I found information about another Vernon Sewell who died in
 Pietermaritzburg. Since the director of "Curse Of The Crimson Altar" is
 Vernon Sewell and there is no specific information about his place of death,
 it is reasonable to assume that the place of death mentioned is the same person."
→ <answer>Pietermaritzburg</answer>
```

**실패 원인**: John Sewell (cricketer, 1844–1897)의 사망지 Pietermaritzburg를 Vernon Sewell (director, 1903–2001)의 것으로 혼동. 정보가 부족한 상황에서 **잘못된 entity의 정보로 성급하게 답변**.

### ReasonRAG (Outcome-based, 4 prompts) — ❌ Pietermaritzburg

Search-R1과 동일하게 entity 혼동 발생. 4개 프롬프트를 사용했음에도 동일한 오류.

### V1 (Ours, PRM-guided) — ✅ Durban, South Africa (9 steps)

```
Step 1: <search>director of film Curse Of The Crimson Altar</search>
→ "Vernon Sewell" 확인 ✓

Step 2: <search>Vernon Sewell place of death</search>
→ 관련 문서 없음. John Sewell (cricketer) 문서가 나왔지만 혼동하지 않음:
  "The documents do not provide information about Vernon Sewell's place of death.
   However, the document about John Sewell (cricketer) mentions a place of death,
   which is not the director we are looking for."

Step 3–7: 쿼리를 점진적으로 변경하며 재검색
→ "Vernon Sewell date of death and place of death"
→ "Vernon Sewell biography and place of death"
→ "Vernon Sewell place of death detailed biography"
→ "Vernon Sewell place of death 1968 detailed biography"
→ 여전히 Vernon Sewell 문서 미발견, 하지만 포기하지 않음

Step 8: <search>Vernon Sewell place of death ... film Curse of the Crimson Altar</search>
→ Vernon Sewell 문서 발견!
  "Vernon Campbell Sewell (4 July 1903 – 21 June 2001) was a British film director...
   Vernon Sewell died on 21 June 2001 in Durban, South Africa, at age 97."

Step 9: 최종 답변
  "From the documents, we can see that Vernon Sewell, the director of
   'Curse of the Crimson Altar,' died on 21 June 2001 in Durban, South Africa."
→ <answer>Durban, South Africa</answer>
```

**성공 요인**: PRM이 step마다 추론 quality를 평가하므로, (1) 다른 entity 혼동 없이 정확한 판단, (2) 정보 부족 시 포기하지 않고 쿼리를 변경하며 끈기 있게 검색하는 패턴이 강화됨.

---

## Case 2: Premature Termination at Intermediate Entity (MuSiQue)

**Question**: When is Celebrity Big Brother coming to the network which, along with ABC and the original broadcaster of Here and Now, is the other major broadcaster based in NY?
**Gold Answer**: February 7, 2018
**Required Reasoning**: Here and Now → NBC → (ABC, NBC, CBS are major broadcasters in NY) → CBS → Celebrity Big Brother on CBS → February 7, 2018

### Search-R1 (Outcome-based) — ❌ CBS (2 steps)

```
Step 1: <search>history of Celebrity Big Brother and its broadcasting networks</search>
→ Doc 1(Title: Celebrity Big Brother (American TV series))
  "Celebrity Big Brother... The series premiered on CBS in the United States
   on February 7, 2018 and is simulcast in Canada on Global."
→ Doc 2(Title: Celebrity Big Brother (British TV series))
  British version, Channel 4/Channel 5 — 관련 없음
→ Doc 3(Title: Celebrity Big Brother (British series 1))
  British first series — 관련 없음

추론: "Based on the information provided, Celebrity Big Brother is broadcasted on CBS
in the United States. Now, I need to find the original broadcaster of Here and Now
and if it's based in NY."

Step 2: <search>original broadcaster of Here and Now and its location</search>
→ Doc 1(Title: Here and Now (Boston)) — NPR/WBUR 라디오 프로그램
→ Doc 2(Title: Here and Now (Toronto)) — 캐나다 CBC 라디오
→ Doc 3(Title: Here and Now (1992 TV series))
  "Here and Now is an American television sitcom that aired on NBC
   from September 19, 1992 to January 2, 1993."

추론: "The original broadcaster of Here and Now is WBUR in Boston,
which is based in NY. Now I can answer the question."
→ <answer>CBS</answer>
```

**실패 원인**: 질문이 "When"을 묻고 있는데, 중간 추론 결과인 네트워크 이름(CBS)을 최종 답변으로 반환. 1번째 검색에서 이미 "premiered on CBS... on February 7, 2018" 문서를 찾았음에도, 날짜가 아닌 네트워크명을 답변. Multi-hop 추론의 **마지막 hop을 수행하지 않고 중간 단계에서 조기 종료**.

### ReasonRAG (Outcome-based, 4 prompts) — ❌ 2023

```
[BEGIN_REASONING] 질문 분해:
  Sub-Q1: "Here and Now"의 원래 방송사 → NBC (실제로는 NPR로 잘못 판단)
  Sub-Q2: ABC, NBC와 함께 NY 기반 major broadcaster → NBC로 식별 (정답은 CBS)
  → "When is Celebrity Big Brother coming to NBC?" 쿼리 생성

[DOCUMENT_ANALYSIS #1]
  Retrieved: Celebrity Big Brother 2 (American season) — "premiered on CBS on January 21, 2019"
  Evidence: "Celebrity Big Brother... premiered on CBS on January 21, 2019"
  ※ CBS에서 방영된다는 증거를 찾았지만, NBC를 찾고 있어서 무시

[REASONING #1 → QUERY #2]
  동일 쿼리 반복: "When is Celebrity Big Brother coming to NBC?"
  → Retrieved: 동일 문서, Evidence: None

[REASONING #2 → QUERY #3]
  동일 쿼리 반복: "When is Celebrity Big Brother coming to NBC?"
  → Retrieved: 동일 문서, Evidence: None

[REASONING #3 → QUERY #4]
  동일 쿼리 반복: "When is Celebrity Big Brother coming to NBC?"
  → Retrieved: Celebrity Big Brother 2 — "premiered on CBS on January 21, 2019"

[ANSWER_GENERATION]
  4번 검색했지만 NBC에서 Celebrity Big Brother를 찾지 못함
  → 환각으로 <answer>2023</answer> 출력
```

**실패 원인**: Sub-Q1에서 "Here and Now" 원래 방송사를 NPR로 잘못 판단 → 정답 네트워크(CBS) 대신 NBC를 탐색 → 4회 동일 쿼리 반복 후 결국 환각 답변. **첫 번째 hop의 오류가 전체 reasoning chain을 오염시키는 전형적 error propagation**.

### V1 (Ours, PRM-guided) — ✅ February 7, 2018 (5 steps)

```
Step 1: <search>When did Here and Now start and which network originally broadcast it?</search>
→ Doc 1(Title: Here and Now (Boston)) — NPR/WBUR 라디오 프로그램
→ Doc 2(Title: Here and Now (1992 TV series))
  "Here and Now is an American television sitcom that aired on NBC
   from September 19, 1992 to January 2, 1993."
→ Doc 3(Title: Today (American TV program)) — NBC 관련 정보

추론: "'Here and Now' is a radio program, not a TV show... However, the question
mentions a TV show, so we need to look for the network that co-broadcasts with ABC
and the original broadcaster of a TV show called 'Here and Now'."
※ 라디오 vs TV를 구별하고, TV 시리즈(NBC)를 올바르게 식별

Step 2: <search>Which network co-broadcasts with ABC and the original broadcaster of Here and Now TV show?</search>
→ Doc 2(Title: Here and Now (1992 TV series)) — NBC 확인
→ "Here and Now" = NBC → ABC, NBC, 그리고 나머지 major broadcaster = CBS

추론: "there is no TV show called 'Here and Now' that co-broadcasts with ABC...
Let's try to find out when Celebrity Big Brother is coming to a network based in NY."

Step 3: <search>When is Celebrity Big Brother coming to a network based in New York?</search>
→ Doc 1(Title: Celebrity Big Brother (American TV series))
  "The series premiered on CBS in the United States on February 7, 2018"
→ Doc 2(Title: Celebrity Big Brother 2) — Season 2: CBS, January 21, 2019
→ Doc 3(Title: Celebrity Big Brother: After Dark) — Pop channel

추론: "Celebrity Big Brother is a CBS show that premiered on February 7, 2018...
the question asks about a network based in New York."

Step 4: <search>Which network co-broadcasts with ABC and is based in New York?</search>
→ Doc 1(Title: WABC-TV) — ABC's flagship, New York
→ Doc 2(Title: American Broadcasting Company)
  "The American Broadcasting Company (ABC)... headquartered in Burbank, California...
   second corporate headquarters... in New York City"
→ ABC, NBC, CBS 모두 NY 기반 확인

추론: "WABC-TV, which is based in New York and co-broadcasts with ABC...
Now, we need to find out when Celebrity Big Brother is coming to WABC-TV."

Step 5: <search>When is Celebrity Big Brother coming to WABC-TV?</search>
→ Doc 1(Title: Celebrity Big Brother 2) — "premiered on CBS on January 21, 2019"
→ Doc 2(Title: Celebrity Big Brother (American TV series))
  "The series premiered on CBS in the United States on February 7, 2018"

추론: "Celebrity Big Brother has already aired on CBS... The first season premiered
on February 7, 2018... the answer is that it has already come to the network."
→ <answer>February 7, 2018</answer>
```

**성공 요인**: PRM이 각 step의 추론 완성도를 평가하므로:
1. **라디오 vs TV 구별** — 같은 이름의 프로그램에서 TV 버전(NBC)을 올바르게 식별
2. **중간 entity에서 멈추지 않음** — CBS를 식별한 후에도 질문이 묻는 "When"까지 추론 완료
3. **질문 유형 인식** — "When" 질문에 네트워크명이 아닌 날짜를 답변

---

## Analysis Summary

| Failure Pattern | Search-R1 | ReasonRAG | V1 (Ours) |
|---|---|---|---|
| **Entity Confusion**: 검색 결과에서 다른 entity의 정보를 혼동 | ✗ 발생 | ✗ 발생 | ✓ 회피 |
| **Premature Termination**: 중간 추론 결과를 최종 답변으로 출력 | ✗ 발생 | ✗ 발생 | ✓ 회피 |
| **Insufficient Search**: 정보 부족 시 조기 포기 | ✗ 2 steps | ✗ 발생 | ✓ 9 steps까지 지속 |

### Why Process Reward Matters

**Outcome-based models** (Search-R1, ReasonRAG)은 최종 정답 여부만으로 reward를 받기 때문에:
- 잘못된 추론으로 **우연히 정답을 맞추는 경우**와 올바른 추론을 구별하지 못함
- 결과적으로 entity 혼동, 조기 종료 같은 **잘못된 reasoning shortcut**이 학습될 수 있음

**Process Reward Model (Ours)**은 각 step의 추론 quality를 평가하므로:
- 잘못된 entity를 혼동하는 step → 낮은 reward → 억제
- 정보 부족 시 추가 검색하는 step → 높은 reward → 강화
- 중간 entity에서 멈추지 않고 끝까지 추론 → 높은 reward → 강화
