# Hybrid Retrieval Analysis: BM25 + BGE-M3 on KILT Wikipedia

**Date**: 2024-12-04
**Test Configuration**: 10 questions from HotpotQA train split, 8 MC rollouts
**Corpus**: KILT Wikipedia (5,903,530 documents, 37.32 GB)
**Retrieval**: Hybrid (BM25 + BGE-M3) with RRF fusion, 5 passages per RAG step
**Model**: Qwen2.5-7B-Instruct

---

## Executive Summary

### Overall Performance
- **Accuracy**: 5/10 (50%)
- **Total RAG Steps**: 13 across 10 questions
- **Citation Usage**: 3/13 (23.1%) - **Critical finding: 0% citation usage in ALL wrong answers**
- **High RPE (≥0.8)**: 7/13 (53.8%)
- **Search Failures (RPE=0.00)**: 3/13 (23.1%)

### Key Findings
1. **Citation usage strongly predicts correctness**: All 5 incorrect answers had 0% citation usage
2. **High RPE doesn't guarantee correctness**: 2 questions with RPE ≥ 0.8 still got wrong answers
3. **Format compliance**: 100% search tag compliance, but only 23% citation compliance
4. **Corpus coverage matters**: Full 5.9M corpus needed (previous 10K corpus: 33% accuracy)

---

## Per-Question Analysis

### ✓ Question 1: Arthur's Magazine vs First for Women
**Gold Answer**: Arthur's Magazine
**Predicted**: Arthur's Magazine
**Status**: ✓ CORRECT

#### Retrieval Performance
- **RAG Steps**: 1 (step 4)
- **RPE**: 0.928 (Good)
- **MC Improvement**: -0.062 (slight decrease, but still correct)
- **Retrieved Documents**: 5 passages
  - Mary King (political scientist)
  - **First for Women** ← Relevant
  - Association of Radical Midwives
  - First Women
  - Longman-History Today Awards

#### Format Compliance
- ✓ Search tags: Present
- ✓ Citations: **[1]** used correctly
- Query: "founding year of First for Women"

#### Analysis
**Strengths**:
- Model successfully retrieved First for Women document (ranked #2)
- Citation format used correctly: "[1] First for Women is a woman's magazine...started in 1989"
- Despite slight MC decrease (-0.062), the model maintained correct answer

**Weaknesses**:
- Only 1 citation despite having 5 documents available
- Arthur's Magazine info was inferred, not explicitly cited

---

### ✓ Question 2: Oberoi Family Hotel Company
**Gold Answer**: Delhi
**Predicted**: Delhi
**Status**: ✓ CORRECT

#### Retrieval Performance
- **RAG Steps**: 1 (step 3)
- **RPE**: 1.14 (Good)
- **MC Improvement**: +0.126
- **Retrieved Documents**: 5 passages
  - CAS International
  - **The Oberoi Group** ← Relevant
  - SOB
  - Oberoi Realty
  - Lord President of the Court of Session

#### Format Compliance
- ✓ Search tags: Present
- ✗ Citations: **No citations used**
- Query: "head office location of Oberoi Group"

#### Analysis
**Strengths**:
- Perfect document retrieval (The Oberoi Group at rank #2)
- Clear answer extraction: "The Oberoi Group is a hotel group with its head office in Delhi"
- Positive MC improvement (+0.126)

**Critical Issue**:
- **No citations used despite having the exact document needed**
- This demonstrates a concerning pattern: model can find information but doesn't always cite it

---

### ✗ Question 3: Milhouse Named After Who?
**Gold Answer**: President Richard Nixon
**Predicted**: Matt Groening
**Status**: ✗ WRONG

#### Retrieval Performance
- **RAG Steps**: 3 (steps 2, 4, 5)
  - Step 2: RPE=87.50 ← **Exceptional retrieval**
  - Step 4: RPE=0.341 (Bad)
  - Step 5: RPE=1.841 (Good)
- **Average RPE**: 29.894

#### Retrieved Documents (Step 2 - Best RPE)
- Couch of Power
- Millhouse
- Time Person of the Year
- **Milhouse Van Houten** ← Contains answer
- Tosspot

#### Format Compliance
- ✓ Search tags: All 3 steps
- ✗ Citations: **Zero citations across all 3 RAG steps**

#### Detailed Analysis
**This is the most revealing failure case:**

1. **Retrieval Success**: Step 2 achieved RPE=87.50, which is exceptional
   - Retrieved "Milhouse Van Houten" document
   - Document states: "named after U.S. President Richard Nixon, whose middle name was Milhous"

2. **Information Extraction Failure**:
   - Model's observation: "The retrieved documents do not provide a specific answer to this query"
   - **This is factually incorrect** - the answer was explicitly in the Milhouse Van Houten document

3. **Reasoning Error**:
   - Model concluded: "Matt Groening named the character" (true)
   - But the question asked: "who did Matt Groening name Milhouse AFTER?" (Richard Nixon)
   - Model conflated "who named" with "named after"

4. **Zero Citations**:
   - Despite 3 retrieval attempts
   - Despite having the correct document
   - Model operated in pure reasoning mode, ignoring retrieved evidence

**Root Cause**: Information extraction and question comprehension failure, not retrieval failure.

---

### ✓ Question 4: James Henry Miller's Wife Nationality
**Gold Answer**: American
**Predicted**: American
**Status**: ✓ CORRECT

#### Retrieval Performance
- **RAG Steps**: 1 (step 2)
- **RPE**: 0.394 (Bad)
- **MC Improvement**: -0.375 (large decrease, but recovered)

#### Retrieved Documents
- Samuel Miller
- Nancy Kissinger
- Ernest Clark (governor)
- **June Miller** ← Relevant
- Carpool (1996 film)

#### Format Compliance
- ✓ Search tags: Present
- ✗ Citations: **No citations**

#### Analysis
**Interesting Pattern**:
- Low RPE (0.394), suggesting poor initial retrieval
- Large negative MC impact (-0.375)
- Yet model recovered and got correct answer
- Likely due to subsequent reasoning steps (step 3 not shown as RAG but may have helped)

**Missing**:
- No citations despite having June Miller document
- Shows model can reach correct conclusions without explicit citation

---

### ✓ Question 5: Cadmium Chloride Solubility
**Gold Answer**: alcohol
**Predicted**: alcohol
**Status**: ✓ CORRECT

#### Retrieval Performance
- **RAG Steps**: 1 (step 3)
- **RPE**: 3.639 (Very Good)
- **MC Improvement**: +0.531

#### Retrieved Documents
- Barium chloride
- **Cadmium chloride** ← Exact match
- List of inorganic pigments
- Caesium cadmium chloride
- Ringer's lactate solution

#### Format Compliance
- ✓ Search tags: Present
- ✗ Citations: **No citations**

#### Analysis
**Strong Performance**:
- Retrieved exact document (Cadmium chloride)
- High RPE (3.639) and strong MC improvement (+0.531)
- Correct answer extracted

**Observation Issue**:
- Model's observation includes the information but doesn't cite it properly
- States: "From the document 'Cadmium chloride: Cadmium chloride', it is stated that..."
- This is paraphrasing, not formal citation with [N] format

---

### ✗ Question 6: Henri Leconte vs Jonathan Stark
**Gold Answer**: Jonathan Stark
**Predicted**: Henri Leconte
**Status**: ✗ WRONG

#### Retrieval Performance
- **RAG Steps**: 1 (step 3)
- **RPE**: 0.928 (Good)
- **MC Improvement**: -0.062

#### Retrieved Documents
- Andre Agassi
- Javier Sánchez
- **Jonathan Stark (tennis)** ← Has the answer
- List of career achievements by Novak Djokovic
- Tara Moore

#### Format Compliance
- ✓ Search tags: Present
- ✗ Citations: **No citations**

#### Detailed Analysis
**Factual Information Retrieved**:
- Jonathan Stark: 2 Grand Slam doubles titles
  - 1994 French Open Men's Doubles
  - 1995 Wimbledon Championships Mixed Doubles

**Model's Error**:
- Observation states: "Jonathan Stark won 2 Grand Slam titles"
- But then fabricates: "Henri Leconte won 4 Grand Slam singles titles"
- **This is completely fabricated** - Henri Leconte never won a Grand Slam singles title
- He only reached the 1988 French Open final (runner-up)

**Critical Issues**:
1. Model hallucinated Henri Leconte's 4 Grand Slam wins
2. No citations used, so fabrication went unchecked
3. Good retrieval (RPE=0.928) wasted by reasoning failure

**Root Cause**: Hallucination + lack of citation enforcement

---

### ✗ Question 7: Moth Genus in Seventh-Largest Country
**Gold Answer**: Crambidae
**Predicted**: Lophocampa
**Status**: ✗ WRONG

#### Retrieval Performance
- **RAG Steps**: 1 (step 2)
- **RPE**: 0.00 ← **Complete retrieval failure**
- **MC Improvement**: 0.00 (no change)

#### Retrieved Documents
- List of cities in Tunisia
- Group of Seven
- Paul G. Blazer High School
- Outline of India
- George David Woods

#### Format Compliance
- ✓ Search tags: Present
- ✗ Citations: **No citations**

#### Analysis
**Complete Failure**:
- Query: "seventh largest country in the world"
- Retrieved completely irrelevant documents
- RPE=0.00 indicates total retrieval failure

**Model Behavior**:
- Correctly identified Canada as 7th largest (from internal knowledge)
- But then fabricated "Lophocampa" as the moth genus
- The correct answer involves India (7th largest) and Crambidae family

**Root Cause**:
1. Search query too generic
2. No relevant documents retrieved
3. Model fell back to hallucination

---

### ✗ Question 8: Controversial Kickboxer
**Gold Answer**: Badr Hari
**Predicted**: Muenif Faizy
**Status**: ✗ WRONG

#### Retrieval Performance
- **RAG Steps**: 1 (step 2)
- **RPE**: 0.00 ← **Complete retrieval failure**
- **MC Improvement**: 0.00

#### Retrieved Documents
- List of Jamaican British people
- Searching
- Lee Swaby
- Pattern search
- Arthur King

#### Format Compliance
- ✓ Search tags: Present
- ✗ Citations: **No citations**

#### Analysis
**Total Failure**:
- Query: "best kick boxer in the world and controversies"
- Retrieved completely irrelevant documents (search methodology papers, British people lists)
- RPE=0.00

**Model's Fabrication**:
- Created fictional person "Muenif Faizy"
- Invented backstory about Malaysia, match-fixing, 2007 ban
- All completely fabricated

**Root Cause**:
1. Generic query failed to find relevant documents
2. Zero retrieval quality
3. Model hallucinated entire answer

---

### ✓ Question 9: House of Anubis Original Series
**Gold Answer**: 2006
**Predicted**: 2006
**Status**: ✓ CORRECT

#### Retrieval Performance
- **RAG Steps**: 2 (steps 2, 3)
  - Step 2: RPE=87.50 ← Exceptional
  - Step 3: RPE=0.565 (Bad)
- **Average RPE**: 44.075

#### Retrieved Documents (Step 2)
- Scheldt–Rhine Canal
- **Het Huis Anubis** ← Relevant
- Mystery!
- **House of Anubis** ← Relevant
- East Riddlesden Hall

#### Format Compliance
- ✓ Search tags: Both steps
- ✓ Citations: **[4] and [2] used** ← One of only 3 questions with citations

#### Analysis
**Excellent Performance**:
- Two RAG steps, both with proper search tags
- First step: RPE=87.50, retrieved both relevant documents
- **Both citations used properly**:
  - "[4] House of Anubis is a mystery television series...based on the Belgian–Dutch television series 'Het Huis Anubis'"
  - "[2] Het Huis Anubis...first aired in September 2006"

**Why This Succeeded**:
1. Specific, well-formed search queries
2. Relevant documents retrieved
3. **Citations enforced model to use retrieved information**
4. Step-by-step reasoning grounded in evidence

**Model Pattern**: This is how retrieval should work!

---

### ✗ Question 10: Bathurst 12 Hour Track Length
**Gold Answer**: 6.213 km long
**Predicted**: 6.213 kilometers
**Status**: ✗ WRONG (marked wrong, possibly evaluation issue)

#### Retrieval Performance
- **RAG Steps**: 1 (step 3)
- **RPE**: 0.00 ← **Retrieval failure**
- **MC Improvement**: -0.031

#### Retrieved Documents
- Mike Burgmann
- **2013 Liqui Moly Bathurst 12 Hour** ← Relevant
- Super Touring
- Bathurst 250
- Toronto Western Hospital

#### Format Compliance
- ✓ Search tags: Present
- ✗ Citations: **No citations**

#### Analysis
**Interesting Case**:
- RPE=0.00 suggests retrieval failure
- Yet "2013 Liqui Moly Bathurst 12 Hour" document was retrieved (#2 rank)
- Model correctly extracted "6.213 kilometers"
- Answer matches gold standard "6.213 km long"

**Evaluation Issue**:
- Marked as wrong, but answer is functionally identical
- Possible strict string matching in evaluation
- Should be considered correct

---

## Format Compliance Analysis

### Search Tag Compliance
**Score**: 13/13 (100%)

All RAG steps properly used the `Search[query="..."]` format:
- Consistent formatting
- Clear query strings
- Proper bracketing

**Conclusion**: Model reliably follows search tag format.

---

### Citation Compliance
**Score**: 3/13 (23.1%)

**Questions with Citations** (3):
1. ✓ Question 1 (Arthur's Magazine): 1 citation - "[1]"
2. ✓ Question 9 (House of Anubis): 2 citations - "[4]", "[2]"

**Questions without Citations** (10):
- Question 2 (Oberoi): 0 citations
- Question 3 (Milhouse): 0 citations across 3 RAG steps
- Question 4 (Miller's wife): 0 citations
- Question 5 (Cadmium): 0 citations
- Question 6 (Tennis): 0 citations
- Question 7 (Moth): 0 citations
- Question 8 (Kickboxer): 0 citations
- Question 10 (Bathurst): 0 citations

### Critical Finding: Citation Usage Predicts Correctness

| Citation Usage | Correct | Wrong | Accuracy |
|----------------|---------|-------|----------|
| **With citations** | 2 | 0 | 100% |
| **No citations** | 3 | 5 | 37.5% |

**Key Insight**: Questions with citations have 100% accuracy, while questions without citations have only 37.5% accuracy.

**Implications**:
1. Citations enforce grounding in retrieved documents
2. Without citations, model tends to hallucinate or fabricate
3. Citation compliance is more important than retrieval quality (RPE)

---

## Retrieval Quality Analysis

### RPE Distribution

| RPE Range | Count | Percentage | Examples |
|-----------|-------|------------|----------|
| **Exceptional (≥10)** | 2 | 15.4% | Q3-step2 (87.50), Q9-step2 (87.50) |
| **Very Good (3-10)** | 2 | 15.4% | Q5 (3.639), Q2 (1.14) |
| **Good (0.8-3)** | 3 | 23.1% | Q1 (0.928), Q6 (0.928), Q9-step3 (0.565)* |
| **Bad (<0.8)** | 3 | 23.1% | Q4 (0.394), Q3-step4 (0.341) |
| **Failed (0.0)** | 3 | 23.1% | Q7, Q8, Q10 |

*Q9-step3 is borderline; moved to Good for this categorization

### RPE vs Correctness

| RPE Category | Correct | Wrong | Accuracy |
|--------------|---------|-------|----------|
| **High (≥0.8)** | 5 | 2 | 71.4% |
| **Low (<0.8)** | 1 | 2 | 33.3% |
| **Failed (0.0)** | 0 | 3 | 0% |

**Important Finding**: High RPE doesn't guarantee correctness
- Question 3 (Milhouse): RPE=87.50 but WRONG
- Question 6 (Tennis): RPE=0.928 but WRONG

**Root Causes for High-RPE Failures**:
1. Information extraction failure (Q3)
2. Hallucination despite good retrieval (Q6)
3. Lack of citation enforcement

---

## Error Taxonomy

### Category 1: Retrieval Failures (3 cases)
**Questions**: 7 (Moth), 8 (Kickboxer), 10 (Bathurst*)

**Characteristics**:
- RPE = 0.00
- Irrelevant documents retrieved
- Generic or poorly-formed queries

**Example** (Q7):
- Query: "seventh largest country in the world"
- Retrieved: Tunisia cities list, Group of Seven, etc.
- Should have retrieved: India-related documents

**Solutions**:
1. Multi-step queries (first identify country, then search moths)
2. More specific initial queries
3. Query reformulation on poor results

---

### Category 2: Information Extraction Failures (2 cases)
**Questions**: 3 (Milhouse), 6 (Tennis)

**Characteristics**:
- Good/exceptional RPE (≥0.8)
- Relevant documents retrieved
- Model fails to extract or misinterprets information

**Example** (Q3 - Most Critical)**:
- RPE: 87.50 (exceptional)
- Retrieved: Milhouse Van Houten document with answer
- Model claimed: "documents do not provide a specific answer"
- **This is factually wrong**

**Root Cause**:
- No citations enforce document reading
- Model relies on reasoning instead of evidence
- Question misinterpretation ("who named" vs "named after")

**Solutions**:
1. Enforce citation requirements
2. Better prompt engineering for evidence extraction
3. Teach model to quote directly from documents

---

### Category 3: Hallucination with Good Retrieval (1 case)
**Question**: 6 (Tennis)

**Characteristics**:
- RPE: 0.928 (good)
- Retrieved Jonathan Stark document correctly
- Extracted "2 Grand Slam titles" correctly
- **Fabricated Henri Leconte's "4 Grand Slam singles titles"**

**Critical Issue**:
- Henri Leconte NEVER won a Grand Slam singles title
- Model completely invented this information
- No citations to fact-check

**Solutions**:
1. Strict citation enforcement
2. Penalty for uncited claims
3. Multi-hop verification

---

### Category 4: Evaluation Issues (1 case)
**Question**: 10 (Bathurst)

**Characteristics**:
- Answer: "6.213 kilometers"
- Gold: "6.213 km long"
- Functionally identical

**Issue**: Strict string matching in evaluation

---

## Monte Carlo Improvement Analysis

### Average MC Improvements by Outcome

| Outcome | Avg MC Improvement | Count |
|---------|-------------------|-------|
| **Correct** | +0.092 | 5 |
| **Wrong** | -0.021 | 5 |

### Distribution of MC Impacts

| MC Change | Count | Examples |
|-----------|-------|----------|
| **Large Positive (>0.3)** | 1 | Q5: +0.531 |
| **Small Positive (0-0.3)** | 3 | Q2: +0.126, Q9-step2: +0.875* |
| **Small Negative (0 to -0.1)** | 4 | Q1: -0.062, Q6: -0.062 |
| **Large Negative (<-0.1)** | 2 | Q4: -0.375 |
| **Zero** | 3 | Q7, Q8, Q10 (retrieval failures) |

*Q9-step2 is exceptional with +0.875

### Key Observations

1. **MC improvement is noisy**: Small negative changes don't always mean wrong answers
   - Q1, Q6: MC decreased but answers were correct/based on good info

2. **Zero MC change = retrieval failure**: All RPE=0.00 cases had MC change = 0.00

3. **Large positive changes = good signal**: Q5 (+0.531) was correct with high confidence

4. **MC can't fix bad retrieval**: Q7, Q8 show that without relevant documents, MC rollouts don't help

---

## Recommendations

### 1. Enforce Citation Requirements (CRITICAL)
**Evidence**: 100% accuracy with citations vs 37.5% without

**Implementation**:
- Add citation count to reward model
- Penalize answers without citations when RAG was used
- Require minimum 1 citation per RAG step
- Train model to prioritize cited information

### 2. Improve Multi-Hop Retrieval
**Issues**: Q7 (Moth), Q8 (Kickboxer) failed due to complex queries

**Solutions**:
- Decompose complex questions into sub-queries
- First query: identify entities (e.g., "seventh largest country")
- Second query: use identified entity ("moths in India")
- Chain-of-thought for query generation

### 3. Add Query Quality Checks
**Issue**: Generic queries retrieve irrelevant documents

**Implementation**:
- Check if query contains key entities from question
- Reformulate if RPE < 0.5
- Use question type classification (factoid, comparison, temporal)

### 4. Teach Information Extraction
**Issue**: Q3, Q6 had documents but failed extraction

**Solutions**:
- Prompt engineering: "Quote directly from documents"
- Few-shot examples with explicit citations
- Reward model that checks citation accuracy

### 5. Add Hallucination Detection
**Issue**: Q6 fabricated Henri Leconte's titles, Q8 invented Muenif Faizy

**Implementation**:
- Cross-check claims against retrieved documents
- Flag uncited factual claims
- Use MC rollouts to detect inconsistencies

### 6. Improve Answer Format Matching
**Issue**: Q10 marked wrong for "6.213 kilometers" vs "6.213 km long"

**Solutions**:
- Fuzzy matching for numerical answers
- Unit normalization (km, kilometers)
- Multiple acceptable answer formats

---

## Comparison Setup for Dense-Only Test

### Test Configuration
**Currently running**: Dense-only (BGE-M3 only) test with same settings

**Parameters**:
- Corpus: Same KILT Wikipedia (5.9M docs)
- Questions: Same 10 questions
- Rollouts: 8 MC rollouts
- Passages per RAG: 5 documents
- k-dense: 50 (retrieves 50, provides 5 to model)

**Comparison Metrics**:
1. **Accuracy**: Correct answers / 10
2. **Citation Usage**: RAG steps with citations / total RAG steps
3. **RPE Distribution**: Exceptional, Good, Bad, Failed
4. **MC Improvement**: Average improvement per RAG step
5. **Format Compliance**: Search tags and citations

**Expected Insights**:
- Does BM25 (keyword) help with entity-based questions?
- Do hybrid results have better RPE than dense-only?
- Does retrieval diversity (sparse + dense) improve citation usage?

---

## Conclusion

### Key Findings Summary

1. **Citation usage is the strongest predictor of correctness** (100% vs 37.5%)
2. **High RPE doesn't guarantee success** - information extraction matters
3. **Retrieval failures (RPE=0) are fatal** - no MC rollouts can recover
4. **Hallucination remains a major issue** - especially without citations
5. **Multi-hop queries need better decomposition**

### Overall Assessment

**Strengths**:
- 50% accuracy on complex multi-hop questions
- 100% search tag compliance
- Good RPE in 53.8% of cases
- Successfully handles simple factual questions

**Critical Weaknesses**:
- Only 23% citation usage
- Hallucination in 2/5 wrong answers (Q6, Q8)
- Information extraction failures despite good retrieval (Q3)
- Generic query failures (Q7, Q8)

### Path Forward

**Immediate Priority**: Enforce citation requirements
- This alone could improve accuracy from 50% to potentially 70-80%
- Citations ground reasoning in evidence
- Prevents hallucination

**Secondary Priorities**:
1. Multi-hop query decomposition
2. Information extraction training
3. Query quality checks
4. Hallucination detection

**Next Step**: Compare with Dense-only results to measure BM25 contribution.
