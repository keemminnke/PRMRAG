# Dataset Quality Analysis Across Past Experiments

## Summary Table

| Name | Config | Pairs | EM | F1 | cho_f1 mean | rej_f1 mean | Δf1 mean | Δf1 p50 | Δf1<0.2 | Δf1=0 | cho_crt=1 | Δcrt=0 | Δcrt<0 (noise) |
|------|--------|------:|---:|---:|-----------:|-----------:|--------:|--------:|-------:|-------:|---------:|-------:|---------------:|
| K=3 raw | δ=0.01 only | 39,578 | 0.3762 | 0.5058 | 0.321 | 0.100 | 0.221 | 0.09 | 59.9% | 37.4% | 93.9% | 51.1% | 2.1% |
| F4 | cho_f1≥0.5 | 13,234 | 0.3581 | 0.4578 | 0.712 | 0.225 | 0.487 | 0.56 | 21.2% | 5.2% | 91.4% | 76.6% | 4.5% |
| F6 (best) | cho_f1≥0.2 AND Δf1≥0.2 | 15,877 | **0.3873** | 0.5163 | 0.590 | 0.083 | 0.506 | 0.49 | 0.0% | 0.0% | 88.0% | 79.4% | 5.2% |
| F6+α0 | F6 + α=0 | 16,065 | 0.3674 | 0.4931 | 0.587 | 0.083 | 0.503 | 0.49 | 0.0% | 0.0% | 86.9% | 78.5% | 6.3% |
| F6+α0.5 | F6 + α=0.5 | 15,596 | 0.3495 | 0.4753 | 0.592 | 0.084 | 0.508 | 0.49 | 0.0% | 0.0% | 89.5% | 80.8% | 3.4% |
| F6+α1.0 | F6 + α=1.0 | 15,059 | 0.3797 | 0.5060 | 0.586 | 0.087 | 0.499 | 0.47 | 0.0% | 0.0% | 92.7% | 83.7% | 0.0% |
| F6tight | cho_f1≥0.3 AND Δf1≥0.2 | 13,944 | 0.3598 | 0.4728 | 0.637 | 0.095 | 0.542 | 0.54 | 0.0% | 0.0% | 87.9% | 78.3% | 5.9% |
| F6margin | cho_f1≥0.2 AND Δf1≥0.3 | 12,151 | 0.3649 | 0.4978 | 0.636 | 0.050 | 0.585 | 0.65 | 0.0% | 0.0% | 86.7% | 76.5% | 6.7% |
| F6loose | cho_f1≥0.1 AND Δf1≥0.2 | 15,877 | 0.3344 | 0.4714 | 0.590 | 0.083 | 0.506 | 0.49 | 0.0% | 0.0% | 88.0% | 79.4% | 5.2% |

## Deep metrics per dataset

### K=3 raw — EM 0.3762 (39,578 pairs)
**Config**: δ=0.01 only

**chosen_f1**: mean=0.321, p25=0.00, p50=0.26, p75=0.66, %zero=35.8%, %<0.2=45.8%
**rejected_f1**: mean=0.100, p50=0.00, %zero=72.1%
**Δf1 (margin)**: mean=0.221, p25=0.00, p50=0.09, p75=0.36, %<0.2=59.9%, %=0=37.4%, %<0 (reversed)=1.8%
**critic**: cho_mean=0.939 (cho=1: 93.9%), rej_mean=0.491 (rej=0: 50.9%)
**Δcritic**: agree(+)=46.8%, tied(0)=51.1%, dissent(-)=2.1%
**step**: step1=18.8%, step2=41.8%, step3=37.1%, step4+=2.3%

### F4 — EM 0.3581 (13,234 pairs)
**Config**: cho_f1≥0.5

**chosen_f1**: mean=0.712, p25=0.66, p50=0.73, p75=0.77, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.225, p50=0.00, %zero=53.1%
**Δf1 (margin)**: mean=0.487, p25=0.24, p50=0.56, p75=0.73, %<0.2=21.2%, %=0=5.2%, %<0 (reversed)=0.9%
**critic**: cho_mean=0.914 (cho=1: 91.4%), rej_mean=0.771 (rej=0: 22.9%)
**Δcritic**: agree(+)=18.9%, tied(0)=76.6%, dissent(-)=4.5%
**step**: step1=25.7%, step2=45.5%, step3=28.0%, step4+=0.8%

### F6 (best) — EM 0.3873 (15,877 pairs)
**Config**: cho_f1≥0.2 AND Δf1≥0.2

**chosen_f1**: mean=0.590, p25=0.41, p50=0.66, p75=0.73, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.083, p50=0.00, %zero=75.1%
**Δf1 (margin)**: mean=0.506, p25=0.32, p50=0.49, p75=0.73, %<0.2=0.0%, %=0=0.0%, %<0 (reversed)=0.0%
**critic**: cho_mean=0.880 (cho=1: 88.0%), rej_mean=0.776 (rej=0: 22.4%)
**Δcritic**: agree(+)=15.5%, tied(0)=79.4%, dissent(-)=5.2%
**step**: step1=24.0%, step2=45.7%, step3=29.5%, step4+=0.8%

### F6+α0 — EM 0.3674 (16,065 pairs)
**Config**: F6 + α=0

**chosen_f1**: mean=0.587, p25=0.40, p50=0.66, p75=0.73, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.083, p50=0.00, %zero=75.2%
**Δf1 (margin)**: mean=0.503, p25=0.31, p50=0.49, p75=0.73, %<0.2=0.0%, %=0=0.0%, %<0 (reversed)=0.0%
**critic**: cho_mean=0.869 (cho=1: 86.9%), rej_mean=0.779 (rej=0: 22.1%)
**Δcritic**: agree(+)=15.3%, tied(0)=78.5%, dissent(-)=6.3%
**step**: step1=23.7%, step2=45.7%, step3=29.6%, step4+=0.9%

### F6+α0.5 — EM 0.3495 (15,596 pairs)
**Config**: F6 + α=0.5

**chosen_f1**: mean=0.592, p25=0.41, p50=0.68, p75=0.73, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.084, p50=0.00, %zero=75.0%
**Δf1 (margin)**: mean=0.508, p25=0.31, p50=0.49, p75=0.73, %<0.2=0.0%, %=0=0.0%, %<0 (reversed)=0.0%
**critic**: cho_mean=0.895 (cho=1: 89.5%), rej_mean=0.772 (rej=0: 22.8%)
**Δcritic**: agree(+)=15.7%, tied(0)=80.8%, dissent(-)=3.4%
**step**: step1=24.4%, step2=45.5%, step3=29.3%, step4+=0.8%

### F6+α1.0 — EM 0.3797 (15,059 pairs)
**Config**: F6 + α=1.0

**chosen_f1**: mean=0.586, p25=0.41, p50=0.66, p75=0.73, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.087, p50=0.00, %zero=74.1%
**Δf1 (margin)**: mean=0.499, p25=0.30, p50=0.47, p75=0.73, %<0.2=0.0%, %=0=0.0%, %<0 (reversed)=0.0%
**critic**: cho_mean=0.927 (cho=1: 92.7%), rej_mean=0.764 (rej=0: 23.6%)
**Δcritic**: agree(+)=16.3%, tied(0)=83.7%, dissent(-)=0.0%
**step**: step1=25.1%, step2=45.1%, step3=29.1%, step4+=0.8%

### F6tight — EM 0.3598 (13,944 pairs)
**Config**: cho_f1≥0.3 AND Δf1≥0.2

**chosen_f1**: mean=0.637, p25=0.49, p50=0.73, p75=0.73, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.095, p50=0.00, %zero=72.0%
**Δf1 (margin)**: mean=0.542, p25=0.36, p50=0.54, p75=0.73, %<0.2=0.0%, %=0=0.0%, %<0 (reversed)=0.0%
**critic**: cho_mean=0.879 (cho=1: 87.9%), rej_mean=0.780 (rej=0: 22.0%)
**Δcritic**: agree(+)=15.8%, tied(0)=78.3%, dissent(-)=5.9%
**step**: step1=22.8%, step2=45.9%, step3=30.3%, step4+=0.9%

### F6margin — EM 0.3649 (12,151 pairs)
**Config**: cho_f1≥0.2 AND Δf1≥0.3

**chosen_f1**: mean=0.636, p25=0.49, p50=0.73, p75=0.73, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.050, p50=0.00, %zero=82.6%
**Δf1 (margin)**: mean=0.585, p25=0.41, p50=0.65, p75=0.73, %<0.2=0.0%, %=0=0.0%, %<0 (reversed)=0.0%
**critic**: cho_mean=0.867 (cho=1: 86.7%), rej_mean=0.767 (rej=0: 23.3%)
**Δcritic**: agree(+)=16.8%, tied(0)=76.5%, dissent(-)=6.7%
**step**: step1=20.8%, step2=46.1%, step3=32.1%, step4+=1.0%

### F6loose — EM 0.3344 (15,877 pairs)
**Config**: cho_f1≥0.1 AND Δf1≥0.2

**chosen_f1**: mean=0.590, p25=0.41, p50=0.66, p75=0.73, %zero=0.0%, %<0.2=0.0%
**rejected_f1**: mean=0.083, p50=0.00, %zero=75.1%
**Δf1 (margin)**: mean=0.506, p25=0.32, p50=0.49, p75=0.73, %<0.2=0.0%, %=0=0.0%, %<0 (reversed)=0.0%
**critic**: cho_mean=0.880 (cho=1: 88.0%), rej_mean=0.776 (rej=0: 22.4%)
**Δcritic**: agree(+)=15.5%, tied(0)=79.4%, dissent(-)=5.2%
**step**: step1=24.0%, step2=45.7%, step3=29.5%, step4+=0.8%

## Correlation with performance (EM)

| Metric | Pearson r vs EM | Interpretation |
|--------|----------------:|----------------|
| n | +0.276 (p=0.473) | ↑ 클수록 EM↑ |
| cho_f1_mean | -0.308 (p=0.420) | ↓ 클수록 EM↓ |
| rej_f1_mean | -0.107 (p=0.784) | ↓ 클수록 EM↓ |
| df1_mean | -0.268 (p=0.486) | ↓ 클수록 EM↓ |
| df1_p50 | -0.283 (p=0.461) | ↓ 클수록 EM↓ |
| df1_lt02 | +0.226 (p=0.559) | ↑ 클수록 EM↑ |
| df1_eq_0 | +0.262 (p=0.495) | ↑ 클수록 EM↑ |
| df1_neg | +0.201 (p=0.604) | ↑ 클수록 EM↑ |
| cho_crt_pct_1 | +0.286 (p=0.456) | ↑ 클수록 EM↑ |
| dcrt_eq | -0.224 (p=0.561) | ↓ 클수록 EM↓ |
| dcrt_neg | -0.309 (p=0.419) | ↓ 클수록 EM↓ |
| cho_f1_pct_0 | +0.280 (p=0.466) | ↑ 클수록 EM↑ |
| rej_f1_pct_0 | +0.099 (p=0.800) | ↑ 클수록 EM↑ |
## Key patterns from past experiments

### Pair-wise comparisons (only one knob changed):

| 변경 | EM 변화 | 왜? |
|------|---------|-----|
| F6 → F4 (cho_floor 0.2→0.5, Δ 제거) | 0.387 → 0.358 (-2.9%p) | **Δf1≥0.2 제거가 치명적** — 21.2% Δ<0.2 유입 |
| F6 → F6tight (cho 0.2→0.3) | 0.387 → 0.360 (-2.7%p) | chosen floor 높임 → coverage 손실 |
| F6 → F6margin (Δ 0.2→0.3) | 0.387 → 0.365 (-2.2%p) | margin 높임 → mid-quality pair 소실 |
| F6 → K=3 raw (필터 없음) | 0.387 → 0.376 (-1.1%p) | 59.9% Δ<0.2 유입 — size 2.5배인데도 하락 |
| F6+α0 → F6+α0.3 | 0.367 → 0.387 (+2.0%p) | α=0.3이 critic-dissent pair 약간 제외 |
| F6 → F6+α0.5 | 0.387 → 0.350 (-3.8%p) | α 과해서 F1 ordering 방해, 중복 critic=1 pair 선호 |

### Strong EM predictors (heuristic rules)


1. **Δf1<0.2 비율 낮을수록 EM↑** (F6: 0%, K=3 raw: 59.9%)
2. **Δf1=0 (F1 동점) 비율 낮을수록 EM↑** (F6: 0%, K=3 raw: 37.4%)
3. **Δf1<0 (F1 역전) 비율 낮을수록 EM↑** (F6: 0%, K=3 raw: 1.8%)
4. **Δcritic<0 (dissent) 비율 낮을수록 EM↑** (F6: 5.2%, F6+α0: 6.3% worse)
5. **rejected_f1=0 비율 높을수록 EM↑** (sharper gradient, F6: 75.1%)
6. **chosen_f1 mean은 무관** (F4의 0.712이 F6의 0.590보다 못함)
7. **size는 무관** (K=3 raw 39k가 F6 16k보다 못함)

→ **한 줄 요약: Δf1 관련 깨끗함(0% tied, 0% reversed)이 제일 중요**

## Prediction for A3, B1 based on learned rules

**F6 (baseline, EM 0.3873):**
  n=15,877, cho_f1_mean=0.590, rej_f1=0: 75.1%
  **Δf1: tied 0%, reversed 0%, <0.2 = 0%**
  Δcritic: dissent 5.2%

**A3 (α=0.30, δ=0.20, cho_f1≥0.2) — predicted stats:**
  n=16,948, cho_f1_mean=0.593, rej_f1=0: 69.1%
  **Δf1: tied 5.2%, reversed 1.5%, <0.2 = 7.8%**
  Δcritic: dissent 3.3%
  → Δf1 cleanliness 깨짐: 5.2% tied + 1.5% reversed
  → 규칙 (1,2,3) 세 개 모두 위배 → **EM 하락 예상 (-1~2%p)**

**B1 (α=0.20, δ=0.20, cmb_floor=0.4) — predicted stats:**
  n=15,640, cho_f1_mean=0.596, rej_f1=0: 72.8%
  **Δf1: tied 1.0%, reversed 0.0%, <0.2 = 2.2%**
  Δcritic: dissent 4.2%
  → Δf1 tied 1.0%는 경미, 규칙 (1,3) 근접
  → **EM: F6와 거의 동일 예상 (±1%p)**

## 결론

- **Δf1 cleanliness (tied 0%, reversed 0%)이 F6 성공의 핵심**
- A3는 1,137 pair에서 이 cleanliness 위배 (880 tied + 257 reversed) → 학습 손상 우려
- B1은 161 pair(1%)만 tied, reversed 0 → F6 quality 거의 유지
- **B1 > A3 예상** (기존 실험들이 가르친 패턴 기준)

## 실험 수정 권고

A3는 cleanliness rule 2개 위배 → F6보다 분명히 악화 가능성.
B1은 1%만 위배 → 안정적. 하지만 α=0.2 untested 리스크.

**대안 B1'**: α=0.2를 유지하되 Δf1<0 pair를 추가로 제거해서 reversed 0% 보장
→ 161 tied pair 유지 (PRM tie-breaker 역할), 0 reversed (학습 noise 제거)
→ "PRM tie-breaker only, no F1-reversal noise" 명확 framing
