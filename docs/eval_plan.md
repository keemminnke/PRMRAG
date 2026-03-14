# Evaluation Plan: PRO-Step Main Table

## Table Structure
12 methods × 5 benchmarks × 2 metrics (EM, F1) + Avg = **144 cells, 60 runs**

## Benchmarks
| Dataset | Type | Test Size | Source |
|---------|------|-----------|--------|
| PopQA | Single-hop | 11,267 | FlashRAG |
| HotpotQA | Multi-hop | 7,405 | FlashRAG (dev split) |
| 2WikiMultiHopQA | Multi-hop | 12,576 | FlashRAG (dev split) |
| Bamboogle | Multi-hop (OOD) | 125 | FlashRAG |
| MuSiQue | Multi-hop (OOD) | 2,417 | FlashRAG |

## Methods

### Group 1: No training, FlashRAG built-in (바로 실행 가능)
| # | Type | Method | Pipeline | Model | 비고 |
|---|------|--------|----------|-------|------|
| 1 | Zero-shot | Naïve Generation | SequentialPipeline (no retrieval) | Qwen2.5-7B-Instruct | |
| 2 | Zero-shot | Standard RAG | SequentialPipeline | Qwen2.5-7B-Instruct + BGE | |
| 3 | Active | FLARE | FLAREPipeline | Qwen2.5-7B-Instruct + BGE | |
| 4 | RAG-CoT | IRCoT | IRCOTPi
peline | Qwen2.5-7B-Instruct + BGE | |
| 5 | RAG-CoT | Iter-RetGen | IterativePipeline | Qwen2.5-7B-Instruct + BGE | |

### Group 2: Checkpoint 다운로드 필요
| # | Type | Method | Pipeline | Model | Download |
|---|------|--------|----------|-------|----------|
| 7 | Active | Self-RAG | SelfRAGPipeline | selfrag/selfrag_llama2_13b (~26GB) | HF |
| 8 | Reasoning | Search-R1 | SearchR1Pipeline | PeterJinGo/SearchR1-...-qwen2.5-7b-it-em-ppo (~15GB) | HF |
| 9 | Reasoning | AutoRAG | ReasoningPipeline | ICTNLP/Auto-RAG-Llama-3-8B-Instruct (~16GB) | HF |

### Group 3: 모델 공개 여부 확인 필요 (없으면 논문 수치 인용)
| # | Type | Method | 확인사항 |
|---|------|--------|---------|
| 10 | Reasoning | Search-o1 | 공개 checkpoint? 없으면 ReasonRAG Table 2 인용 |
| 11 | Reasoning | ReasonRAG | 데이터(RAG_ProGuide)만 공개, 학습된 모델 checkpoint? 없으면 직접 학습 or 인용 |

### Group 4: Ours
| # | Type | Method | Model | 비고 |
|---|------|--------|-------|------|
| 12 | Reasoning | PRO-Step (ours) | Qwen2.5-7B-Instruct + DPO (MCTS) | outputs/dpo_policy_mcts_v1 |

## Shared Infrastructure
- **Retriever**: BGE-base-en-v1.5 + FAISS IVF4096 index (KILT Wikipedia)
- **Corpus**: Wikidump 2018 (FlashRAG format, 5.9M docs)
- **Generator framework**: vLLM (TP=2) for speed
- **Eval metrics**: EM, F1 (FlashRAG Evaluator)

## Execution Order
1. **DPO 학습 완료 대기** (현재 진행 중)
2. **DPO 모델 merge** (LoRA → full model)
3. **Group 1 실행** (6 methods × 5 datasets = 30 runs, 추가 다운로드 없음)
4. **Group 2 checkpoint 다운로드** + 실행 (3 methods × 5 = 15 runs)
5. **Group 3 확인** → checkpoint 있으면 실행, 없으면 인용
6. **PRO-Step 평가** (1 method × 5 = 5 runs)
7. **(Optional) Recovering Trajectories** → 추가 DPO pair → 재학습 → 재평가

## Estimated Time
- Group 1 (inference only): ~2-3시간/dataset × 5 = ~12시간
- Group 2 (download + inference): ~1시간 다운 + ~10시간 inference
- Group 4 (ours): ~3시간
- **Total**: ~25-30시간 (병렬화 시 단축 가능)

## Config Template (FlashRAG)
```yaml
# 공통
retrieval_method: "e5"  # or "bge"
retrieval_topk: 3
corpus_path: "data/kilt/kilt_corpus_flashrag.jsonl"
index_path: "data/indexes/bge_Flat_IVF4096.index"
framework: "vllm"
gpu_memory_utilization: 0.85
tensor_parallel_size: 2
metrics: ["em", "f1"]

# Method별 override
# Naïve: retrieval_topk: 0
# Standard RAG: default
# FLARE: pipeline_class: FLAREPipeline
# IRCoT: pipeline_class: IRCOTPipeline
# ...
```

## Notes
- ReasonRAG와 동일 조건: BGE retriever, top-3, Wikidump 2018
- Self-RAG만 Llama2-13B (원논문 모델), 나머지는 Qwen2.5-7B 계열
- Bamboogle, MuSiQue는 OOD (학습 데이터에 없음) → generalization 평가
