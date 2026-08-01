# Evaluation report

Generated: `2026-07-31T16:42:39.160738Z`

## Scope

- Cases: **30** (25 answerable, 5 abstention)
- Chat model: `llama3.2`
- Embedding model: `mxbai-embed-large`
- Ollama runtime: `0.32.5`
- Evidence limit: **5**
- Dataset SHA-256: `8ace5c1cb728c3c8a355d90a414849ebd645e021aae8a5b32303ef310917254e`
- Cases SHA-256: `b46d008043ad5a815fc6f9563362901f2a4368a408739c811ae80a614556a141`

## Retrieval comparison

| Retriever | Recall@k | Hit rate@k | MRR@k |
| --- | ---: | ---: | ---: |
| Semantic | 0.913 | 1.000 | 0.960 |
| BM25 keyword baseline | 0.770 | 0.880 | 0.753 |

## Model-dependent results

| Metric | Score |
| --- | ---: |
| Citation validity | 0.560 |
| Reference-term support proxy | 0.520 |
| Expected-action accuracy | 0.633 |
| Answer success (answerable cases) | 0.560 |
| Abstention recall (abstention cases) | 1.000 |

## Timing

- Total RAG latency: **130.76s**
- Mean per case: **4358.7ms**
- Median per case: **2006.2ms**

These scores describe this fixed dataset, case set, retrieval limit, and local model configuration. Relevance judgments are known-positive, not exhaustive. The reference-term support score is a transparent heuristic, not an LLM judge or a general factuality guarantee.
