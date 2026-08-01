# Local AI Agent v0.2.0

## What changed

- Expanded the bundled RAG evaluation from 4 to 30 cases: 25 answerable cases across nine domains and 5 abstention cases.
- Bound evaluation relevance judgments to immutable content-derived source IDs and validated the dataset hash and row count.
- Added a deterministic BM25 keyword baseline alongside semantic retrieval.
- Added versioned JSON and Markdown reports with exact model digests, runtime versions, dataset/case hashes, per-case outcomes, rankings, and latency.
- Split end-to-end behavior into explicit metrics: expected-action accuracy, answer success rate, abstention recall, citation validity, and a lexical reference-term support proxy.
- Added stricter manifest/schema validation, duplicate-observation rejection, CLI/report tests, typing, coverage reporting, dependency auditing, package-asset checks, and installed-wheel smoke testing.
- Added a scoped security policy for the embedded-Chroma deployment boundary.
- Added a silent live-dashboard demo showing the real question trigger, answer, validated citations, and expanded evidence.

## Verified local benchmark

Runtime:

- Ollama `0.32.5`
- `llama3.2:latest` digest `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`
- `mxbai-embed-large:latest` digest `468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8`
- 123 bundled restaurant reviews
- 30 evaluation cases; retrieval limit 5

Retrieval results on the 25 answerable cases:

| Metric | Semantic retrieval | BM25 baseline |
| --- | ---: | ---: |
| Recall@5 | 0.913 | 0.770 |
| Hit rate@5 | 1.000 | 0.880 |
| MRR@5 | 0.960 | 0.753 |

Model-dependent RAG results:

| Metric | Result |
| --- | ---: |
| Citation validity | 0.560 |
| Reference-term support proxy | 0.520 |
| Answer success rate | 0.560 |
| Abstention recall | 1.000 |
| Expected-action accuracy | 0.633 |
| Mean latency | 4.359 s |
| Median latency | 2.006 s |

Outcomes: 14 answered, 13 model abstentions, and 3 citation-validation rejections.

Authoritative artifacts:

- `evaluation/results/v0.2.0-ollama-0.32.5/evaluation-report.json`
- `evaluation/results/v0.2.0-ollama-0.32.5/README.md`
- `evaluation/results/v0.2.0-ollama-0.32.5/run.log`
- `docs/assets/local-ai-agent-v0.2-demo.mp4`

Report SHA-256: `5700748c2abe1b5446a850c08291c41e8bbab6aff663cd5ac783513880d02718`

## Verification

- 61 tests passed.
- Ruff check and format check passed.
- Mypy passed for six core modules.
- Coverage reports all six core modules, including `evaluation.py`: 84% overall. No arbitrary pass threshold is imposed.
- Wheel and source distribution passed Twine checks.
- Wheel contains both bundled evaluation/data assets and the installed CLI smoke test passed.
- Runtime dependency audit reported no known vulnerabilities other than one explicitly ignored advisory: `CVE-2026-45829` / `PYSEC-2026-311`.

## Evidence boundaries

These results apply only to the checked-in dataset, case manifest, model digests, runtime, retrieval limit, and machine recorded in the report. Known-positive relevance judgments are not exhaustive, so the semantic-versus-BM25 comparison is not evidence of general retrieval superiority. Citation validity checks whether citations resolve to retrieved evidence; the reference-term metric is a lexical support proxy, not claim-level factual-faithfulness evaluation.

The Chroma advisory exception is accepted only for embedded, process-local Chroma. It must be removed if a fixed compatible release becomes available or the application exposes/uses a remote Chroma server.
