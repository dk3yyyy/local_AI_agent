# 🍕 Local Restaurant Review Intelligence

[![CI](https://github.com/dk3yyyy/local_AI_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/dk3yyyy/local_AI_agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A local-first review-intelligence application powered by Ollama and embedded ChromaDB. Explore restaurant-review metrics, filter evidence, upload compatible CSV datasets, and ask grounded questions with validated source citations.

By default, runtime review data, embeddings, and prompts are processed by embedded Chroma and Ollama on the same machine. If `OLLAMA_HOST` points to another computer or hosted endpoint, prompts and retrieved review excerpts are sent there. Installing dependencies and downloading models also use the network.

## Highlights

- **Visual dashboard:** rating metrics, distribution chart, date and rating filters, and review browser.
- **Validated citations:** short evidence citations map strictly to stable retrieved source IDs; invented or missing citations fail safely.
- **Safe offline state:** analytics still load when Ollama is unavailable, while the app shows exact setup commands instead of crashing.
- **Adaptive CSV upload:** automatically detect common headers, manually map unfamiliar names, and isolate every dataset in content-addressed Chroma storage.
- **Reconciled indexing:** content-derived IDs survive reordering; additions, changed records, and deletions are synchronized safely.
- **Measured RAG:** a versioned 30-case evaluation across nine answerable domains plus abstention compares semantic retrieval with a deterministic BM25 keyword baseline, then measures citation validity, a transparent reference-term support proxy, expected-action accuracy, answer success, and abstention recall.
- **Installable CLI:** the packaged `local-ai-agent` command exposes status, ask, chat, and evaluate workflows.
- **Local execution by default:** Ollama handles embeddings and answer generation unless a remote host is explicitly configured.

See the [architecture diagram and boundary notes](docs/architecture.md).

## Live workflow demo

[Watch the silent 48-second dashboard demo](docs/assets/local-ai-agent-v0.2-demo.mp4). It shows the real local question trigger, indexing state, generated answer, validated citations, and an expanded source record using the bundled dataset.

## Requirements

- [Python 3.11 or newer](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download)
- Enough local memory and disk space for the selected models

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/dk3yyyy/local_AI_agent.git
cd local_AI_agent
uv sync
```

### 2. Start Ollama

The Ollama desktop application starts the service automatically on supported systems. For a manual or headless installation, run:

```bash
ollama serve
```

Keep that process running in another terminal.

### 3. Download the required models

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2
```

| Model | Purpose |
| --- | --- |
| `mxbai-embed-large` | Creates embeddings for semantic review retrieval |
| `llama3.2` | Generates answers from retrieved review evidence |

Model downloads can be several gigabytes. The application never downloads them silently.

### 4. Launch the dashboard

```bash
uv run streamlit run dashboard.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The dashboard can immediately display dataset metrics and reviews. AI questions are enabled only after Ollama and both required models pass the health check.

## Dashboard workflow

1. Use the bundled 123-review dataset or upload a CSV.
2. Confirm the suggested column mapping. Only review text is required.
3. Filter by any mapped ratings, dates, sentiments, restaurants, or countries.
4. Review the adaptive metrics, distribution chart, and matching rows.
5. Ask a question and expand the source records beneath the answer.

Common headers are detected automatically, including:

| Role | Examples |
| --- | --- |
| Review text | `Review`, `Review Text`, `Feedback`, `Comment`, `Body` |
| Title | `Title`, `Review Title`, `Headline`, `Subject` |
| Date | `Date`, `Review Date`, `Published On`, `Created At` |
| Rating | `Rating`, `Stars`, `Score`, `Review Score` |
| Sentiment | `Sentiment`, `Polarity`, `Label` |
| Restaurant | `Restaurant Name`, `Restaurant`, `Venue`, `Business` |
| Country | `Country`, `Nation`, `Market`, `Region` |

For unfamiliar names, select the role manually in the mapping panel. Unmapped columns are preserved as source metadata and included in indexed context. A dataset such as this works without a rating column:

```text
Country,Restaurant Name,Sentiment,Review Title,Review Date,Review
```

Review text cannot be empty. Mapped ratings must be integers from 1 through 5, and supplied mapped dates must be valid. Uploads are limited to 10 MB.

## Terminal commands

Check the dataset, Ollama service, and required models without starting indexing:

```bash
uv run local-ai-agent status
```

Ask one question:

```bash
uv run local-ai-agent ask "What do guests say about the crust?"
```

Start an interactive terminal session:

```bash
uv run local-ai-agent chat
```

Run the measured RAG evaluation against the configured local models:

```bash
uv run local-ai-agent evaluate
```

Write a reproducible machine-readable report and a Markdown summary:

```bash
uv run local-ai-agent evaluate --report-dir evaluation/results/my-run
```

The versioned case manifest is tied to the dataset SHA-256 and uses immutable
content-derived source IDs for gold relevance. The report records model tags
and immutable Ollama digests, dataset and case-set hashes, retrieval limit,
runtime versions, per-case RAG and BM25 rankings, and aggregate metrics.
Retrieval quality is reported as recall@k, hit rate@k, and MRR@k for both
semantic search and the BM25 baseline. Relevance judgments are known-positive,
not exhaustive. The generated report is evidence for this fixed benchmark
configuration, not a claim about general RAG performance.

The original source-tree command remains available for development:

```bash
uv run python main.py
```

Use `--help` on the main command or a subcommand to see model, host, dataset, rating, date, sentiment, restaurant, country, and evidence-limit options. The CLI auto-detects recognized aliases; use the dashboard mapping panel for unfamiliar column names.

## Configuration

The defaults can be changed through command options, dashboard controls, or these environment variables:

| Variable | Default |
| --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `CHAT_MODEL` | `llama3.2` |
| `EMBEDDING_MODEL` | `mxbai-embed-large` |
| `LOCAL_AI_STORAGE_ROOT` | `~/.local/share/local-ai-agent` |

## Storage and indexing

Every normalized review receives a content-derived source ID. The dataset fingerprint is calculated from the sorted record digests, so row reordering does not create a new dataset. The fingerprint automatically derives both the Chroma database directory and collection name under `LOCAL_AI_STORAGE_ROOT`.

Validated uploads are retained under `<storage-root>/uploads/`. Their embedded Chroma databases live under `<storage-root>/chroma/<dataset-fingerprint>/`. Identical content and mapping reuse storage; changed datasets are isolated automatically.

On startup, reconciliation compares desired content IDs with Chroma. New IDs are embedded, changed records receive new IDs, and stale IDs from deleted or changed records are removed. Interrupted embedding runs retry the records that are still missing.

To remove local application data and indexes, stop the application and delete the configured `LOCAL_AI_STORAGE_ROOT`. This deletes local indexes and retained uploads; it does not remove Ollama models.

## Tests and quality checks

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

The suite covers:

- schema alias detection and manual column mapping;
- optional rating, date, sentiment, restaurant, and country fields;
- content-derived IDs and order-independent dataset fingerprints;
- Chroma recovery plus addition, change, and deletion reconciliation;
- automatic database and collection isolation;
- adaptive rating, date, and categorical filtering;
- Ollama health states;
- evidence-alias-to-source citation validation and model abstention;
- the 30-case evaluation set, BM25 baseline, retrieval metrics, and decomposed
  answer, citation, support-proxy, and abstention metrics;
- evaluation-report serialization and credential-safe provenance;
- deterministic upload storage;
- Streamlit rendering without Ollama.

Run the same quality gates used by CI:

```bash
uv run --with coverage==7.13.4 coverage run -m unittest discover -s tests -v
uv run --with coverage==7.13.4 coverage report
uv run --with mypy==1.18.2 mypy
uvx --from ruff==0.16.1 ruff check .
uvx --from ruff==0.16.1 ruff format --check .
uv export --frozen --no-dev --format requirements-txt \
  --no-emit-project -o /tmp/runtime-requirements.txt
uvx --from pip-audit==2.10.1 pip-audit \
  -r /tmp/runtime-requirements.txt --progress-spinner off \
  --ignore-vuln CVE-2026-45829
```

GitHub Actions tests Python 3.11 and Python 3.14 and reports coverage for all
six core Python modules, including `evaluation.py`. On Python 3.11 it also runs
static typing, Ruff, formatting, the runtime-dependency audit, package build,
and an installed-CLI smoke test. Coverage remains visible without an arbitrary
pass threshold.

## Project layout

```text
.
├── agent.py                         # Answer generation and source-ID validation
├── dashboard.py                     # Streamlit review-intelligence dashboard
├── dashboard_support.py             # Validated, content-addressed CSV uploads
├── evaluation.py                    # RAG evaluation runner and metrics
├── main.py                          # Status, ask, chat, and evaluate CLI
├── ollama_health.py                 # Ollama and model preflight checks
├── vector.py                        # Validation, identity, reconciliation, and retrieval
├── local_ai_agent/                  # Installed entry point and packaged data
│   └── data/rag_cases.json          # Curated RAG evaluation set
├── docs/architecture.md             # Architecture and privacy boundaries
├── docs/architecture.svg            # Editable architecture diagram
├── evaluation/results/               # Versioned JSON and Markdown benchmark reports
├── SECURITY.md                       # Reporting policy and scoped risk acceptance
├── tests/                            # Unit, integration, evaluation, and dashboard tests
├── pyproject.toml
└── uv.lock
```

## Dependency audit

ChromaDB is declared directly at the newest PyPI release verified during this update (`1.5.9`). `pip-audit` still reports `PYSEC-2026-311` / `CVE-2026-45829`, and the advisory currently lists no fixed release. It concerns an unauthenticated Chroma HTTP server endpoint that accepts `trust_remote_code`; this application uses embedded Chroma and does not start that server. CI ignores only this finding and fails on every other known vulnerability. The scope and removal conditions are documented in [SECURITY.md](SECURITY.md).

## Current limitations

- Semantic questions and measured evaluations require a reachable Ollama service and both configured models.
- Browser uploads remain limited to 10 MB. Large datasets require a future chunked/local-path ingestion path rather than merely increasing Streamlit's limit.
- Rating, date, sentiment, restaurant, and country filters are applied after semantic ranking. This is suitable for small local datasets but not optimized for very large collections.
- Automatic mapping is conservative. Unfamiliar or ambiguous headers require confirmation in the dashboard.
- Sentiment labels are displayed and filtered as supplied. The application does not infer sentiment when the dataset lacks a sentiment column.
- The reference-term support proxy checks expected terms against answers and cited source text; it is not an LLM judge or proof that every claim is correct.
- Topic modeling, hybrid keyword retrieval, and reranking are not implemented.
- The application does not provide authentication or multi-user isolation and should not be exposed directly as a shared public service.

## License

Licensed under the [MIT License](LICENSE).
