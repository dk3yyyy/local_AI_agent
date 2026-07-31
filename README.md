# 🍕 Local Restaurant Review Intelligence

[![CI](https://github.com/dk3yyyy/local_AI_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/dk3yyyy/local_AI_agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A local review-intelligence application powered by Ollama and ChromaDB. Explore restaurant-review metrics, filter the evidence, upload compatible CSV datasets, and ask grounded questions with cited source reviews.

Review data and model prompts stay on your machine. No hosted model API is required.

## Highlights

- **Visual dashboard:** rating metrics, distribution chart, date and rating filters, and review browser.
- **Grounded answers:** the model receives only retrieved reviews and returns numbered citations.
- **Safe offline state:** analytics still load when Ollama is unavailable, while the app shows exact setup commands instead of crashing.
- **Adaptive CSV upload:** automatically detect common headers, manually map unfamiliar names, and isolate each schema in content-addressed Chroma storage.
- **Incremental indexing:** only missing review IDs are embedded, including recovery after an interrupted first run.
- **Terminal interface:** health, one-shot question, and interactive chat commands use the same tested core.
- **Local execution:** Ollama handles both embeddings and answer generation.

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
uv run python main.py status
```

Ask one question:

```bash
uv run python main.py ask "What do guests say about the crust?"
```

Start an interactive terminal session:

```bash
uv run python main.py chat
```

The original no-argument command remains equivalent to `chat`:

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
| `LOCAL_AI_STORAGE_ROOT` | `.local_data/` inside the project |

## Storage and indexing

The bundled dataset uses `chrome_langchain_db/`. Validated uploads are saved under `.local_data/uploads/`, while their isolated vector collections live under `.local_data/chroma/`.

Both locations are excluded from Git. Uploaded datasets use a SHA-256 hash of the file content and confirmed mapping. Identical files with the same mapping reuse storage, while different files or interpretations cannot overwrite one another.

On startup, indexing compares source review IDs with IDs already stored in Chroma. Missing reviews are embedded and existing reviews are left untouched. If an embedding run fails after the database is created, the next attempt retries the missing reviews.

To rebuild the bundled index, stop the application and remove `chrome_langchain_db/`. To remove uploaded datasets and their indexes, remove `.local_data/`.

## Tests and quality checks

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

The suite covers:

- schema alias detection and manual column mapping;
- optional rating, date, sentiment, restaurant, and country fields;
- Chroma recovery and incremental indexing;
- adaptive rating, date, and categorical filtering;
- Ollama health states;
- grounded prompt and citation construction;
- deterministic upload storage;
- Streamlit rendering without Ollama.

Run lint and formatting checks:

```bash
uvx --from ruff==0.16.1 ruff check .
uvx --from ruff==0.16.1 ruff format --check .
```

GitHub Actions runs the tests on Python 3.11 and Python 3.14.

## Project layout

```text
.
├── agent.py                         # Grounded answer and citation service
├── dashboard.py                     # Streamlit review-intelligence dashboard
├── dashboard_support.py             # Validated, content-addressed CSV uploads
├── main.py                          # Status, ask, and chat CLI
├── ollama_health.py                 # Ollama and model preflight checks
├── vector.py                        # Validation, indexing, metrics, and retrieval
├── realistic_restaurant_reviews.csv
├── .streamlit/config.toml           # Dashboard theme and upload limit
├── tests/                            # Unit, integration, and dashboard tests
├── pyproject.toml
└── uv.lock
```

## Current limitations

- Semantic questions require a live local Ollama service and both configured models.
- Rating, date, sentiment, restaurant, and country filters are applied after semantic ranking. This is suitable for small local datasets but not optimized for very large collections.
- Automatic mapping is conservative. Unfamiliar or ambiguous headers require confirmation in the dashboard.
- Sentiment labels are displayed and filtered as supplied. The application does not infer sentiment when the dataset lacks a sentiment column.
- Topic modeling, hybrid keyword retrieval, reranking, and formal RAG evaluation are not implemented yet.
- The application is intended for local use and does not provide authentication or multi-user isolation.

## License

Licensed under the [MIT License](LICENSE).
