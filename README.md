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
- **CSV upload:** validated datasets receive isolated, content-addressed Chroma storage.
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
2. Choose a rating range and date range.
3. Review the deterministic metrics, rating distribution, and matching rows.
4. Ask a question in the chat box.
5. Expand the evidence citations beneath the answer.

Uploaded files must contain these columns:

```text
Title, Date, Rating, Review
```

Validation rejects missing values, empty text, malformed dates, and ratings outside the integer range 1 through 5. Uploads are limited to 10 MB.

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

Use `--help` on the main command or a subcommand to see model, host, dataset, rating, date, and evidence-limit options.

## Configuration

The defaults can be changed through command options, dashboard controls, or these environment variables:

| Variable | Default |
| --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `CHAT_MODEL` | `llama3.2` |
| `EMBEDDING_MODEL` | `mxbai-embed-large` |

## Storage and indexing

The bundled dataset uses `chrome_langchain_db/`. Validated uploads are saved under `.local_data/uploads/`, while their isolated vector collections live under `.local_data/chroma/`.

Both locations are excluded from Git. Uploaded datasets use a SHA-256 content hash, so identical files reuse the same storage and different datasets cannot overwrite one another.

On startup, indexing compares source review IDs with IDs already stored in Chroma. Missing reviews are embedded and existing reviews are left untouched. If an embedding run fails after the database is created, the next attempt retries the missing reviews.

To rebuild the bundled index, stop the application and remove `chrome_langchain_db/`. To remove uploaded datasets and their indexes, remove `.local_data/`.

## Tests and quality checks

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

The suite covers:

- dataset validation and metrics;
- Chroma recovery and incremental indexing;
- rating and date filtering;
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
- Rating and date filters are applied after semantic ranking. This is suitable for the bundled dataset but not optimized for very large collections.
- Uploaded datasets must follow the documented four-column schema.
- Sentiment classification, topic modeling, hybrid keyword retrieval, reranking, and formal RAG evaluation are not implemented yet.
- The application is intended for local use and does not provide authentication or multi-user isolation.

## License

Licensed under the [MIT License](LICENSE).
