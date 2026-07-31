# 🍕 Local AI Agent

[![CI](https://github.com/dk3yyyy/local_AI_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/dk3yyyy/local_AI_agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A local retrieval-augmented generation application for asking questions about restaurant reviews. It uses Ollama for local embeddings and language generation, with ChromaDB for persistent vector search.

## What it does

- Indexes the included restaurant-review dataset into a persistent Chroma collection.
- Retrieves relevant reviews for each question.
- Uses a local Ollama model to answer from the retrieved context.
- Recovers from interrupted or partially completed indexing.
- Adds only reviews that are missing from the persisted collection.
- Runs locally without sending review data to a hosted model API.

## Requirements

- [Python 3.11 or newer](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download)
- Enough local memory and disk space for the selected Ollama models

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

Keep that process running in a separate terminal.

### 3. Download the required models

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2
```

| Model | Purpose |
| --- | --- |
| `mxbai-embed-large` | Creates embeddings for review retrieval |
| `llama3.2` | Generates answers from retrieved reviews |

Model downloads can be several gigabytes. The application does not download them automatically.

### 4. Run the agent

```bash
uv run python main.py
```

Enter a question at the prompt. Type `q` to quit.

Example questions:

- What do customers like about the pizza crust?
- What complaints appear in low-rated reviews?
- How do customers describe the service?

## How indexing works

The application stores vectors in `chrome_langchain_db/`, which is excluded from Git. On startup it compares the review IDs in the CSV with the IDs already stored in Chroma and embeds only missing reviews.

If indexing is interrupted, the next startup retries the missing reviews instead of treating the database directory as a completed index.

To rebuild the index from scratch, stop the application and remove `chrome_langchain_db/` locally. The next run will recreate it.

## Tests and quality checks

Run the test suite:

```bash
uv run python -m unittest discover -s tests -v
```

Run lint and formatting checks:

```bash
uvx --from ruff==0.16.1 ruff check .
uvx --from ruff==0.16.1 ruff format --check .
```

The tests use deterministic fake embeddings and temporary Chroma storage. Ollama is not required for the automated test suite.

## Project layout

```text
.
├── main.py                              # Interactive question-and-answer loop
├── vector.py                            # Review loading, indexing, and retrieval
├── realistic_restaurant_reviews.csv    # Demonstration dataset
├── tests/test_vector.py                 # Index recovery and path regressions
├── pyproject.toml                       # Python package metadata and dependencies
└── uv.lock                              # Reproducible dependency lockfile
```

## Current limitations

- The included interface is an interactive terminal loop.
- The dataset schema is currently fixed to `Title`, `Date`, `Rating`, and `Review`.
- Sentiment classification, source citations, metadata filters, and evaluation metrics are not implemented yet.
- A live Ollama service and both documented models are required for interactive answers.

## License

Licensed under the [MIT License](LICENSE).
