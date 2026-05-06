# Local AI Agent

A Python-based AI agent that processes and analyzes restaurant reviews using vector embeddings and local language models.

## Features

- Vector-based review analysis
- Local AI processing (no external API dependencies)
- Restaurant review sentiment analysis
- ChromaDB integration for vector storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dk3yyyy/local_AI_agent.git
cd local_AI_agent
```

2. Install dependencies using uv:
```bash
uv sync
```

## Usage

Run the main agent:
```bash
uv run python main.py
```

## Data

The project includes a dataset of realistic restaurant reviews (`realistic_restaurant_reviews.csv`) for testing and demonstration purposes.

## Requirements

- Python 3.8+
- uv package manager

## License

MIT License