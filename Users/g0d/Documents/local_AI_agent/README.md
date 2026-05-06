# Local AI Agent

A local AI agent using Ollama embeddings with a vector store (ChromaDB) to process and retrieve restaurant reviews.

## Features

- Uses **Ollama** (`mxbai-embed-large` model) for local embeddings
- **ChromaDB** vector store for persisting document embeddings
- Processes restaurant review data from CSV files
- Retrieves relevant reviews using semantic search

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Ensure Ollama is running locally with the required model:
   ```bash
   ollama pull mxbai-embed-large
   ```

3. Run the application:
   ```bash
   uv run main.py
   ```

## Data

Uses `realistic_restaurant_reviews.csv` containing restaurant reviews with columns: Title, Date, Rating, Review.
