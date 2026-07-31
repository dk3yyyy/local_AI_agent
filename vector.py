from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import IO, Any, cast

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "realistic_restaurant_reviews.csv"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "chrome_langchain_db"
DEFAULT_COLLECTION_NAME = "restaurant_reviews"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
REQUIRED_COLUMNS = ("Title", "Date", "Rating", "Review")

ReviewSource = str | Path | IO[str] | IO[bytes] | pd.DataFrame


class ReviewDataError(ValueError):
    """Raised when a review dataset cannot be indexed safely."""


@dataclass(frozen=True)
class ReviewSummary:
    total_reviews: int
    average_rating: float
    high_rated: int
    low_rated: int
    rating_counts: dict[int, int]
    first_date: date
    last_date: date


@dataclass(frozen=True)
class ReviewMatch:
    document: Document
    score: float


def load_reviews(source: ReviewSource = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and normalize a restaurant review dataset."""
    if isinstance(source, pd.DataFrame):
        dataframe = cast(pd.DataFrame, source).copy()
    else:
        dataframe = pd.read_csv(source)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ReviewDataError(f"Dataset is missing required columns: {missing}")

    dataframe = dataframe.loc[:, REQUIRED_COLUMNS].copy()
    missing_values = dataframe[list(REQUIRED_COLUMNS)].isna().any()
    missing_value_columns = missing_values[missing_values].index.tolist()
    if missing_value_columns:
        columns = ", ".join(missing_value_columns)
        raise ReviewDataError(f"Dataset contains missing values in: {columns}")

    ratings = pd.to_numeric(dataframe["Rating"], errors="coerce")
    invalid_ratings = ratings.isna() | (ratings % 1 != 0) | ~ratings.between(1, 5)
    if invalid_ratings.any():
        rows = ", ".join(str(index) for index in dataframe.index[invalid_ratings])
        raise ReviewDataError(
            f"Rating must be an integer from 1 to 5; invalid rows: {rows}"
        )

    dates = pd.to_datetime(dataframe["Date"], errors="coerce")
    if dates.isna().any():
        rows = ", ".join(str(index) for index in dataframe.index[dates.isna()])
        raise ReviewDataError(f"Date must be a valid date; invalid rows: {rows}")

    for column in ("Title", "Review"):
        dataframe[column] = dataframe[column].astype(str).str.strip()
        empty_rows = dataframe.index[dataframe[column] == ""]
        if len(empty_rows):
            rows = ", ".join(str(index) for index in empty_rows)
            raise ReviewDataError(f"{column} cannot be empty; invalid rows: {rows}")

    dataframe["Rating"] = ratings.astype(int)
    dataframe["Date"] = dates.dt.strftime("%Y-%m-%d")
    return dataframe.reset_index(drop=True)


def dataset_summary(dataframe: pd.DataFrame) -> ReviewSummary:
    """Calculate deterministic review metrics for the dashboard."""
    normalized = load_reviews(dataframe)
    dates = pd.to_datetime(normalized["Date"])
    counts = normalized["Rating"].value_counts().to_dict()
    rating_counts = {rating: int(counts.get(rating, 0)) for rating in range(1, 6)}
    return ReviewSummary(
        total_reviews=len(normalized),
        average_rating=float(normalized["Rating"].mean()),
        high_rated=int((normalized["Rating"] >= 4).sum()),
        low_rated=int((normalized["Rating"] <= 2).sum()),
        rating_counts=rating_counts,
        first_date=dates.min().date(),
        last_date=dates.max().date(),
    )


def filter_reviews(
    dataframe: pd.DataFrame,
    *,
    min_rating: int = 1,
    max_rating: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Filter a normalized review dataframe for dashboard analytics."""
    normalized = load_reviews(dataframe)
    dates = pd.to_datetime(normalized["Date"]).dt.date
    mask = normalized["Rating"].between(min_rating, max_rating)
    if start_date is not None:
        mask &= dates >= start_date
    if end_date is not None:
        mask &= dates <= end_date
    return normalized.loc[mask].reset_index(drop=True)


def _documents_and_ids(dataframe: pd.DataFrame) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    for index, row in dataframe.iterrows():
        item_id = str(index)
        documents.append(
            Document(
                page_content=f"{row['Title']}\n{row['Review']}",
                metadata={
                    "title": row["Title"],
                    "rating": int(row["Rating"]),
                    "date": row["Date"],
                },
                id=item_id,
            )
        )
        ids.append(item_id)
    return documents, ids


def create_embeddings(
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str | None = None,
) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model, base_url=base_url)


def create_vector_store(
    source: ReviewSource = DEFAULT_DATA_PATH,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings: Any | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_host: str | None = None,
) -> Chroma:
    """Open a Chroma collection and index only reviews that are missing."""
    dataframe = load_reviews(source)
    documents, ids = _documents_and_ids(dataframe)
    embedding_function = embeddings or create_embeddings(
        model=embedding_model,
        base_url=ollama_host,
    )
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(Path(database_path)),
        embedding_function=embedding_function,
    )

    existing_ids = set(vector_store.get(ids=ids, include=[])["ids"])
    missing_indexes = [
        index for index, item_id in enumerate(ids) if item_id not in existing_ids
    ]
    if missing_indexes:
        vector_store.add_documents(
            documents=[documents[index] for index in missing_indexes],
            ids=[ids[index] for index in missing_indexes],
        )
    return vector_store


def index_count(vector_store: Any) -> int:
    return len(vector_store.get(include=[])["ids"])


def search_reviews(
    vector_store: Any,
    query: str,
    *,
    limit: int = 5,
    min_rating: int = 1,
    max_rating: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[ReviewMatch]:
    """Search semantically, then apply exact rating and date filters."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not 1 <= min_rating <= max_rating <= 5:
        raise ValueError("ratings must satisfy 1 <= min_rating <= max_rating <= 5")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    total = index_count(vector_store)
    if total == 0:
        return []

    results = vector_store.similarity_search_with_score(query, k=total)
    matches: list[ReviewMatch] = []
    for document, score in results:
        rating = int(document.metadata.get("rating", 0))
        try:
            review_date = date.fromisoformat(str(document.metadata.get("date", "")))
        except ValueError:
            continue
        if not min_rating <= rating <= max_rating:
            continue
        if start_date is not None and review_date < start_date:
            continue
        if end_date is not None and review_date > end_date:
            continue
        matches.append(ReviewMatch(document=document, score=float(score)))
        if len(matches) == limit:
            break
    return matches
