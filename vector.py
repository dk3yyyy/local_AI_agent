import json
import os
import re
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import IO, Any, cast

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = Path(
    str(files("local_ai_agent.data").joinpath("realistic_restaurant_reviews.csv"))
)
DEFAULT_STORAGE_ROOT = Path(
    os.getenv(
        "LOCAL_AI_STORAGE_ROOT",
        str(Path.home() / ".local" / "share" / "local-ai-agent"),
    )
)
DEFAULT_DATABASE_PATH = DEFAULT_STORAGE_ROOT / "chroma"
DEFAULT_COLLECTION_NAME = "restaurant_reviews"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
CANONICAL_COLUMNS = (
    "Title",
    "Date",
    "Rating",
    "Sentiment",
    "Restaurant",
    "Country",
    "Review",
)
ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "review": (
        "review",
        "reviewtext",
        "customerreview",
        "feedback",
        "customerfeedback",
        "comment",
        "comments",
        "body",
        "content",
    ),
    "title": ("title", "reviewtitle", "headline", "subject"),
    "date": (
        "date",
        "reviewdate",
        "createdat",
        "publishedat",
        "publishedon",
        "timestamp",
        "submissiondate",
    ),
    "rating": ("rating", "stars", "star", "score", "reviewscore"),
    "sentiment": ("sentiment", "polarity", "sentimentlabel", "label"),
    "restaurant": (
        "restaurant",
        "restaurantname",
        "venue",
        "business",
        "businessname",
        "locationname",
    ),
    "country": ("country", "nation", "market", "region"),
}
ROLE_TO_CANONICAL = {
    "review": "Review",
    "title": "Title",
    "date": "Date",
    "rating": "Rating",
    "sentiment": "Sentiment",
    "restaurant": "Restaurant",
    "country": "Country",
}

ReviewSource = str | Path | IO[str] | IO[bytes] | pd.DataFrame


class ReviewDataError(ValueError):
    """Raised when a review dataset cannot be indexed safely."""


@dataclass(frozen=True)
class ColumnMapping:
    review: str
    title: str | None = None
    date: str | None = None
    rating: str | None = None
    sentiment: str | None = None
    restaurant: str | None = None
    country: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {role: getattr(self, role) for role in ROLE_TO_CANONICAL}


@dataclass(frozen=True)
class ReviewSummary:
    total_reviews: int
    average_rating: float | None
    high_rated: int
    low_rated: int
    rating_counts: dict[int, int]
    sentiment_counts: dict[str, int]
    first_date: date | None
    last_date: date | None
    restaurant_count: int
    country_count: int


@dataclass(frozen=True)
class ReviewMatch:
    document: Document
    score: float


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _alias_score(column: str, aliases: tuple[str, ...]) -> int:
    normalized = _normalize_column_name(column)
    return max(
        (1000 + len(alias) for alias in aliases if normalized == alias),
        default=0,
    )


def suggest_column_mapping(columns: Any) -> dict[str, str | None]:
    """Suggest semantic roles without guessing when no useful alias exists."""
    available = [str(column) for column in columns]
    suggestions: dict[str, str | None] = {}
    claimed: set[str] = set()
    for role, aliases in ROLE_ALIASES.items():
        ranked = sorted(
            (
                (_alias_score(column, aliases), -index, column)
                for index, column in enumerate(available)
                if column not in claimed
            ),
            reverse=True,
        )
        score, _, column = ranked[0] if ranked else (0, 0, None)
        suggestions[role] = column if score >= 500 else None
        if score >= 500 and column is not None:
            claimed.add(column)
    return suggestions


def detect_column_mapping(columns: Any) -> ColumnMapping:
    suggestions = suggest_column_mapping(columns)
    review_column = suggestions["review"]
    if review_column is None:
        raise ReviewDataError(
            "Could not detect a review text column. Select one in the column mapping."
        )
    return ColumnMapping(
        review=review_column,
        title=suggestions["title"],
        date=suggestions["date"],
        rating=suggestions["rating"],
        sentiment=suggestions["sentiment"],
        restaurant=suggestions["restaurant"],
        country=suggestions["country"],
    )


def _read_dataframe(source: ReviewSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return cast(pd.DataFrame, source).copy()
    return pd.read_csv(source)


def _validate_mapping(mapping: ColumnMapping, columns: Any) -> None:
    available = {str(column) for column in columns}
    selected = [column for column in mapping.as_dict().values() if column is not None]
    missing = [column for column in selected if column not in available]
    if missing:
        raise ReviewDataError(f"Mapped columns do not exist: {', '.join(missing)}")
    duplicates = sorted({column for column in selected if selected.count(column) > 1})
    if duplicates:
        raise ReviewDataError(
            "A source column cannot be assigned to more than one role: "
            + ", ".join(duplicates)
        )


def _optional_text(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values == "", pd.NA)


def load_reviews(
    source: ReviewSource = DEFAULT_DATA_PATH,
    *,
    mapping: ColumnMapping | None = None,
) -> pd.DataFrame:
    """Map a review dataset into a stable internal schema."""
    dataframe = _read_dataframe(source)
    if set(CANONICAL_COLUMNS).issubset(dataframe.columns) and "_extra" in dataframe:
        return dataframe.reset_index(drop=True)

    resolved_mapping = mapping or detect_column_mapping(dataframe.columns)
    _validate_mapping(resolved_mapping, dataframe.columns)

    review_values = _optional_text(dataframe[resolved_mapping.review])
    invalid_reviews = review_values.isna()
    if invalid_reviews.any():
        rows = ", ".join(str(index) for index in dataframe.index[invalid_reviews])
        raise ReviewDataError(f"Review cannot be empty; invalid rows: {rows}")

    normalized = pd.DataFrame(index=dataframe.index)
    normalized["Review"] = review_values

    if resolved_mapping.title:
        normalized["Title"] = _optional_text(dataframe[resolved_mapping.title])
    else:
        normalized["Title"] = pd.Series(pd.NA, index=dataframe.index, dtype="string")

    if resolved_mapping.rating:
        raw_ratings = dataframe[resolved_mapping.rating]
        ratings = pd.to_numeric(raw_ratings, errors="coerce")
        supplied = raw_ratings.notna() & (
            raw_ratings.astype("string").str.strip() != ""
        )
        invalid_ratings = supplied & (
            ratings.isna() | (ratings % 1 != 0) | ~ratings.between(1, 5)
        )
        if invalid_ratings.any():
            rows = ", ".join(str(index) for index in dataframe.index[invalid_ratings])
            raise ReviewDataError(
                f"Rating must be an integer from 1 to 5; invalid rows: {rows}"
            )
        normalized["Rating"] = ratings.astype("Int64")
    else:
        normalized["Rating"] = pd.Series(pd.NA, index=dataframe.index, dtype="Int64")

    if resolved_mapping.date:
        raw_dates = dataframe[resolved_mapping.date]
        dates = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
        supplied = raw_dates.notna() & (raw_dates.astype("string").str.strip() != "")
        invalid_dates = supplied & dates.isna()
        if invalid_dates.any():
            rows = ", ".join(str(index) for index in dataframe.index[invalid_dates])
            raise ReviewDataError(f"Date must be a valid date; invalid rows: {rows}")
        normalized["Date"] = dates.dt.strftime("%Y-%m-%d").astype("string")
    else:
        normalized["Date"] = pd.Series(pd.NA, index=dataframe.index, dtype="string")

    for role in ("sentiment", "restaurant", "country"):
        canonical = ROLE_TO_CANONICAL[role]
        source_column = getattr(resolved_mapping, role)
        if source_column:
            normalized[canonical] = _optional_text(dataframe[source_column])
        else:
            normalized[canonical] = pd.Series(
                pd.NA, index=dataframe.index, dtype="string"
            )

    normalized = normalized.loc[:, CANONICAL_COLUMNS]
    used_columns = {
        column for column in resolved_mapping.as_dict().values() if column is not None
    }
    extra_columns = [
        column for column in dataframe.columns if column not in used_columns
    ]
    normalized["_extra"] = [
        {
            str(column): str(dataframe.at[index, column])
            for column in extra_columns
            if pd.notna(dataframe.at[index, column])
            and str(dataframe.at[index, column]).strip()
        }
        for index in dataframe.index
    ]
    normalized.attrs["column_mapping"] = resolved_mapping.as_dict()
    return normalized.reset_index(drop=True)


def dataset_summary(dataframe: pd.DataFrame) -> ReviewSummary:
    """Calculate metrics from whichever semantic fields are available."""
    normalized = load_reviews(dataframe)
    ratings = normalized["Rating"].dropna().astype(int)
    rating_values = ratings.value_counts().to_dict()
    rating_counts = {
        rating: int(rating_values.get(rating, 0)) for rating in range(1, 6)
    }
    sentiments = normalized["Sentiment"].dropna().astype(str)
    sentiment_values = sentiments.value_counts().sort_index().to_dict()
    dates = pd.to_datetime(normalized["Date"], errors="coerce").dropna()
    return ReviewSummary(
        total_reviews=len(normalized),
        average_rating=float(ratings.mean()) if len(ratings) else None,
        high_rated=int((ratings >= 4).sum()),
        low_rated=int((ratings <= 2).sum()),
        rating_counts=rating_counts,
        sentiment_counts={
            str(key): int(value) for key, value in sentiment_values.items()
        },
        first_date=dates.min().date() if len(dates) else None,
        last_date=dates.max().date() if len(dates) else None,
        restaurant_count=int(normalized["Restaurant"].nunique(dropna=True)),
        country_count=int(normalized["Country"].nunique(dropna=True)),
    )


def _casefold_values(values: tuple[str, ...] | list[str]) -> set[str]:
    return {value.casefold() for value in values}


def filter_reviews(
    dataframe: pd.DataFrame,
    *,
    min_rating: int | None = None,
    max_rating: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    sentiments: tuple[str, ...] | list[str] = (),
    restaurants: tuple[str, ...] | list[str] = (),
    countries: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Filter normalized reviews using only requested semantic fields."""
    normalized = load_reviews(dataframe)
    if min_rating is not None and max_rating is not None and min_rating > max_rating:
        raise ValueError("min_rating cannot exceed max_rating")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    mask = pd.Series(True, index=normalized.index)
    if min_rating is not None or max_rating is not None:
        ratings = normalized["Rating"]
        mask &= ratings.notna()
        if min_rating is not None:
            mask &= ratings >= min_rating
        if max_rating is not None:
            mask &= ratings <= max_rating
    if start_date is not None or end_date is not None:
        dates = pd.to_datetime(normalized["Date"], errors="coerce")
        mask &= dates.notna()
        if start_date is not None:
            mask &= dates >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= dates <= pd.Timestamp(end_date)

    for canonical, selected in (
        ("Sentiment", sentiments),
        ("Restaurant", restaurants),
        ("Country", countries),
    ):
        if selected:
            allowed = _casefold_values(selected)
            mask &= normalized[canonical].astype("string").str.casefold().isin(allowed)
    return normalized.loc[mask].reset_index(drop=True)


def _metadata_value(value: Any) -> str | int | float | bool | None:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _record_payload(row: pd.Series) -> str:
    canonical = {column: _metadata_value(row[column]) for column in CANONICAL_COLUMNS}
    extras = sorted((str(key), str(value)) for key, value in row["_extra"].items())
    return json.dumps(
        {"canonical": canonical, "extras": extras},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_digest(row: pd.Series) -> str:
    return sha256(_record_payload(row).encode("utf-8")).hexdigest()


def dataset_fingerprint(dataframe: pd.DataFrame) -> str:
    """Return an order-insensitive digest of normalized review content."""
    required = set(CANONICAL_COLUMNS) | {"_extra"}
    if (
        required <= set(dataframe.columns)
        and dataframe["_extra"].map(lambda value: isinstance(value, dict)).all()
    ):
        normalized = dataframe
    else:
        normalized = load_reviews(dataframe)
    record_digests = sorted(_record_digest(row) for _, row in normalized.iterrows())
    payload = json.dumps(record_digests, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def dataset_storage(
    dataframe: pd.DataFrame,
    *,
    storage_root: str | Path | None = None,
) -> tuple[Path, str]:
    """Derive an isolated persistence path and collection from dataset content."""
    digest = dataset_fingerprint(dataframe)
    root = Path(storage_root) if storage_root is not None else DEFAULT_STORAGE_ROOT
    return root / "chroma" / digest, f"reviews_{digest[:16]}"


def _documents_and_ids(dataframe: pd.DataFrame) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    occurrences: dict[str, int] = {}
    metadata_fields = {
        "Title": "title",
        "Date": "date",
        "Rating": "rating",
        "Sentiment": "sentiment",
        "Restaurant": "restaurant",
        "Country": "country",
    }
    for _, row in dataframe.iterrows():
        digest = _record_digest(row)
        occurrence = occurrences.get(digest, 0) + 1
        occurrences[digest] = occurrence
        suffix = f"_{occurrence}" if occurrence > 1 else ""
        item_id = f"review_{digest[:32]}{suffix}"
        metadata: dict[str, str | int | float | bool] = {"source_id": item_id}
        content_lines: list[str] = []
        for canonical, key in metadata_fields.items():
            value = _metadata_value(row[canonical])
            if value is not None:
                metadata[key] = value
                if canonical in {"Title", "Sentiment", "Restaurant", "Country"}:
                    content_lines.append(f"{canonical}: {value}")
        for source_column, raw_value in row["_extra"].items():
            base_key = f"extra_{_normalize_column_name(source_column)}"
            if base_key == "extra_":
                continue
            safe_key = base_key
            suffix = 2
            while safe_key in metadata:
                safe_key = f"{base_key}_{suffix}"
                suffix += 1
            metadata[safe_key] = str(raw_value)
            content_lines.append(f"{source_column}: {raw_value}")
        content_lines.append(f"Review: {row['Review']}")
        documents.append(
            Document(
                page_content="\n".join(content_lines),
                metadata=metadata,
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
    mapping: ColumnMapping | None = None,
    database_path: str | Path | None = None,
    collection_name: str | None = None,
    embeddings: Any | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_host: str | None = None,
) -> Chroma:
    """Open an isolated Chroma collection and reconcile it with the source."""
    dataframe = load_reviews(source, mapping=mapping)
    derived_database, derived_collection = dataset_storage(dataframe)
    resolved_database = Path(database_path) if database_path else derived_database
    resolved_collection = collection_name or derived_collection
    documents, ids = _documents_and_ids(dataframe)
    embedding_function = embeddings or create_embeddings(
        model=embedding_model,
        base_url=ollama_host,
    )
    vector_store = Chroma(
        collection_name=resolved_collection,
        persist_directory=str(resolved_database),
        embedding_function=embedding_function,
    )

    desired_ids = set(ids)
    existing_ids = set(vector_store.get(include=[])["ids"])
    stale_ids = sorted(existing_ids - desired_ids)
    if stale_ids:
        vector_store.delete(ids=stale_ids)

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


def _matches_category(
    metadata: dict[str, Any], key: str, selected: tuple[str, ...] | list[str]
) -> bool:
    if not selected:
        return True
    value = metadata.get(key)
    return value is not None and str(value).casefold() in _casefold_values(selected)


def search_reviews(
    vector_store: Any,
    query: str,
    *,
    limit: int = 5,
    min_rating: int | None = None,
    max_rating: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    sentiments: tuple[str, ...] | list[str] = (),
    restaurants: tuple[str, ...] | list[str] = (),
    countries: tuple[str, ...] | list[str] = (),
) -> list[ReviewMatch]:
    """Search semantically, then apply filters supported by the dataset."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if min_rating is not None and max_rating is not None and min_rating > max_rating:
        raise ValueError("min_rating cannot exceed max_rating")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    total = index_count(vector_store)
    if total == 0:
        return []

    results = vector_store.similarity_search_with_score(query, k=total)
    matches: list[ReviewMatch] = []
    for document, score in results:
        metadata = document.metadata
        if min_rating is not None or max_rating is not None:
            try:
                rating = int(metadata["rating"])
            except (KeyError, TypeError, ValueError):
                continue
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue
        if start_date is not None or end_date is not None:
            try:
                review_date = date.fromisoformat(str(metadata["date"]))
            except (KeyError, ValueError):
                continue
            if start_date is not None and review_date < start_date:
                continue
            if end_date is not None and review_date > end_date:
                continue
        if not _matches_category(metadata, "sentiment", sentiments):
            continue
        if not _matches_category(metadata, "restaurant", restaurants):
            continue
        if not _matches_category(metadata, "country", countries):
            continue
        matches.append(ReviewMatch(document=document, score=float(score)))
        if len(matches) == limit:
            break
    return matches
