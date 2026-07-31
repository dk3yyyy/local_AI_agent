import json
import os
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

from vector import (
    PROJECT_ROOT,
    ColumnMapping,
    detect_column_mapping,
    load_reviews,
)

DEFAULT_STORAGE_ROOT = PROJECT_ROOT / ".local_data"


@dataclass(frozen=True)
class DatasetSelection:
    csv_path: Path
    database_path: Path
    collection_name: str
    digest: str
    review_count: int
    mapping: ColumnMapping


def read_csv_columns(content: bytes) -> tuple[str, ...]:
    try:
        dataframe = pd.read_csv(BytesIO(content), nrows=0)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(f"Could not read the CSV header: {error}") from error
    columns = tuple(str(column) for column in dataframe.columns)
    if not columns:
        raise ValueError("The CSV does not contain any columns")
    return columns


def prepare_uploaded_dataset(
    content: bytes,
    *,
    mapping: ColumnMapping | None = None,
    storage_root: Path | None = None,
) -> DatasetSelection:
    """Validate an uploaded CSV, then persist it under a schema-aware hash."""
    resolved_mapping = mapping or detect_column_mapping(read_csv_columns(content))
    dataframe = load_reviews(BytesIO(content), mapping=resolved_mapping)
    mapping_bytes = json.dumps(
        resolved_mapping.as_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    digest = sha256(content + b"\0" + mapping_bytes).hexdigest()
    resolved_storage_root = storage_root or Path(
        os.getenv("LOCAL_AI_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
    )
    uploads_directory = resolved_storage_root / "uploads"
    csv_path = uploads_directory / f"{digest}.csv"
    database_path = resolved_storage_root / "chroma" / digest

    if not csv_path.exists():
        uploads_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = csv_path.with_suffix(".tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(csv_path)

    return DatasetSelection(
        csv_path=csv_path,
        database_path=database_path,
        collection_name=f"reviews_{digest[:12]}",
        digest=digest,
        review_count=len(dataframe),
        mapping=resolved_mapping,
    )
