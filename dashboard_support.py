from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from vector import PROJECT_ROOT, load_reviews

DEFAULT_STORAGE_ROOT = PROJECT_ROOT / ".local_data"


@dataclass(frozen=True)
class DatasetSelection:
    csv_path: Path
    database_path: Path
    collection_name: str
    digest: str
    review_count: int


def prepare_uploaded_dataset(
    content: bytes,
    *,
    storage_root: Path = DEFAULT_STORAGE_ROOT,
) -> DatasetSelection:
    """Validate an uploaded CSV, then persist it under a content-derived path."""
    dataframe = load_reviews(BytesIO(content))
    digest = sha256(content).hexdigest()
    uploads_directory = storage_root / "uploads"
    csv_path = uploads_directory / f"{digest}.csv"
    database_path = storage_root / "chroma" / digest

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
    )
