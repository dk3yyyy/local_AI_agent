import tempfile
import unittest
from pathlib import Path

from dashboard_support import prepare_uploaded_dataset
from vector import ReviewDataError

VALID_CSV = b"""Title,Date,Rating,Review
Great crust,2024-03-01,5,Crisp and flavorful
"""


class DashboardUploadTest(unittest.TestCase):
    def test_upload_uses_content_hash_for_stable_isolated_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = prepare_uploaded_dataset(VALID_CSV, storage_root=root)
            second = prepare_uploaded_dataset(VALID_CSV, storage_root=root)

            self.assertEqual(first, second)
            self.assertTrue(first.csv_path.is_file())
            self.assertEqual(first.review_count, 1)
            self.assertTrue(first.collection_name.startswith("reviews_"))
            self.assertEqual(first.database_path.parent, root / "chroma")

    def test_invalid_upload_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            with self.assertRaises(ReviewDataError):
                prepare_uploaded_dataset(b"wrong,columns\n1,2\n", storage_root=root)

            self.assertFalse((root / "uploads").exists())


if __name__ == "__main__":
    unittest.main()
