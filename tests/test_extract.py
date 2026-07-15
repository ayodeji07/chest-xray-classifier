"""
tests/test_extract.py — Unit tests for the data download/extraction pipeline.
"""
from __future__ import annotations

import hashlib
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.config import PATHOLOGY_CLASSES, Paths


class TestLoadLabelsDataframe:
    """Tests for load_labels_dataframe() — label string parsing."""

    def _write_csv(self, tmp_path, rows):
        """Write a minimal NIH-format labels CSV and point Paths at it."""
        csv_path = tmp_path / "Data_Entry_2017.csv"
        df = pd.DataFrame(rows, columns=["Image Index", "Finding Labels"])
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_pleural_thickening_matches_underscore_format(self, tmp_path, monkeypatch):
        """Regression test: NIH's Finding Labels column uses
        'Pleural_Thickening' (underscore), matching our class name
        exactly. A previous bug replaced '_' with ' ' before matching,
        which meant this class never matched any real label.
        """
        from src.data.extract import load_labels_dataframe

        csv_path = self._write_csv(tmp_path, [
            {"Image Index": "00000001_000.png", "Finding Labels": "Pleural_Thickening"},
        ])
        monkeypatch.setattr(Paths, "nih_labels_csv", csv_path)

        df = load_labels_dataframe()
        assert df.loc[0, "Pleural_Thickening"] == 1

    def test_pipe_separated_labels_expand_to_multiple_columns(self, tmp_path, monkeypatch):
        from src.data.extract import load_labels_dataframe

        csv_path = self._write_csv(tmp_path, [
            {"Image Index": "00000001_000.png", "Finding Labels": "Cardiomegaly|Effusion"},
        ])
        monkeypatch.setattr(Paths, "nih_labels_csv", csv_path)

        df = load_labels_dataframe()
        assert df.loc[0, "Cardiomegaly"] == 1
        assert df.loc[0, "Effusion"]     == 1
        other_classes = [c for c in PATHOLOGY_CLASSES if c not in ("Cardiomegaly", "Effusion")]
        assert df.loc[0, other_classes].sum() == 0

    def test_no_finding_sets_no_finding_flag_and_no_pathologies(self, tmp_path, monkeypatch):
        from src.data.extract import load_labels_dataframe

        csv_path = self._write_csv(tmp_path, [
            {"Image Index": "00000001_000.png", "Finding Labels": "No Finding"},
        ])
        monkeypatch.setattr(Paths, "nih_labels_csv", csv_path)

        df = load_labels_dataframe()
        assert df.loc[0, "no_finding"] == 1
        assert df.loc[0, PATHOLOGY_CLASSES].sum() == 0

    def test_missing_csv_raises_file_not_found(self, tmp_path, monkeypatch):
        from src.data.extract import load_labels_dataframe

        monkeypatch.setattr(Paths, "nih_labels_csv", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            load_labels_dataframe()


class TestValidateImages:

    def test_filters_to_only_existing_images(self, tmp_path):
        from src.data.extract import validate_images

        (tmp_path / "present.png").write_bytes(b"fake")
        labels_df = pd.DataFrame({
            "image_id": ["present.png", "missing.png"],
        })
        valid = validate_images(tmp_path, labels_df)
        assert list(valid["image_id"]) == ["present.png"]

    def test_all_missing_returns_empty_dataframe(self, tmp_path):
        from src.data.extract import validate_images

        labels_df = pd.DataFrame({"image_id": ["a.png", "b.png"]})
        valid = validate_images(tmp_path, labels_df)
        assert len(valid) == 0


class TestDownloadFile:

    def test_skips_download_when_checksum_matches(self, tmp_path):
        from src.data.extract import _download_file

        dest = tmp_path / "file.bin"
        dest.write_bytes(b"hello world")
        md5 = hashlib.md5(b"hello world").hexdigest()

        with patch("src.data.extract.urllib.request.urlretrieve") as mock_dl:
            _download_file(url="http://example.com/f", dest=dest, verify_md5=md5)
            mock_dl.assert_not_called()

    def test_redownloads_when_checksum_mismatches(self, tmp_path):
        from src.data.extract import _download_file

        dest = tmp_path / "file.bin"
        dest.write_bytes(b"stale content")

        def _fake_urlretrieve(url, filename, reporthook=None):
            with open(filename, "wb") as f:
                f.write(b"fresh content")

        with patch("src.data.extract.urllib.request.urlretrieve", side_effect=_fake_urlretrieve) as mock_dl:
            _download_file(
                url="http://example.com/f", dest=dest,
                verify_md5=hashlib.md5(b"fresh content").hexdigest(),
            )
            mock_dl.assert_called_once()
        assert dest.read_bytes() == b"fresh content"

    def test_downloads_when_file_does_not_exist(self, tmp_path):
        from src.data.extract import _download_file

        dest = tmp_path / "new_file.bin"

        def _fake_urlretrieve(url, filename, reporthook=None):
            with open(filename, "wb") as f:
                f.write(b"content")

        with patch("src.data.extract.urllib.request.urlretrieve", side_effect=_fake_urlretrieve) as mock_dl:
            _download_file(url="http://example.com/f", dest=dest)
            mock_dl.assert_called_once()
        assert dest.exists()


class TestDownloadSubset:

    def _make_fixture_zip(self, tmp_path, n_images=5):
        """Build a fake NIH-mirror zip: real PNGs under images/, plus
        __MACOSX junk entries that must be filtered out on extraction.
        """
        zip_path = tmp_path / "fixture.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("images/", "")
            zf.writestr("__MACOSX/._images", "junk")
            for i in range(n_images):
                name = f"{i:08d}_000.png"
                zf.writestr(f"images/{name}", b"fake png bytes")
                zf.writestr(f"__MACOSX/images/._{name}", "junk")
        return zip_path

    def test_extracts_only_real_pngs_and_respects_target_count(self, tmp_path, monkeypatch):
        from src.data.extract import download_subset
        from src.utils.config import TrainingConfig

        fixture_zip = self._make_fixture_zip(tmp_path, n_images=10)
        dest_dir = tmp_path / "subset_images"
        monkeypatch.setattr(Paths, "subset_images", dest_dir)
        monkeypatch.setattr(Paths, "raw", tmp_path / "raw")
        monkeypatch.setattr(TrainingConfig, "subset_size", 3)

        def _fake_download_file(url, dest, desc=""):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(fixture_zip.read_bytes())
            return dest

        with patch("src.data.extract._download_file", side_effect=_fake_download_file):
            result_dir = download_subset()

        pngs = list(result_dir.glob("*.png"))
        assert len(pngs) == 3
        assert all("MACOSX" not in p.name for p in pngs)


class TestPrepareData:

    def test_kaggle_mode_uses_kaggle_images_dir(self, tmp_path, monkeypatch):
        from src.data.extract import prepare_data
        from src.utils.config import TrainingConfig

        monkeypatch.setattr(TrainingConfig, "mode", "kaggle")
        monkeypatch.setattr(Paths, "nih_labels_csv", tmp_path / "labels.csv")

        with patch("src.data.extract.download_labels_csv"), \
             patch("src.data.extract.load_labels_dataframe", return_value=pd.DataFrame({"image_id": []})), \
             patch("src.data.extract.validate_images", side_effect=lambda d, df: df):
            labels_df, image_dir = prepare_data()

        assert image_dir == Paths.kaggle_images
