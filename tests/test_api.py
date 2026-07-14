"""
tests/test_api.py — Integration tests for the FastAPI app.
"""
from __future__ import annotations
import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=True)


def _make_image_bytes() -> bytes:
    """Create a minimal valid PNG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestHealth:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]  == "ok"
        assert "version" in data
        assert "model"   in data
        assert "device"  in data

    def test_root_redirects(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)


class TestPredict:

    def _mock_model(self):
        from src.utils.config import PATHOLOGY_CLASSES
        m = MagicMock()
        m.backbone_name = "densenet121"
        m.predict_single.return_value = {
            cls: 0.1 for cls in PATHOLOGY_CLASSES
        }
        m.get_features_layer.return_value = MagicMock()
        return m

    def test_predict_returns_expected_shape(self, client):
        img_bytes = _make_image_bytes()
        with patch("src.api.routes.predict._get_model",
                   return_value=(self._mock_model(), "cpu")):
            resp = client.post(
                "/predict",
                files={"file": ("xray.png", img_bytes, "image/png")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions"    in data
        assert "processing_ms"  in data
        assert "model_backbone" in data
        assert "disclaimer"     in data

    def test_predict_no_file_returns_422(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_predict_invalid_file_returns_400(self, client):
        with patch("src.api.routes.predict._get_model",
                   return_value=(self._mock_model(), "cpu")):
            resp = client.post(
                "/predict",
                files={"file": ("bad.png", b"not an image", "image/png")},
            )
        assert resp.status_code == 400
