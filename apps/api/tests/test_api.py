from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def png_bytes(color=(35, 90, 150)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (96, 72), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_mock_and_device():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["device"] in {"cpu", "mps", "cuda"}


def test_single_analysis_contract():
    response = client.post(
        "/api/v1/analyze",
        data={"query": "Highlight the water body", "input_mode": "single"},
        files=[("files", ("scene.png", png_bytes(), "image/png"))],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task"] == "REGION_GROUNDING"
    assert payload["evidence"]
    assert payload["execution_trace"]["status"] == "success"
    assert payload["development_label"].startswith("Development Mock Result")


def test_change_and_cross_modal_routes():
    change = client.post(
        "/api/v1/analyze",
        data={"query": "What changed?", "input_mode": "bi_temporal"},
        files=[
            ("files", ("t1.png", png_bytes((35, 100, 55)), "image/png")),
            ("files", ("t2.png", png_bytes((150, 150, 145)), "image/png")),
        ],
    )
    assert change.status_code == 200
    assert change.json()["task"] == "CHANGE_DESCRIPTION"
    assert "changed_area_percent" in change.json()["statistics"]

    fusion = client.post(
        "/api/v1/analyze",
        data={"query": "Use both sensors to identify water", "input_mode": "cross_modal"},
        files=[
            ("files", ("optical.png", png_bytes((35, 90, 150)), "image/png")),
            ("files", ("sar.png", png_bytes((30, 30, 30)), "image/png")),
        ],
    )
    assert fusion.status_code == 200
    assert fusion.json()["task"] == "OPTICAL_SAR_WATER"
    assert len(fusion.json()["evidence"]) == 3
