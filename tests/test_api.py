import cv2
import pytest
from fastapi.testclient import TestClient

from assembly.api import app, active_detector, get_store
from assembly.detectors import ColorDemoDetector
from assembly.storage import InspectionStore
from demo.generate_samples import panel_image


@pytest.fixture
def client(tmp_path):
    store = InspectionStore(tmp_path)
    app.dependency_overrides[active_detector] = lambda: ColorDemoDetector()
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as instance:
        yield instance
    app.dependency_overrides.clear()


def png(missing=(), moves=None):
    img, _ = panel_image(missing=missing, moves=moves)
    ok, image = cv2.imencode(".png", img)
    assert ok
    return image.tobytes()


def test_health_and_layout(client):
    assert client.get("/health").status_code == 200
    result = client.get("/assembly/layout").json()
    assert len(result["components"]) == 3


def test_demo_image_pass(client):
    response = client.post("/inspect", files={"file": ("good.png", png(), "image/png")})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "Pass", result
    assert result["detector_mode"] == "demo_color"
    annotated = client.get(result["annotated_image_url"])
    assert annotated.status_code == 200
    assert annotated.headers["content-type"].startswith("image/png")
    stored = client.get(f"/inspections/{result['inspection_id']}")
    assert stored.status_code == 200


def test_missing_blue_component(client):
    response = client.post("/inspect", files={"file": ("missing.png", png(missing=["blue_connector"]), "image/png")})
    assert response.status_code == 200
    assert response.json()["missing"] == {"blue_connector": 1}


def test_yellow_component_out_of_zone(client):
    response = client.post("/inspect", files={"file": (
        "wrong.png", png(moves={"yellow_latch": (580, 315)}), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "Needs Review"
    assert "misplacement" in result["reason_codes"]


def test_invalid_image_is_not_accepted(client):
    response = client.post("/inspect", files={"file": ("not_image.png", b"some text", "image/png")})
    assert response.status_code == 422


def test_batch_warning_retrieval_and_csv(client):
    uploads = [("files", (f"pass{i}.png", png(), "image/png")) for i in range(8)]
    uploads += [("files", ("missing.png", png(missing=["blue_connector"]), "image/png")),
                ("files", ("wrong.png", png(moves={"red_switch": (350, 315)}), "image/png"))]
    response = client.post("/batches/inspect", files=uploads)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["flagged_units"] == 2
    assert result["batch_quality_warning"] is True
    retrieved = client.get("/batches/" + result["batch_id"]).json()
    assert retrieved["flagged_units"] == 2
    assert retrieved["batch_quality_warning"] is True
    assert client.get("/summary").json()["inspected_units"] == 10
    exported = client.get("/inspections/export.csv")
    assert exported.status_code == 200
    assert "missing.png" in exported.text


def test_invalid_batch_image_still_counts_in_history(client):
    files = [("files", ("good.png", png(), "image/png")),
             ("files", ("broken.png", b"bad", "image/png"))]
    response = client.post("/batches/inspect", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["flagged_units"] == 1
    assert payload["inspections"][1]["processing_error"] is True
    fresh = client.get("/batches/" + payload["batch_id"]).json()
    assert fresh["flagged_units"] == 1
    assert fresh["inspections"][1]["annotated_image_url"] is None
