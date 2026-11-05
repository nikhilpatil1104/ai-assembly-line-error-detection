"""Independent assembly-line API. Not a renamed warehouse packing API."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from .detectors import get_detector
from .images import InvalidImage, decode_image, draw_overlay
from .settings import get_layout, settings
from .storage import InspectionStore
from .verification import batch_quality, verify

app = FastAPI(
    title="AI-Based Assembly Line Error Detection API",
    version="1.0.0",
    description="Fixed-camera manufacturing quality-control: part presence, placement regions, batch error rate."
)


def get_store():
    return InspectionStore(settings()["runtime_dir"])


def active_detector():
    try:
        return get_detector()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def inspect_bytes(raw: bytes, filename: str, detector, store: InspectionStore, batch_id=None):
    start = perf_counter()
    config = settings()
    if len(raw) > config["max_image_bytes"]:
        raise HTTPException(status_code=413, detail="Image exceeds configured size limit")
    try:
        image = decode_image(raw)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    layout = get_layout()
    detections = detector.detect(image)
    h, w = image.shape[:2]
    decision = verify(detections, (w, h), layout, config["confidence_threshold"])
    overlay = draw_overlay(image, layout, detections, decision, config["confidence_threshold"])
    inspection_id = uuid4().hex
    result = {
        "inspection_id": inspection_id,
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": Path(filename or "upload.jpg").name,
        "assembly_id": layout["assembly_id"],
        "detector_mode": detector.mode,
        "confidence_note": (
            "Demo segmentation scores are rule-based matches, NOT neural-network confidence"
            if detector.mode == "demo_color" else "Custom-trained YOLO confidence"
        ),
        "processing_error": False,
        **decision,
        "processing_ms": round((perf_counter() - start) * 1000, 2),
        "detections": [d.as_dict() for d in detections],
        "annotated_image_url": f"/inspections/{inspection_id}/annotated",
    }
    store.save(result, overlay)
    return result


def _error_record(filename: str, detector_mode: str, batch_id: str, error: HTTPException):
    return {
        "inspection_id": uuid4().hex,
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": Path(filename or "unknown").name,
        "detector_mode": detector_mode,
        "assembly_id": get_layout()["assembly_id"],
        "status": "Needs Review",
        "reason_codes": ["processing_error"],
        "reasons": [f"Image cannot be inspected ({error.status_code}): {error.detail}"],
        "missing": {}, "misplaced": [], "extra": {}, "unknown": {},
        "uncertain": True, "detected_counts": {}, "detections": [],
        "processing_ms": 0.0, "processing_error": True,
        "confidence_note": "No inference performed",
        "annotated_image_url": None,
    }


@app.get("/")
def root():
    return {"project": "AI-Based Assembly Line Error Detection", "documentation": "/docs",
            "note": "demo_color is not YOLO; real AI inference requires custom weights"}


@app.get("/health")
def health():
    conf = settings()
    return {"status": "ok", "detector_mode": conf["detector_mode"],
            "custom_weights_available": conf["weights"].is_file(),
            "assembly_id": get_layout()["assembly_id"]}


@app.get("/assembly/layout")
def layout():
    return get_layout()


@app.post("/inspect")
def inspect(file: UploadFile = File(...), detector=Depends(active_detector), store=Depends(get_store)):
    raw = file.file.read(settings()["max_image_bytes"] + 1)
    return inspect_bytes(raw, file.filename, detector, store)


@app.post("/batches/inspect")
def inspect_batch(files: list[UploadFile] = File(...), detector=Depends(active_detector),
                  store=Depends(get_store)):
    if not 1 <= len(files) <= 100:
        raise HTTPException(status_code=422, detail="Upload between 1 and 100 images")
    batch_id = uuid4().hex
    results = []
    for file in files:
        try:
            raw = file.file.read(settings()["max_image_bytes"] + 1)
            record = inspect_bytes(raw, file.filename, detector, store, batch_id=batch_id)
        except HTTPException as exc:
            # Store errors too: GET /batches/{batch_id} must preserve correct
            # batch denominator and warning decisions after the initial request.
            record = _error_record(file.filename, detector.mode, batch_id, exc)
            store.save(record, None)
        results.append(record)
    summary = batch_quality([r["status"] for r in results], settings()["batch_warning_threshold"])
    return {"batch_id": batch_id, **summary, "inspections": results}


@app.get("/inspections")
def history(limit: int = 100, store=Depends(get_store)):
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    return store.list(limit)


@app.get("/summary")
def summary(store=Depends(get_store)):
    return store.summary()


@app.get("/batches/{batch_id}")
def batch_results(batch_id: str, store=Depends(get_store)):
    rows = store.batch(batch_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Batch inspections not found")
    return {"batch_id": batch_id,
            **batch_quality([r["status"] for r in rows], settings()["batch_warning_threshold"]),
            "inspections": rows}


@app.get("/inspections/export.csv")
def export_csv(store=Depends(get_store)):
    return Response(content=store.export_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=assembly_inspections.csv"})


@app.get("/inspections/{inspection_id}")
def inspection(inspection_id: str, store=Depends(get_store)):
    row = store.by_id(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return row


@app.get("/inspections/{inspection_id}/annotated")
def annotated(inspection_id: str, store=Depends(get_store)):
    if len(inspection_id) != 32 or any(c not in "0123456789abcdef" for c in inspection_id):
        raise HTTPException(status_code=404, detail="Inspection not found")
    record = store.by_id(inspection_id)
    path = store.overlays / (inspection_id + ".png")
    if record is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Annotated image not found")
    return FileResponse(path, media_type="image/png")
