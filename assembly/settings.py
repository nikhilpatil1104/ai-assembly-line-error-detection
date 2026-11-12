"""Paths, checked thresholds, and assembly configuration."""
from functools import lru_cache
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_PATH = ROOT / "config" / "layout.json"


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def settings() -> dict:
    # Optional convenience: `Copy-Item .env.example .env` on Windows, then edit.
    # Loading without override respects variables explicitly set by the shell.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    conf = {
        "detector_mode": os.getenv("ASSEMBLY_DETECTOR", "demo_color").strip().lower(),
        "weights": _rooted(os.getenv("ASSEMBLY_WEIGHTS", "weights/best.pt")),
        "confidence_threshold": float(os.getenv("ASSEMBLY_CONFIDENCE", "0.55")),
        "low_confidence_floor": float(os.getenv("ASSEMBLY_LOW_CONFIDENCE", "0.25")),
        "batch_warning_threshold": float(os.getenv("ASSEMBLY_BATCH_WARNING", "0.10")),
        "runtime_dir": _rooted(os.getenv("ASSEMBLY_RUNTIME_DIR", ".runtime")),
        "max_image_bytes": int(os.getenv("ASSEMBLY_MAX_IMAGE_MB", "12")) * 1024 * 1024,
    }
    if not 0 < conf["low_confidence_floor"] <= conf["confidence_threshold"] <= 1:
        raise ValueError("Require 0 < low confidence <= accepted confidence <= 1")
    if not 0 <= conf["batch_warning_threshold"] <= 1:
        raise ValueError("Batch warning threshold must be 0..1")
    return conf


@lru_cache(maxsize=1)
def get_layout() -> dict:
    layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    if layout.get("inspection_mode") == "detection_only":
        classes = layout.get("classes")
        if not isinstance(classes, list) or not classes or len(classes) != len(set(classes)):
            raise ValueError("Detection-only layout must provide a unique nonempty class list")
        return layout
    if not layout.get("components"):
        raise ValueError("Assembly layout must include components")
    for label, spec in layout["components"].items():
        if not isinstance(spec.get("count"), int) or spec["count"] <= 0:
            raise ValueError(f"Invalid quantity for {label}")
        zone = spec.get("zone")
        if not isinstance(zone, list) or len(zone) != 4:
            raise ValueError(f"Invalid normalized zone for {label}")
        x1, y1, x2, y2 = zone
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Invalid normalized zone for {label}")
    return layout
