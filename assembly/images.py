"""Safe image decoding plus bounding boxes and normalized placement zone overlays."""
import cv2
import numpy as np
from .models import Detection


class InvalidImage(ValueError):
    pass


def decode_image(raw: bytes):
    if not raw:
        raise InvalidImage("Empty image")
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage("Invalid image. Please upload JPG or PNG.")
    height, width = image.shape[:2]
    if height < 80 or width < 80 or height * width > 24_000_000:
        raise InvalidImage("Image dimensions outside accepted range (80px minimum, 24MP maximum)")
    return image


def draw_overlay(image, layout: dict, detections: list[Detection], decision: dict,
                 confidence_threshold: float = 0.55) -> bytes:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    if layout.get("inspection_mode") != "detection_only":
        for label, spec in layout["components"].items():
            x1, y1, x2, y2 = spec["zone"]
            start = (int(x1 * width), int(y1 * height))
            end = (int(x2 * width), int(y2 * height))
            cv2.rectangle(overlay, start, end, (120, 175, 80), 2)
            cv2.putText(overlay, label, (start[0], max(start[1] - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 80, 20), 2, cv2.LINE_AA)
    misplaced = {tuple(row["bbox"]) for row in decision["misplaced"]}
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        bad = tuple(d.bbox) in misplaced or d.label in decision["unknown"]
        color = ((10, 10, 230) if bad else
                 (15, 180, 230) if d.confidence < confidence_threshold else (20, 175, 30))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        # Circle the center: this is the exact coordinate used by placement rules.
        cv2.circle(overlay, ((x1 + x2) // 2, (y1 + y2) // 2), 5, color, -1)
        cv2.putText(overlay, f"{d.label}: {d.confidence:.2f}", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode annotated output")
    return encoded.tobytes()
