"""Assembly component detectors: real trained YOLO or a clearly marked color DEMO.

The demo method is NOT machine learning. Switching to YOLO requires your own
custom-class weights; a COCO model is not accepted as a manufacturing detector.
"""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from .models import Detection
from .settings import settings, get_layout


class ColorDemoDetector:
    """HSV color-segmentation baseline for generated control-panel test fixtures."""
    mode = "demo_color"
    LABEL_HSV = {
        "red_switch": [((0, 100, 90), (10, 255, 255)), ((170, 100, 90), (179, 255, 255))],
        "blue_connector": [((96, 95, 60), (132, 255, 255))],
        "yellow_latch": [((18, 100, 90), (37, 255, 255))],
    }

    def detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        height, width = bgr.shape[:2]
        min_area = max(100, int(width * height * 0.00055))
        found = []
        for label, ranges in self.LABEL_HSV.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                # The score means a rule matched, NOT learned model confidence.
                found.append(Detection(label, 1.0, (x, y, x + w, y + h)))
        return sorted(found, key=lambda d: d.label)


class YoloAssemblyDetector:
    mode = "yolo"

    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Trained assembly weights not found at {path}. Train a custom model "
                "first; COCO-pretrained labels are not assembly component labels."
            )
        from ultralytics import YOLO  # Lazy import keeps demo/test installation lean.
        self.model = YOLO(str(path))
        trained_names = self.model.names
        model_classes = set(trained_names.values() if isinstance(trained_names, dict) else trained_names)
        required_classes = set(get_layout()["components"])
        if model_classes != required_classes:
            raise ValueError(
                f"Model classes {sorted(model_classes)} do not match required "
                f"assembly classes {sorted(required_classes)}. Train with the correct data.yaml."
            )

    def detect(self, bgr):
        # Retain weak candidates so the review engine can classify uncertainty.
        result = self.model.predict(
            source=bgr, conf=settings()["low_confidence_floor"], verbose=False
        )[0]
        names = result.names
        found = []
        for box in result.boxes:
            label = names[int(box.cls[0].item())]
            confidence = float(box.conf[0].item())
            xyxy = tuple(int(round(v)) for v in box.xyxy[0].tolist())
            found.append(Detection(label, confidence, xyxy))
        return found


@lru_cache(maxsize=4)
def get_detector(mode=None, weights=None):
    conf = settings()
    mode = mode or conf["detector_mode"]
    if mode == "demo_color":
        return ColorDemoDetector()
    if mode == "yolo":
        return YoloAssemblyDetector(Path(weights or conf["weights"]))
    raise ValueError("ASSEMBLY_DETECTOR must be 'demo_color' or 'yolo'")
