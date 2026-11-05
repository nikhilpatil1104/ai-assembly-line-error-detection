"""Data contracts used by the detector, rules, API, and database."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    def as_dict(self):
        return {"label": self.label, "confidence": round(self.confidence, 4), "bbox": list(self.bbox)}
