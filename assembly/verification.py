"""Assembly-focused decision engine: required parts + center-in-zone + uncertainty."""
from collections import Counter
from .models import Detection


def verify(detections: list[Detection], image_size: tuple[int, int], layout: dict,
           confidence_threshold: float = 0.55) -> dict:
    """All components must be confidently present AND centered in their zones.

    A box must have >= threshold score to count as present. Low-confidence
    predictions trigger review, not silent acceptance. The system cannot
    distinguish true absence from a failure to detect an occluded part.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    if layout.get("inspection_mode") == "detection_only":
        certain = [d for d in detections if d.confidence >= confidence_threshold]
        low = [d for d in detections if d.confidence < confidence_threshold]
        counts = Counter(d.label for d in certain)
        reasons = []
        codes = []
        if low:
            codes.append("uncertain_detection")
            reasons.append("Low-confidence detections need human review: " +
                           ", ".join(sorted({d.label for d in low})))
        if not certain:
            codes.append("no_confident_detections")
            reasons.append("No engine parts were confidently detected; check the image manually.")
        return {
            "status": "Needs Review" if reasons else "Detection Complete",
            "reason_codes": codes,
            "reasons": reasons,
            "missing": {}, "misplaced": [], "extra": {}, "unknown": {},
            "uncertain": bool(low),
            "detected_counts": dict(counts),
            "decision_scope": "Object detection only; assembly completeness and placement are not assessed.",
        }
    required = layout["components"]
    certain = [d for d in detections if d.confidence >= confidence_threshold]
    low = [d for d in detections if d.confidence < confidence_threshold]
    counts = Counter(d.label for d in certain)

    missing = {
        label: spec["count"] - counts.get(label, 0)
        for label, spec in required.items() if counts.get(label, 0) < spec["count"]
    }
    extra = {
        label: counts[label] - spec["count"]
        for label, spec in required.items() if counts[label] > spec["count"]
    }
    unknown = {label: count for label, count in counts.items() if label not in required}

    misplaced = []
    for det in certain:
        if det.label not in required:
            continue
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / (2 * width)
        cy = (y1 + y2) / (2 * height)
        zone = required[det.label]["zone"]
        if not (zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]):
            misplaced.append({"component": det.label,
                              "center": [round(cx, 4), round(cy, 4)],
                              "allowed_zone": zone, "bbox": list(det.bbox)})

    reasons = []
    codes = []
    if missing:
        codes.append("missing_component")
        reasons.append("Required components not confidently detected: " +
                       ", ".join(f"{k} x{v}" for k, v in missing.items()))
    if misplaced:
        codes.append("misplacement")
        reasons.append("Component center outside approved zone: " +
                       ", ".join(sorted({p["component"] for p in misplaced})))
    if extra:
        codes.append("extra_component")
        reasons.append("Extra instances require review: " +
                       ", ".join(f"{k} x{v}" for k, v in extra.items()))
    if unknown:
        codes.append("unknown_component")
        reasons.append("Unknown detected classes require review: " + ", ".join(sorted(unknown)))
    if low:
        codes.append("uncertain_detection")
        reasons.append("Low-confidence detections require human review: " +
                       ", ".join(sorted({d.label for d in low})))
    return {
        "status": "Needs Review" if reasons else "Pass",
        "reason_codes": codes,
        "reasons": reasons,
        "missing": missing,
        "misplaced": misplaced,
        "extra": extra,
        "unknown": unknown,
        "uncertain": bool(low),
        "detected_counts": dict(counts),
    }


def batch_quality(inspection_statuses: list[str], warning_threshold: float = 0.10) -> dict:
    total = len(inspection_statuses)
    flagged = sum(status not in ("Pass", "Detection Complete") for status in inspection_statuses)
    passed = sum(status == "Pass" for status in inspection_statuses)
    completed = sum(status in ("Pass", "Detection Complete") for status in inspection_statuses)
    rate = flagged / total if total else 0.0
    return {
        "inspected_units": total,
        "flagged_units": flagged,
        "passed_units": passed,
        "completed_units": completed,
        "flagged_rate": round(rate, 4),
        "warning_threshold": warning_threshold,
        # Requirement: more than 10%, not >=10%.
        "batch_quality_warning": total > 0 and rate > warning_threshold,
    }
