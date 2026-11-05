"""Package-level (here, ASSEMBLY-level) status evaluation vs reviewer labels.

Requires a manifest of images and expected statuses. Reports *measured* results
from the selected detector, never silently claims real manufacturing accuracy.
"""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from assembly.detectors import get_detector
from assembly.images import decode_image
from assembly.settings import ROOT, get_layout, settings
from assembly.verification import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "demo" / "examples" / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "decision_evaluation_results.json")
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    detector = get_detector()
    conf = settings()
    truth = []
    predicted = []
    processing_ms = []
    for entry in entries:
        start = perf_counter()
        image = decode_image((args.manifest.parent / entry["filename"]).read_bytes())
        detections = detector.detect(image)
        h, w = image.shape[:2]
        decision = verify(detections, (w, h), get_layout(), conf["confidence_threshold"])
        processing_ms.append((perf_counter() - start) * 1000)
        truth.append(entry["expected_status"])
        predicted.append(decision["status"])
        print(entry["filename"], "expected:", entry["expected_status"], "got:", decision["status"])
    total = len(truth)
    if not total:
        raise SystemExit("No labeled examples in manifest")
    false_flags = sum(t == "Pass" and p != "Pass" for t, p in zip(truth, predicted))
    missed_flags = sum(t != "Pass" and p == "Pass" for t, p in zip(truth, predicted))
    passed_truth = sum(t == "Pass" for t in truth)
    result = {
        "mode": detector.mode, "manifest": str(args.manifest.resolve()),
        "dataset_type": "synthetic_demo" if "demo/examples" in args.manifest.as_posix() else "user_supplied",
        "num_images": total,
        "pass_review_accuracy": sum(t == p for t, p in zip(truth, predicted)) / total,
        "false_flag_rate_among_true_pass": false_flags / passed_truth if passed_truth else None,
        "missed_defects": missed_flags,
        "mean_processing_ms": round(mean(processing_ms), 2),
        "note": "Use labeled REAL assembly images and a custom YOLO model for manufacturing accuracy claims."
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
