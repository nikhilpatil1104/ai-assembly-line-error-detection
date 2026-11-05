"""Evaluate model on held-out data. Metrics are NOT claimed unless script runs."""
import argparse
import json
from pathlib import Path
from assembly.settings import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "assembly_dataset.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        raise SystemExit("Need both trained weights and labeled dataset YAML (see README).")
    from ultralytics import YOLO
    results = YOLO(str(args.weights)).val(data=str(args.data.resolve()), split=args.split,
                                          imgsz=640, plots=True)
    summary = {
        "split": args.split, "data": str(args.data.resolve()),
        "weights": str(args.weights.resolve()),
        "precision": float(results.box.mp), "recall": float(results.box.mr),
        "mAP50": float(results.box.map50), "mAP50_95": float(results.box.map),
    }
    dest = ROOT / "evaluation_results.json"
    dest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {dest}. Include this with a note on which dataset was evaluated.")


if __name__ == "__main__":
    main()
