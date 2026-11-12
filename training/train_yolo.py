"""Fine-tune YOLO on the real Engine Assembly Dataset.

Usage: python -m training.prepare_engine_dataset
       python -m training.train_yolo --epochs 40 --device cpu
Windows multiprocessing requires the `if __name__ == '__main__'` guard below.
"""
import argparse
from pathlib import Path
from assembly.settings import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "engine_assembly_yolo" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu", help="cpu, 0 (NVIDIA GPU), or mps")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO checkpoint for transfer learning")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}. See dataset/README.md")
    from ultralytics import YOLO
    model = YOLO(args.model)
    results = model.train(data=str(args.data.resolve()), epochs=args.epochs,
                          batch=args.batch, imgsz=args.imgsz, device=args.device,
                          project=str(ROOT / "runs"), name="engine_assembly")
    best = Path(results.save_dir) / "weights" / "best.pt"
    destination = ROOT / "weights" / "engine_assembly_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(best, destination)
    print(f"Training finished. Model copied to {destination}")


if __name__ == "__main__":
    main()
