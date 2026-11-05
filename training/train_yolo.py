"""Fine-tune a YOLO model on three labeled custom assembly components.

Usage: python -m training.train_yolo --data dataset/assembly_dataset.yaml --epochs 40
Windows multiprocessing requires the `if __name__ == '__main__'` guard below.
"""
import argparse
from pathlib import Path
from assembly.settings import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "assembly_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu, 0 (NVIDIA GPU), or mps")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained baseline for transfer learning")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}. See dataset/README.md")
    from ultralytics import YOLO
    model = YOLO(args.model)
    results = model.train(data=str(args.data.resolve()), epochs=args.epochs,
                          batch=args.batch, imgsz=args.imgsz, device=args.device,
                          project=str(ROOT / "runs"), name="assembly_training")
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training finished. Copy {best} to weights/best.pt, then switch ASSEMBLY_DETECTOR=yolo")


if __name__ == "__main__":
    main()
