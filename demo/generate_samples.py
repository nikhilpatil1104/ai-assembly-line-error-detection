"""Generate transparent, labeled *synthetic* assembly-panel examples.

This is a drawing routine, not real manufacturing data. You may optionally use
--toy-training to explore the YOLO training pipeline. Never report results on
these generated samples as evidence of deployment accuracy.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("red_switch", "blue_connector", "yellow_latch")
BASE = {"red_switch": (225, 315), "blue_connector": (468, 315), "yellow_latch": (710, 315)}
SIZE = (900, 600)


def panel_image(missing=(), moves=None, jitter=0, seed=7):
    """Return (BGR image, ground-truth class->bbox) for a constructed fixture."""
    rng = random.Random(seed)
    width, height = SIZE
    image = np.full((height, width, 3), 234, dtype=np.uint8)
    for row in range(height):
        image[row, :, :] -= int((row / height) * 9)
    cv2.rectangle(image, (75, 115), (825, 515), (85, 98, 107), -1)
    cv2.rectangle(image, (90, 130), (810, 500), (165, 172, 182), -1)
    cv2.rectangle(image, (90, 130), (810, 500), (70, 78, 84), 4)
    for x in (110, 790):
        for y in (150, 480):
            cv2.circle(image, (x, y), 7, (80, 85, 86), -1)
            cv2.line(image, (x - 4, y), (x + 4, y), (200, 201, 203), 2)
    cv2.putText(image, "CONTROL PANEL / ASSEMBLY QC", (242, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 58, 66), 2, cv2.LINE_AA)
    bounds = {}
    for label, (px, py) in BASE.items():
        if label in missing:
            continue
        x, y = (moves or {}).get(label, (px, py))
        x += rng.randint(-jitter, jitter) if jitter else 0
        y += rng.randint(-jitter, jitter) if jitter else 0
        if label == "red_switch":
            cv2.circle(image, (x, y), 49, (43, 44, 46), -1)
            cv2.circle(image, (x, y), 42, (28, 40, 226), -1)
            cv2.circle(image, (x - 9, y - 9), 14, (85, 100, 251), -1)
            bounds[label] = (x - 42, y - 42, x + 43, y + 43)
        elif label == "blue_connector":
            cv2.rectangle(image, (x - 53, y - 48), (x + 53, y + 48), (50, 56, 62), -1)
            cv2.rectangle(image, (x - 44, y - 38), (x + 44, y + 38), (226, 104, 28), -1)
            cv2.rectangle(image, (x - 27, y - 23), (x + 27, y + 23), (240, 135, 55), -1)
            bounds[label] = (x - 44, y - 38, x + 45, y + 39)
        elif label == "yellow_latch":
            cv2.rectangle(image, (x - 53, y - 45), (x + 53, y + 45), (56, 58, 61), -1)
            cv2.rectangle(image, (x - 44, y - 35), (x + 44, y + 35), (29, 215, 244), -1)
            cv2.rectangle(image, (x - 24, y - 27), (x + 24, y + 27), (50, 233, 252), -1)
            bounds[label] = (x - 44, y - 35, x + 45, y + 36)
    return image, bounds


def write_example_images(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    examples = {
        "pass.png": {"missing": (), "moves": {}, "expected_status": "Pass"},
        "missing_blue.png": {"missing": ("blue_connector",), "moves": {}, "expected_status": "Needs Review"},
        "misplaced_yellow.png": {"missing": (), "moves": {"yellow_latch": (580, 315)}, "expected_status": "Needs Review"},
        "misplaced_red.png": {"missing": (), "moves": {"red_switch": (350, 315)}, "expected_status": "Needs Review"},
    }
    manifest = []
    for name, spec in examples.items():
        image, truth = panel_image(missing=spec["missing"], moves=spec["moves"])
        cv2.imwrite(str(folder / name), image)
        manifest.append({"filename": name, "expected_status": spec["expected_status"],
                         "ground_truth_bboxes": {label: list(box) for label, box in truth.items()}})
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} synthetic examples to {folder}")


def create_toy_yolo_dataset(folder: Path, seed=17):
    """Small toy labeled dataset for pipeline demonstration, not real accuracy claims."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    total = 0
    for split, num in (("train", 100), ("val", 25), ("test", 25)):
        images_dir = folder / "images" / split
        labels_dir = folder / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(num):
            missing = [label for label in CLASS_NAMES if rng.random() < .15]
            moves = {}
            if rng.random() < .22:
                label = rng.choice(CLASS_NAMES)
                moves[label] = (BASE[label][0] + rng.choice([-125, 125]), BASE[label][1])
            img, boxes = panel_image(missing=missing, moves=moves, jitter=10, seed=rng.randint(0, 10**8))
            filename = f"panel_{idx:04}.png"
            cv2.imwrite(str(images_dir / filename), img)
            label_rows = []
            for label, (x1, y1, x2, y2) in boxes.items():
                class_index = CLASS_NAMES.index(label)
                cx, cy = (x1 + x2) / (2 * SIZE[0]), (y1 + y2) / (2 * SIZE[1])
                bw, bh = (x2 - x1) / SIZE[0], (y2 - y1) / SIZE[1]
                label_rows.append(f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (labels_dir / f"panel_{idx:04}.txt").write_text("\n".join(label_rows), encoding="utf-8")
            total += 1
    # Absolute path ensures dataset works regardless of current directory.
    yaml_text = (f"path: '{folder.resolve().as_posix()}'\n"
                 "train: images/train\nval: images/val\ntest: images/test\n"
                 "names:\n  0: red_switch\n  1: blue_connector\n  2: yellow_latch\n")
    yaml_file = folder / "assembly_dataset.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {total} synthetic YOLO-labeled images. Dataset YAML: {yaml_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-training", action="store_true", help="Generate toy train/val/test YOLO dataset")
    args = parser.parse_args()
    write_example_images(ROOT / "demo" / "examples")
    if args.toy_training:
        create_toy_yolo_dataset(ROOT / "dataset")


if __name__ == "__main__":
    main()
