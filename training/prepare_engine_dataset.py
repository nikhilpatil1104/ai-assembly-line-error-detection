"""Convert the Engine Assembly COCO annotations to a leak-conscious YOLO split.

Only the 195 original annotated photographs in the COCO manifest are used.
The archive's generated augmentation set is deliberately excluded.
"""
import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from assembly.settings import ROOT


def build_split(images, annotations, class_ids, seed=690):
    rng = random.Random(seed)
    labels_by_image = defaultdict(set)
    for ann in annotations:
        labels_by_image[ann["image_id"]].add(ann["category_id"])

    unassigned = {image["id"]: image for image in images}
    split_ids = {"val": set(), "test": set()}
    for split in ("val", "test"):
        target_count = round(len(images) * 0.15)
        for class_id in sorted(class_ids, key=lambda c: sum(c in labels_by_image[i] for i in unassigned)):
            class_total = sum(class_id in labels_by_image[i] for i in unassigned)
            needed = max(1, round(class_total * 0.15))
            have = sum(class_id in labels_by_image[i] for i in split_ids[split])
            candidates = [i for i in unassigned if class_id in labels_by_image[i]]
            rng.shuffle(candidates)
            for image_id in candidates[:max(0, needed - have)]:
                split_ids[split].add(image_id)
                unassigned.pop(image_id, None)
        remaining_target = max(0, target_count - len(split_ids[split]))
        candidates = list(unassigned)
        rng.shuffle(candidates)
        for image_id in candidates[:remaining_target]:
            split_ids[split].add(image_id)
            unassigned.pop(image_id, None)
    split_ids["train"] = set(unassigned)
    return split_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "dataset" / "EngineAssembly.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset" / "engine_assembly_yolo")
    parser.add_argument("--seed", type=int, default=690)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"Dataset archive not found: {args.archive}")

    with zipfile.ZipFile(args.archive) as archive:
        candidates = [n for n in archive.namelist() if n.endswith("/Dataset/COCO/CocoJSON/CocoJSON.json")]
        if len(candidates) != 1:
            raise SystemExit("Could not uniquely find the Engine Assembly COCO manifest in the archive.")
        manifest_path = candidates[0]
        root = manifest_path.rsplit("/Dataset/COCO/CocoJSON/CocoJSON.json", 1)[0]
        data = json.loads(archive.read(manifest_path))
        categories = sorted(data["categories"], key=lambda c: c["id"])
        old_to_new = {category["id"]: index for index, category in enumerate(categories)}
        image_names = {image["id"]: image["file_name"] for image in data["images"]}
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)
        splits = build_split(data["images"], data["annotations"], set(old_to_new), args.seed)
        split_for = {image_id: split for split, ids in splits.items() for image_id in ids}
        if len(split_for) != len(data["images"]):
            raise SystemExit("Split did not assign every source image exactly once.")

        args.output.mkdir(parents=True, exist_ok=True)
        counts = {split: Counter() for split in splits}
        for image in data["images"]:
            split = split_for[image["id"]]
            filename = image["file_name"]
            source = f"{root}/Dataset/COCO/Images/{filename}"
            if source not in archive.namelist():
                raise SystemExit(f"Image referenced by annotations is missing: {source}")
            image_dir = args.output / "images" / split
            label_dir = args.output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / filename).write_bytes(archive.read(source))
            lines = []
            width, height = image["width"], image["height"]
            for ann in annotations[image["id"]]:
                x, y, box_w, box_h = ann["bbox"]
                xc = min(1.0, max(0.0, (x + box_w / 2) / width))
                yc = min(1.0, max(0.0, (y + box_h / 2) / height))
                nw = min(1.0, max(0.0, box_w / width))
                nh = min(1.0, max(0.0, box_h / height))
                lines.append(f"{old_to_new[ann['category_id']]} {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")
                counts[split][ann["category_id"]] += 1
            (label_dir / (Path(filename).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

    names = [category["name"] for category in categories]
    yaml = [f"path: {args.output.resolve().as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(names))
    (args.output / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    report = {
        "source": "Engine Assembly Dataset, Zenodo 7669593",
        "source_images": len(data["images"]),
        "augmentation_images_used": 0,
        "seed": args.seed,
        "classes": names,
        "images_per_split": {split: len(ids) for split, ids in splits.items()},
        "boxes_per_class_and_split": {
            split: {category["name"]: counts[split][category["id"]] for category in categories}
            for split in splits
        },
    }
    (args.output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
