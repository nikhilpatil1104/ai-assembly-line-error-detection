# Real assembly dataset (required for a validated AI system)

The included color demo is a **synthetic test fixture**, not a manufacturing model.

1. Choose a single assembly with 3 required parts corresponding to `red_switch`, `blue_connector`, `yellow_latch` or adapt `config/layout.json` and the custom YOLO classes consistently.
2. Capture consented, fixed-camera images of complete, missing-part, misplaced-part, and borderline/occluded assemblies, varying lighting realistically.
3. Annotate **visible part bounding boxes** with a labeling tool using YOLO detection format (`class_id x_center y_center width height`, normalized to 0..1).
4. Keep distinct training, validation, and test images, ideally group by capture session so near-duplicates do not leak across splits.
5. Put the files under `dataset/images/{train,val,test}` and `dataset/labels/{train,val,test}`.
6. Create a `dataset/assembly_dataset.yaml` with:

```yaml
path: C:/absolute/path/to/ai-assembly-line-error-detection/dataset
train: images/train
val: images/val
test: images/test
names:
  0: red_switch
  1: blue_connector
  2: yellow_latch
```

The class names are **exact** and must match `config/layout.json`, or the real detector will refuse to start.

Do **not** put personal, private, or proprietary factory images in a public GitHub repository without permission. Do not upload pretrained weight files, proprietary data, or model output files by default.

For a *toy* end-to-end training smoke test only, run:

```powershell
python -m demo.generate_samples --toy-training
```

That creates an artificial labeled train/val/test split and the dataset YAML automatically. It is useful for exercising the code, but its performance numbers are not evidence of real manufacturing accuracy.
