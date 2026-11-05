# Model weights

In `ASSEMBLY_DETECTOR=yolo` mode, place your **custom assembly-trained** Ultralytics model at `weights/best.pt` (or set `ASSEMBLY_WEIGHTS` to another file).

The unmodified COCO-pretrained `yolov8n.pt` is for **transfer learning only**, not a manufacturing inference model. Class labels must match `red_switch`, `blue_connector`, and `yellow_latch` exactly, and the API validates this.

Weights (`*.pt`) are excluded from Git. Document where trained weights can be retrieved if your instructor needs to run YOLO inference.
