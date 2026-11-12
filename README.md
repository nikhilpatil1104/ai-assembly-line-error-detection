# Engine Assembly Object Detection

This project uses a real image dataset to detect engine assembly objects with YOLO and provides a FastAPI backend plus a Streamlit frontend. It is currently a **detection demo**, not a validated assembly quality inspector: the dataset does not label missing parts, incorrect placement, or pass/fail status.

## Dataset

The project uses the [Engine Assembly Dataset on Zenodo](https://zenodo.org/records/7669593). The downloaded archive is `dataset/EngineAssembly.zip`; the source archive and converted images are excluded from Git. The preparation script converts the 195 original annotated images (excluding augmented copies) into YOLO labels and a reproducible 70/15/15 split:

- 137 training images
- 29 validation images
- 29 test images

Classes: `PushRod`, `Bolt`, `RockerArmObject`, `BoltHoles`, `Engine`, `RockerArm`, `Small_PushRodHoles`, and `Big_PushRodHoles`.

Check the dataset record for usage and redistribution terms before external or commercial use.

## Windows setup

Run these commands in PowerShell from the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt -r requirements-ml.txt
```

Prepare the YOLO dataset if it is not already present:

```powershell
python -m training.prepare_engine_dataset
```

Train the model. CPU training is supported; use `--device 0` if you have a compatible NVIDIA GPU:

```powershell
python -m training.train_yolo --epochs 10 --imgsz 320 --device cpu
```

The trained model is copied to `weights/engine_assembly_best.pt`. Evaluate it on the held-out test split:

```powershell
python -m training.evaluate_yolo --split test
```

## Run the app

Open **two PowerShell windows** in the project directory and activate the same environment in each.

Window 1 - backend API:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn assembly.api:app --reload
```

Window 2 - Streamlit frontend:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

Open the Streamlit URL printed by the command (usually `http://localhost:8501`). The API docs are at `http://localhost:8000/docs`. Upload an engine assembly image to view detected class names, confidence scores, and bounding boxes. The API returns **Detection Complete** when it finds confident objects and **Needs Review** when results are uncertain; neither status means the assembly passed or failed quality control.

## Host a public demo

The repository is prepared for a free prototype deployment: Render hosts the FastAPI backend, and Streamlit Community Cloud hosts the UI.

1. In Render, create a **Blueprint** from this GitHub repository and deploy `render.yaml`. This creates the API using the included model weights. Wait for its deploy to finish and copy its public `https://...onrender.com` URL.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from this repository, branch `main`, entrypoint `dashboard/app.py`.
3. In the app's settings, add this secret, replacing the value with your Render API URL:

   ```toml
   ASSEMBLY_API = "https://your-api-name.onrender.com"
   ```

4. Open the app URL from Streamlit Community Cloud and upload an engine assembly photo.

The free Render API sleeps after 15 minutes without requests, so its first request after idling can take about a minute. Its local inspection history and annotated images are temporary and may be cleared when the service sleeps or redeploys. The free instance has 512 MB RAM; if Render reports an out-of-memory restart, change the API service to a 2 GB plan. This is a public prototype without login protection; do not upload sensitive or production images.

## Troubleshooting

- If a virtual environment was moved from another folder, delete and recreate it with the setup commands above. Virtual environments contain absolute paths and do not survive moves reliably.
- If `uvicorn` or `streamlit` is not recognized, run them through Python: `python -m uvicorn assembly.api:app --reload` and `python -m streamlit run dashboard/app.py`.
- If the backend reports missing model weights, finish training and check that `weights/engine_assembly_best.pt` exists.
- For a clean reinstall when OpenCV is locked by a running app, close the project’s Python processes, then retry the install in the activated environment.
