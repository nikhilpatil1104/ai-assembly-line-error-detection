"""Streamlit frontend for the assembly inspection API."""
import io
import os

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


st.set_page_config(page_title="Assembly Quality Inspector", page_icon="🏭", layout="wide")
st.title("Engine Assembly Object Detection")
st.caption("Upload engine assembly photos to detect and review engine part labels.")

try:
    configured_api = st.secrets.get("ASSEMBLY_API")
except StreamlitSecretNotFoundError:
    configured_api = None
api_base = st.sidebar.text_input(
    "FastAPI base URL", os.getenv("ASSEMBLY_API", configured_api or "http://127.0.0.1:8000")
).strip().rstrip("/")
st.sidebar.caption("Start the API with `uvicorn assembly.api:app --reload`.")


def api_get(path, *, timeout=15):
    response = requests.get(f"{api_base}{path}", timeout=timeout)
    response.raise_for_status()
    return response


def show_api_error(prefix, exc):
    detail = getattr(getattr(exc, "response", None), "text", "")
    st.error(f"{prefix}: {detail or exc}")


try:
    health = api_get("/health").json()
except requests.RequestException as exc:
    st.sidebar.error("API offline")
    st.error(f"Backend not reachable: {exc}")
    st.info("Start the backend in a separate terminal, then refresh this page.")
    st.stop()

st.sidebar.success(f"API online · {health.get('detector_mode', 'unknown')} mode")
try:
    layout = api_get("/assembly/layout").json()
except requests.RequestException as exc:
    show_api_error("Could not load assembly layout", exc)
    st.stop()

if layout.get("inspection_mode") == "detection_only":
    st.info("Engine-part object detection only. This dataset does not label missing-part or placement defects, so results are not assembly Pass/Fail decisions.")
elif health.get("detector_mode") == "demo_color":
    st.warning("Demo mode uses color segmentation on synthetic examples. It is not a trained AI model.")

inspect_tab, history_tab, summary_tab, layout_tab = st.tabs(
    ["Inspect", "History", "Summary", "Assembly layout"]
)

with inspect_tab:
    st.subheader(layout.get("display_name", "Assembly"))
    st.caption(layout.get("description", "Upload engine assembly images to detect labeled parts."))
    uploads = st.file_uploader(
        "Choose JPG, PNG, or WebP images (up to 100 per batch)",
        type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
        key="inspection_uploads",
    )
    if len(uploads) > 100:
        st.error("The API accepts at most 100 images per batch.")
    if st.button("Run inspection", type="primary", disabled=not uploads or len(uploads) > 100):
        try:
            with st.spinner(f"Inspecting {len(uploads)} image(s)…"):
                if len(uploads) == 1:
                    image = uploads[0]
                    response = requests.post(
                        f"{api_base}/inspect",
                        files={"file": (image.name, image.getvalue(), image.type)}, timeout=180,
                    )
                    response.raise_for_status()
                    record = response.json()
                    flagged = int(record.get("status") != "Pass")
                    result = {
                        "inspections": [record], "inspected_units": 1,
                        "flagged_units": flagged, "passed_units": 0,
                        "completed_units": 1 - flagged, "flagged_rate": float(flagged),
                    }
                else:
                    files = [("files", (image.name, image.getvalue(), image.type)) for image in uploads]
                    response = requests.post(f"{api_base}/batches/inspect", files=files, timeout=300)
                    response.raise_for_status()
                    result = response.json()
            st.session_state["inspection_result"] = result
        except requests.RequestException as exc:
            show_api_error("Inspection failed", exc)

    result = st.session_state.get("inspection_result")
    if result:
        records = result.get("inspections", [])
        total = result.get("inspected_units", len(records))
        flagged = result.get("flagged_units", sum(r.get("status") != "Pass" for r in records))
        completed = result.get("completed_units", result.get("passed_units", total - flagged))
        a, b, c = st.columns(3)
        a.metric("Inspected", total)
        b.metric("Detection complete", completed)
        c.metric("Needs review", flagged)
        if result.get("batch_quality_warning"):
            st.error(f"Batch quality warning: {result.get('flagged_rate', 0):.1%} flagged.")
        for record in records:
            with st.expander(f"{record.get('filename', 'Image')} · {record.get('status', 'Unknown')}", expanded=True):
                left, right = st.columns([1, 1])
                with left:
                    st.write("**Reasons**", "; ".join(record.get("reasons", [])) or "No issues detected")
                    st.write("**Detected counts**", record.get("detected_counts", {}))
                    st.caption(record.get("decision_scope", ""))
                    detections = record.get("detections", [])
                    if detections:
                        st.dataframe(pd.DataFrame([
                            {"Part": det["label"], "Confidence": det["confidence"], "Bounding box": det["bbox"]}
                            for det in detections
                        ]), hide_index=True, width="stretch", alt="Detected parts, confidence scores, and bounding boxes")
                    st.caption(f"{record.get('processing_ms', 0)} ms · {record.get('detector_mode', '')}")
                with right:
                    image_url = record.get("annotated_image_url")
                    if image_url:
                        try:
                            picture = api_get(image_url, timeout=30)
                            st.image(io.BytesIO(picture.content), caption="Detected engine assembly objects", alt="Uploaded engine assembly image with model detections overlaid")
                        except requests.RequestException as exc:
                            st.info(f"Annotated image unavailable: {exc}")
        try:
            csv_response = api_get("/inspections/export.csv", timeout=30)
            st.download_button(
                "Download inspection history (CSV)", csv_response.content,
                file_name="assembly_inspection_history.csv", mime="text/csv",
            )
        except requests.RequestException as exc:
            st.info(f"CSV export unavailable: {exc}")

with history_tab:
    st.subheader("Recent inspections")
    history_limit = st.number_input("Rows to load", min_value=1, max_value=1000, value=50, step=25)
    try:
        history = api_get(f"/inspections?limit={int(history_limit)}").json()
        if history:
            st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True, alt="Recent inspection history")
        else:
            st.info("No inspections have been recorded yet.")
    except requests.RequestException as exc:
        show_api_error("Could not load history", exc)

with summary_tab:
    st.subheader("Inspection summary")
    try:
        summary = api_get("/summary").json()
        total = summary.get("inspected_units", 0)
        completed = summary.get("detection_complete_units", summary.get("passed_units", 0))
        flagged = summary.get("flagged_units", 0)
        a, b, c, d = st.columns(4)
        a.metric("Inspected", total)
        b.metric("Detection complete", completed)
        c.metric("Needs review", flagged)
        d.metric("Flag rate", f"{summary.get('flagged_rate', 0):.1%}")
        st.subheader("Detected engine parts")
        class_counts = summary.get("detection_counts", {})
        st.dataframe(pd.DataFrame([
            {"Part class": name, "Detections": count}
            for name, count in class_counts.items()
        ]), hide_index=True, width="stretch", alt="Counts of engine part detections")
        st.subheader("Review reasons")
        reasons = summary.get("review_reasons", {})
        st.dataframe(pd.DataFrame([
            {"Reason": name.replace("_", " ").title(), "Units": count}
            for name, count in reasons.items()
        ]), hide_index=True, width="stretch", alt="Reasons detections were sent for review")
    except requests.RequestException as exc:
        show_api_error("Could not load summary", exc)

with layout_tab:
    st.subheader(layout.get("display_name", "Assembly layout"))
    st.write(layout.get("description", "Configured component placement zones."))
    if layout.get("inspection_mode") == "detection_only":
        rows = [{"Detected part class": name} for name in layout.get("classes", [])]
    else:
        rows = [
            {"Component": name, "Required count": spec["count"], "Zone [x1, y1, x2, y2]": spec["zone"]}
            for name, spec in layout.get("components", {}).items()
        ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, alt="Classes used by the engine object detector")
