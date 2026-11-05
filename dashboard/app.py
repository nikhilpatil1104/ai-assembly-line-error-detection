"""Optional Streamlit UI to exercise the independent assembly inspection API."""
import io
import os

import requests
import streamlit as st

st.set_page_config(page_title="Assembly Quality Inspector", page_icon="🏭", layout="wide")
st.title("Assembly Line · Component Placement Inspection")
st.caption("Fixed-camera manufacturing QC prototype · human review required for flagged units")

api_base = st.sidebar.text_input("FastAPI base URL", os.getenv("ASSEMBLY_API", "http://127.0.0.1:8000")).rstrip("/")
try:
    health = requests.get(f"{api_base}/health", timeout=10)
    health.raise_for_status()
    st.sidebar.success(f"API online · {health.json()['detector_mode']}")
except requests.RequestException as exc:
    st.sidebar.error(f"Backend not reachable: {exc}")
    st.info("Run: uvicorn assembly.api:app --reload")
    st.stop()

if health.json().get("detector_mode") == "demo_color":
    st.warning("DEMONSTRATION MODE: Color segmentation on synthetic example images, not trained YOLO AI. Use your own custom-trained YOLO weights for real inspection.")

try:
    response = requests.get(f"{api_base}/assembly/layout", timeout=10)
    response.raise_for_status()
    layout = response.json()
except requests.RequestException as exc:
    st.error(f"Cannot load assembly layout: {exc}")
    st.stop()

st.subheader(layout.get("display_name", "Assembly"))
st.write("Required components and normalized center-placement zones:")
st.dataframe([{"Component": label, "Required": spec["count"], "Zone [x1,y1,x2,y2]": str(spec["zone"])}
              for label, spec in layout["components"].items()], use_container_width=True)

uploads = st.file_uploader("Upload one or several JPG/PNG assembly images", type=["jpg", "jpeg", "png"],
                           accept_multiple_files=True)

if st.button("Inspect assembly images", type="primary", disabled=not uploads):
    with st.spinner("Inspecting and checking placements..."):
        try:
            if len(uploads) == 1:
                files = {"file": (uploads[0].name, uploads[0].getvalue(), uploads[0].type)}
                response = requests.post(f"{api_base}/inspect", files=files, timeout=180)
                response.raise_for_status()
                record = response.json()
                result = {"inspections": [record]}
            else:
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploads]
                response = requests.post(f"{api_base}/batches/inspect", files=files, timeout=300)
                response.raise_for_status()
                result = response.json()
            st.session_state["inspection_result"] = result
        except requests.RequestException as exc:
            st.error(f"Inspection failed: {exc}")

if "inspection_result" in st.session_state:
    result = st.session_state["inspection_result"]
    inspections = result["inspections"]
    inspected = result.get("inspected_units", len(inspections))
    flagged = result.get("flagged_units", sum(r["status"] != "Pass" for r in inspections))
    a, b, c = st.columns(3)
    a.metric("Inspected", inspected)
    b.metric("Flagged", flagged)
    c.metric("Flag rate", f"{flagged / inspected:.1%}" if inspected else "0%")
    if result.get("batch_quality_warning"):
        st.error("BATCH QUALITY WARNING: More than 10% of units need review.")
    for record in inspections:
        with st.expander(f"{record['filename']} — {record['status']}", expanded=True):
            st.write("**Reason(s):**", "; ".join(record.get("reasons", [])) or "All required parts confidently located in approved zones")
            st.write("**Detected counts:**", record.get("detected_counts", {}))
            st.write("**Missing:**", record.get("missing", {}))
            st.write("**Misplaced:**", record.get("misplaced", []))
            st.caption(f"Processing: {record.get('processing_ms', 0)} ms")
            image_url = record.get("annotated_image_url")
            if image_url:
                try:
                    picture = requests.get(api_base + image_url, timeout=30)
                    picture.raise_for_status()
                    st.image(io.BytesIO(picture.content), caption="Detected boxes and allowable placement zones")
                except requests.RequestException as exc:
                    st.info(f"Annotated image unavailable: {exc}")
    try:
        csv = requests.get(f"{api_base}/inspections/export.csv", timeout=30)
        csv.raise_for_status()
        st.download_button("Download inspection history (CSV)", data=csv.content,
                           file_name="assembly_inspection_history.csv", mime="text/csv")
    except requests.RequestException as exc:
        st.info(f"CSV export unavailable: {exc}")
