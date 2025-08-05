import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Plaque Counter with Locked ROI")

# Upload + parameters
uploaded_file = st.file_uploader("Upload a petri dish photo", type=["png", "jpg", "jpeg", "tiff", "tif"])
advanced = st.checkbox("Advanced Settings")

# --- Colony or Plaque (Invert toggle) ---
mode = st.radio("Image Type", options=["Colony", "Plaque"], index=1, horizontal=True)
invert = mode == "Plaque"

# --- Parameter Sliders ---
slider_kwargs = dict(label_visibility="visible" if advanced else "visible")
diameter = st.slider("Feature Diameter", 5, 51, 15, 2, **slider_kwargs)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1, **slider_kwargs)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1, **slider_kwargs)
separation = st.slider("Minimum Separation", 1, 30, 5, 1, **slider_kwargs) if advanced else None

# --- Parameter Saving & Loading ---
if "saved_param_sets" not in st.session_state:
    st.session_state.saved_param_sets = {}

st.markdown("#### Save or Load Parameter Sets")
with st.expander("Manage Parameter Sets", expanded=False):
    param_name = st.text_input("Save current parameters as:")
    if st.button("Save Parameters"):
        if param_name:
            st.session_state.saved_param_sets[param_name] = {
                "invert": invert,
                "diameter": diameter,
                "minmass": minmass,
                "confidence": confidence,
                "separation": separation,
            }
            st.success(f"Saved as '{param_name}'")
        else:
            st.warning("Please enter a name to save the parameters.")

    if st.session_state.saved_param_sets:
        selected = st.selectbox("Load saved parameters", options=["Select..."] + list(st.session_state.saved_param_sets.keys()))
        if selected != "Select...":
            params = st.session_state.saved_param_sets[selected]
            invert = params["invert"]
            diameter = params["diameter"]
            minmass = params["minmass"]
            confidence = params["confidence"]
            separation = params["separation"]
            # Update mode toggle accordingly
            mode = "Plaque" if invert else "Colony"
            st.success(f"Loaded parameters: '{selected}'")

# --- Session state for ROI ---
if "locked_circle_obj" not in st.session_state:
    st.session_state.locked_circle_obj = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = True

def preprocess_image(img, invert=False):
    if invert:
        img = cv2.bitwise_not(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def detect_features(gray, diameter, minmass, separation, confidence):
    norm_img = (gray / 255.0).astype(np.float32)
    try:
        features = tp.locate(
            norm_img,
            diameter=diameter,
            minmass=minmass,
            separation=separation or diameter,
            percentile=confidence,
            invert=False
        )
    except Exception:
        features = pd.DataFrame()
    return features

def detect_dish(gray_img):
    gray_blur = cv2.medianBlur(gray_img, 5)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray_img.shape[0] // 4,
        param1=100,
        param2=30,
        minRadius=gray_img.shape[0] // 5,
        maxRadius=gray_img.shape[0] // 2
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]  # x, y, r
    return None

def extract_circle_geometry(obj):
    raw_r = obj.get("radius", 50)
    left = obj.get("left", 0)
    top = obj.get("top", 0)
    scale_x = obj.get("scaleX", 1.0)
    scale_y = obj.get("scaleY", 1.0)

    scale = (scale_x + scale_y) / 2
    r = raw_r * scale
    x = left + raw_r * scale_x
    y = top + raw_r * scale_y
    return x, y, r

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    h, w = gray.shape

    overlay_objects = []

    if st.session_state.locked_circle_obj is None or st.session_state.edit_mode:
        # Editable mode
        if st.session_state.locked_circle_obj is None:
            detected = detect_dish(gray)
            if detected is None:
                st.error("Could not detect dish. Please upload a clearer image.")
                st.stop()
            x0, y0, r = detected
            circle_obj = {
                "type": "circle",
                "left": float(x0 - r),
                "top": float(y0 - r),
                "radius": float(r),
                "fill": "rgba(255,255,255,0)",
                "stroke": "#FF0000",
                "strokeWidth": 3,
                "selectable": True
            }
        else:
            # Allow re-editing
            circle_obj = st.session_state.locked_circle_obj.copy()
            circle_obj["selectable"] = True

        overlay_objects = [circle_obj]  # Clear all others
    else:
        # Locked mode, use only final locked circle
        circle_obj = st.session_state.locked_circle_obj.copy()
        circle_obj["selectable"] = False
        overlay_objects = [circle_obj]

    # Get geometry of the final/locked circle
    x0, y0, r = extract_circle_geometry(circle_obj)

    # Create circle mask
    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - x0)**2 + (yy - y0)**2 <= r**2

    # Detect all features
    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    # Filter by circle mask
    inside_features = features.copy()
    fx = inside_features["x"].astype(int)
    fy = inside_features["y"].astype(int)
    inside = circle_mask[fy.clip(0, h-1), fx.clip(0, w-1)]
    inside_features = inside_features[inside]

    # Add overlay dots
    for _, row in inside_features.iterrows():
        overlay_objects.append({
            "type": "circle",
            "left": float(row["x"] - 3),
            "top": float(row["y"] - 3),
            "radius": 3,
            "fill": "rgba(0,255,0,0.9)",
            "stroke": "#003300",
            "strokeWidth": 1,
            "selectable": False
        })

    # Show canvas
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#00FF00",
        background_image=pil_img,
        update_streamlit=True,
        height=h,
        width=w,
        initial_drawing={"version": "4.4.0", "objects": overlay_objects},
        drawing_mode="transform",
        key="editable"
    )

    # Lock in the drawn circle
    if st.session_state.locked_circle_obj is None or st.session_state.edit_mode:
        if st.button("Done (Lock Circle)"):
            locked_circle = None
            for obj in canvas_result.json_data.get("objects", []):
                if obj["type"] == "circle" and obj.get("selectable", True):
                    locked_circle = obj
                    break  # Only one allowed

            if locked_circle is not None:
                st.session_state.locked_circle_obj = locked_circle
                st.session_state.edit_mode = False
                st.experimental_rerun()
            else:
                st.error("No valid circle found. Please draw one.")

    # Optional: Reset ROI
    if not st.session_state.edit_mode:
        if st.button("Reset ROI"):
            st.session_state.locked_circle_obj = None
            st.session_state.edit_mode = True
            st.experimental_rerun()

    # Final count output
    st.markdown("### Plaque Count Inside Circle")
    st.success(f"{len(inside_features)} plaques detected inside ROI")

# --- Batch Mode for Folder Processing ---
st.markdown("---")
st.header("Batch Process a Folder of Images")

batch_files = st.file_uploader("Upload multiple dish images (same ROI and settings will apply)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if batch_files and st.button("Process Folder and Export CSV"):
    if st.session_state.locked_circle_obj is None:
        st.error("You must first lock an ROI on a sample image before batch processing.")
    else:
        results = []
        for file in batch_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            while img.nbytes > 1_000_000:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            proc = preprocess_image(gray, invert)
            h, w = gray.shape

            # Reuse locked ROI
            x0, y0, r = extract_circle_geometry(st.session_state.locked_circle_obj)
            yy, xx = np.ogrid[:h, :w]
            circle_mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= r ** 2

            features = detect_features(proc, diameter, minmass, separation, confidence)
            if features is None or features.empty:
                features = pd.DataFrame(columns=["x", "y"])

            fx = features["x"].astype(int)
            fy = features["y"].astype(int)
            inside = circle_mask[fy.clip(0, h-1), fx.clip(0, w-1)]
            inside_features = features[inside]

            count = len(inside_features)
            results.append((file.name, count))

        df = pd.DataFrame(results, columns=["image_name", "plaque_count"])
        st.markdown("### Batch Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "plaque_counts.csv", "text/csv")

