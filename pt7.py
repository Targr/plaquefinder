import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Touch-Friendly Plaque Counter")

mobile_file = st.file_uploader("Take a photo of a dish", type=["png", "jpg", "jpeg"])
advanced = st.checkbox("Advanced Settings")

invert = st.checkbox("Invert image", value=True) if advanced else False
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1) if advanced else None

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

def ellipse_mask_filter(features, cx, cy, rx, ry):
    dx = features['x'] - cx
    dy = features['y'] - cy
    inside = ((dx / rx) ** 2 + (dy / ry) ** 2) <= 1
    return features[inside]

# Session init
if "plaque_log" not in st.session_state:
    st.session_state.plaque_log = pd.DataFrame(columns=["image_title", "ellipse_id", "num_plaques"])
if "ellipse_initialized" not in st.session_state:
    st.session_state.ellipse_initialized = False
if "ellipse_state" not in st.session_state:
    st.session_state.ellipse_state = None  # Placeholder

if mobile_file:
    file_bytes = np.asarray(bytearray(mobile_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    canvas_w, canvas_h = pil_image.size

    # Detect features
    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    # Initialize ellipse once
    if not st.session_state.ellipse_initialized:
        st.session_state.ellipse_state = {
            "type": "ellipse",
            "left": canvas_w // 4,
            "top": canvas_h // 4,
            "rx": canvas_w // 6,
            "ry": canvas_h // 6,
            "fill": "rgba(255,255,255,0)",
            "stroke": "#FF0000",
            "strokeWidth": 3,
            "angle": 0
        }
        st.session_state.ellipse_initialized = True

    # Assemble canvas objects
    canvas_objects = [st.session_state.ellipse_state]
    for _, row in features.iterrows():
        canvas_objects.append({
            "type": "circle",
            "left": float(row["x"]) - 3,
            "top": float(row["y"]) - 3,
            "radius": 3,
            "fill": "rgba(50,255,50,0.9)",  # lime green
            "stroke": "#003300",
            "strokeWidth": 1,
            "selectable": False
        })

    # Draw the canvas — pass initial_drawing only once
    canvas_kwargs = dict(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=pil_image,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="transform",
        key="ellipse_canvas"
    )

    if not st.session_state.get("ellipse_used_once"):
        canvas_kwargs["initial_drawing"] = {"version": "4.4.0", "objects": canvas_objects}
        st.session_state["ellipse_used_once"] = True

    canvas_result = st_canvas(**canvas_kwargs)

    # Update ellipse only if user modified it
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        ellipses = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "ellipse"]
        if ellipses:
            st.session_state.ellipse_state = ellipses[-1]

    # Geometry from ellipse
    ellipse = st.session_state.ellipse_state
    left = ellipse.get("left", canvas_w // 4)
    top = ellipse.get("top", canvas_h // 4)
    rx = ellipse.get("rx", canvas_w // 6)
    ry = ellipse.get("ry", canvas_h // 6)
    width = ellipse.get("width", 2 * rx)
    height = ellipse.get("height", 2 * ry)
    cx = left + width / 2
    cy = top + height / 2

    # Count features in ellipse
    mask_feats = ellipse_mask_filter(features, cx, cy, rx, ry)

    # Live count
    st.markdown("### Live Plaque Count")
    st.write(f"Ellipse 1: **{len(mask_feats)} plaques detected**")

    # Update session log
    st.session_state.plaque_log = st.session_state.plaque_log[
        st.session_state.plaque_log.image_title != mobile_file.name
    ]
    st.session_state.plaque_log = pd.concat([
        st.session_state.plaque_log,
        pd.DataFrame([{
            "image_title": mobile_file.name,
            "ellipse_id": "Ellipse 1",
            "num_plaques": len(mask_feats)
        }])
    ], ignore_index=True)

    # CSV download
    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
