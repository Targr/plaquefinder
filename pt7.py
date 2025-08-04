import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Touch-Friendly Plaque Counter")

# Upload and parameter UI
mobile_file = st.file_uploader("Take a photo of a dish", type=["png", "jpg", "jpeg"])
advanced = st.checkbox("Advanced Settings")

invert = st.checkbox("Invert image", value=True) if advanced else False
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1) if advanced else None

# Helper functions
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

# Init persistent ellipse
if "ellipse_state" not in st.session_state:
    st.session_state.ellipse_state = None

if mobile_file:
    # Load image and shrink if needed
    file_bytes = np.asarray(bytearray(mobile_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    canvas_w, canvas_h = pil_image.size

    # Set ellipse default if not yet initialized
    if st.session_state.ellipse_state is None:
        st.session_state.ellipse_state = {
            "type": "ellipse",
            "left": canvas_w // 4,
            "top": canvas_h // 4,
            "width": canvas_w // 2,
            "height": canvas_h // 2,
            "fill": "rgba(255,255,255,0)",
            "stroke": "#FF0000",
            "strokeWidth": 3,
            "angle": 0
        }

    # Detect plaques
    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    # Parse ellipse center/radii
    ellipse = st.session_state.ellipse_state
    cx = ellipse["left"] + ellipse["width"] / 2
    cy = ellipse["top"] + ellipse["height"] / 2
    rx = ellipse["width"] / 2
    ry = ellipse["height"] / 2

    # Draw canvas objects
    objects = [ellipse] + [
        {
            "type": "circle",
            "left": float(row["x"]) - 3,
            "top": float(row["y"]) - 3,
            "radius": 3,
            "fill": "rgba(0,255,0,0.9)",
            "stroke": "#003300",
            "strokeWidth": 1,
            "selectable": False
        }
        for _, row in features.iterrows()
    ]

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=pil_image,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="transform",
        initial_drawing={"version": "4.4.0", "objects": objects},
        key="ellipse_canvas"
    )

    # Capture new ellipse only if changed
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if obj.get("type") == "ellipse":
                st.session_state.ellipse_state = obj
                break

    # Re-parse ellipse after potential update
    ellipse = st.session_state.ellipse_state
    cx = ellipse["left"] + ellipse["width"] / 2
    cy = ellipse["top"] + ellipse["height"] / 2
    rx = ellipse["width"] / 2
    ry = ellipse["height"] / 2

    # Count features inside ellipse
    inside_features = ellipse_mask_filter(features, cx, cy, rx, ry)

    st.markdown("### Live Plaque Count")
    st.success(f"{len(inside_features)} plaques detected inside selected region")
