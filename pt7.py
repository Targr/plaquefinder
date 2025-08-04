import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Touch-Friendly Plaque Counter")

# Upload + parameters
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

if mobile_file:
    # Load and resize image if needed
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

    # Detect plaques
    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    # Determine whether to draw or transform
    ellipse_object = None
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=pil_image,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="transform",  # may be updated below
        key="main_canvas"
    )

    # Check if an ellipse has been drawn
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if obj.get("type") == "ellipse":
                ellipse_object = obj
                break

    # If no ellipse: let user draw one
    if ellipse_object is None:
        st.warning("Draw an ellipse to select a region of interest.")
        draw_result = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=pil_image,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="ellipse",
            key="draw_canvas"
        )
        st.stop()

    # Ellipse exists — extract coordinates
    cx = ellipse_object["left"] + ellipse_object["width"] / 2
    cy = ellipse_object["top"] + ellipse_object["height"] / 2
    rx = ellipse_object["width"] / 2
    ry = ellipse_object["height"] / 2

    # Filter features within the ellipse
    inside_features = ellipse_mask_filter(features, cx, cy, rx, ry)

    # Overlay ellipse + green plaques
    overlay_objects = [
        ellipse_object
    ] + [
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
        for _, row in inside_features.iterrows()
    ]

    # Display overlay
    st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#00FF00",
        background_image=pil_image,
        update_streamlit=False,
        height=canvas_h,
        width=canvas_w,
        initial_drawing={"version": "4.4.0", "objects": overlay_objects},
        drawing_mode="transform",
        key="overlay_canvas"
    )

    # Display count
    st.markdown("### Live Plaque Count")
    st.success(f"{len(inside_features)} plaques detected inside selected region")
