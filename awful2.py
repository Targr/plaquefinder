# streamlit app for user-defined multi-plate detection and per-plate counting with color classification

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import trackpy as tp
import json
import io
import base64
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("User-Controlled Multi-Plate Plaque/Colony Counter with Color Classification")

uploaded_file = st.file_uploader("Upload a petri dish image", type=["png", "jpg", "jpeg", "tiff", "tif"])
advanced = st.checkbox("Advanced Settings")
mode = st.radio("Feature Type", options=["Colony", "Plaque"], index=1, horizontal=True)
invert = mode == "Colony"
slider_kwargs = dict(label_visibility="visible" if advanced else "visible")
diameter = st.slider("Feature Diameter (px)", 5, 51, 15, 2, **slider_kwargs)
minmass = st.slider("Minimum Mass (signal:noise)", 1, 100, 10, 1, **slider_kwargs)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1, **slider_kwargs)
separation = st.slider("Minimum Separation (px)", 1, 30, 5, 1, **slider_kwargs) if advanced else None
color_clusters = st.slider("Number of Color Groups", 1, 5, 2, 1)

if "user_rois" not in st.session_state:
    st.session_state.user_rois = []
if "current_roi_index" not in st.session_state:
    st.session_state.current_roi_index = 0
if "plate_params" not in st.session_state:
    st.session_state.plate_params = {}

@st.cache_data(show_spinner=False)
def subtract_background(plate_img):
    bg = cv2.medianBlur(plate_img, 51)
    result = cv2.subtract(plate_img, bg)
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

def detect_features(gray_img, diameter, minmass, separation, confidence):
    norm_img = (gray_img / 255.0).astype(np.float32)
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

def classify_by_color(orig_img, features, n_clusters):
    if features.empty:
        return features, {}
    pixels = []
    for _, row in features.iterrows():
        x, y = int(row['x']), int(row['y'])
        rgb = orig_img[y, x, :3]
        pixels.append(rgb)
    pixels = np.array(pixels)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(pixels)
    features['color_group'] = labels
    color_counts = pd.Series(labels).value_counts().to_dict()
    return features, color_counts

def draw_overlays(image, features):
    draw = ImageDraw.Draw(image)
    for _, row in features.iterrows():
        x, y = row['x'], row['y']
        draw.ellipse([(x-5, y-5), (x+5, y+5)], outline='lime', width=2)
    return image

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
    return int(x), int(y), int(r)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    h, w = img.shape[:2]

    st.markdown("### Draw ROIs for Each Plate")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=pil_img,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="circle",
        key="canvas_roi"
    )

    if st.button("Save ROI"):
        if canvas_result.json_data:
            for obj in canvas_result.json_data.get("objects", []):
                if obj["type"] == "circle":
                    st.session_state.user_rois.append(obj)
            st.rerun()

    if st.session_state.user_rois:
        st.markdown("### Select Plate to Analyze")
        plate_index = st.selectbox("Plate Index", options=list(range(1, len(st.session_state.user_rois) + 1)))
        circle_obj = st.session_state.user_rois[plate_index - 1]
        x, y, r = extract_circle_geometry(circle_obj)
        x0, x1 = max(0, x - r), min(w, x + r)
        y0, y1 = max(0, y - r), min(h, y + r)

        crop = img[y0:y1, x0:x1]
        orig_crop = crop.copy()
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_subtracted = subtract_background(gray_crop)

        st.subheader(f"Plate {plate_index} View")
        st.image(crop, caption="Original Plate Region", use_column_width=True)

        features = detect_features(gray_subtracted, diameter, minmass, separation, confidence)
        features["x"] += x0
        features["y"] += y0
        features, color_counts = classify_by_color(orig_crop, features, color_clusters)

        overlay_img = draw_overlays(Image.fromarray(rgb.copy()), features)
        st.image(overlay_img, caption="Detected Features", use_column_width=True)

        st.success(f"{len(features)} features detected in Plate {plate_index}")
        if color_counts:
            st.markdown("### Colony Counts by Color")
            for label, count in sorted(color_counts.items()):
                st.write(f"Color Group {label + 1}: {count} colonies")

        st.session_state.plate_params[plate_index] = {
            "diameter": diameter,
            "minmass": minmass,
            "confidence": confidence,
            "separation": separation,
            "color_clusters": color_clusters
        }

        st.markdown("### Export Parameters")
        param_json = json.dumps(st.session_state.plate_params, indent=2)
        st.download_button("Download Parameters as JSON", data=param_json, file_name="plate_parameters.json", mime="application/json")

    if st.button("Reset All ROIs"):
        st.session_state.user_rois = []
        st.session_state.plate_params = {}
        st.rerun()
