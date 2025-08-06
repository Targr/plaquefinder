# streamlit app for multi-plate detection and per-plate counting with color classification

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
st.title("Multi-Plate Plaque/Colony Counter with Color Classification")

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

# Store per-plate parameters
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

def detect_plates(gray):
    blurred = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0] // 6,
        param1=100,
        param2=30,
        minRadius=gray.shape[0] // 8,
        maxRadius=gray.shape[0] // 2
    )
    if circles is not None:
        return np.uint16(np.around(circles[0]))
    return []

def draw_overlays(image, features):
    draw = ImageDraw.Draw(image)
    for _, row in features.iterrows():
        x, y = row['x'], row['y']
        draw.ellipse([(x-5, y-5), (x+5, y+5)], outline='lime', width=2)
    return image

def show_plate_selector(pil_image, circles):
    preview = pil_image.copy()
    draw = ImageDraw.Draw(preview)
    for idx, (x, y, r) in enumerate(circles):
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="red", width=3)
        draw.text((x - 10, y - 10), str(idx + 1), fill="white")
    st.image(preview, caption="Click a plate to tune individually", use_column_width=True)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_plates = detect_plates(gray)

    if len(detected_plates) == 0:
        st.error("No plates detected. Try a clearer image.")
        st.stop()

    plate_index = st.selectbox("Select Plate to Analyze", options=list(range(1, len(detected_plates) + 1)), format_func=lambda x: f"Plate {x}")
    selected_plate = detected_plates[plate_index - 1]
    x, y, r = selected_plate

    x0, x1 = max(0, x - r), min(img.shape[1], x + r)
    y0, y1 = max(0, y - r), min(img.shape[0], y + r)

    plate_crop = img[y0:y1, x0:x1]
    orig_crop = plate_crop.copy()
    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    plate_gray = subtract_background(plate_gray)

    st.subheader(f"Plate {plate_index} View")
    st.image(plate_crop, caption="Original Plate Region", use_column_width=True)

    plate_features = detect_features(plate_gray, diameter, minmass, separation, confidence)
    plate_features["x"] += x0
    plate_features["y"] += y0
    plate_features, color_counts = classify_by_color(orig_crop, plate_features, color_clusters)

    overlay_img = draw_overlays(Image.fromarray(rgb.copy()), plate_features)
    st.image(overlay_img, caption="Detected Features", use_column_width=True)

    st.success(f"{len(plate_features)} features detected in Plate {plate_index}")
    if color_counts:
        st.markdown("### Colony Counts by Color")
        for label, count in sorted(color_counts.items()):
            st.write(f"Color Group {label + 1}: {count} colonies")

    # Save parameters for this plate
    st.session_state.plate_params[plate_index] = {
        "diameter": diameter,
        "minmass": minmass,
        "confidence": confidence,
        "separation": separation,
        "color_clusters": color_clusters
    }

    st.markdown("---")
    st.markdown("### All Plate Overview")
    show_plate_selector(pil_img, detected_plates)

    st.markdown("### Export Parameters")
    param_json = json.dumps(st.session_state.plate_params, indent=2)
    st.download_button("Download Parameters as JSON", data=param_json, file_name="plate_parameters.json", mime="application/json")
