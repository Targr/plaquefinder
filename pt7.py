import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Automatic Plaque Counter with Dish Detection")

# Upload + parameters
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
        return circles[0][0]  # take first detected circle (x, y, r)
    return None

def mask_features_in_circle(features, x0, y0, r):
    dx = features['x'] - x0
    dy = features['y'] - y0
    inside = (dx ** 2 + dy ** 2) <= r ** 2
    return features[inside]

if mobile_file:
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

    # Step 1: detect dish automatically
    detected_circle = detect_dish(gray)
    if detected_circle is None:
        st.error("Could not automatically detect dish. Please retake the photo with a clearer circle.")
        st.stop()

    x0, y0, r = detected_circle
    st.success(f"Detected dish at x={x0}, y={y0}, radius={r}")

    # Step 2: detect features
    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    inside_features = mask_features_in_circle(features, x0, y0, r)

    # Step 3: draw overlay
    overlay_objects = [
        {
            "type": "circle",
            "left": float(x0 - r),
            "top": float(y0 - r),
            "radius": float(r),
            "fill": "rgba(255,255,255,0)",
            "stroke": "#FF0000",
            "strokeWidth": 3,
            "selectable": False
        }
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

    # Step 4: show result
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
        key="ai_overlay"
    )

    st.markdown("### Live Plaque Count (Auto-Detected Region)")
    st.success(f"{len(inside_features)} plaques detected inside dish")
