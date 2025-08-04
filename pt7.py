import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Plaque Counter with Multiple Locked ROIs")

# Upload + parameters
uploaded_file = st.file_uploader("Upload a petri dish photo", type=["png", "jpg", "jpeg"])
advanced = st.checkbox("Advanced Settings")

invert = st.checkbox("Invert image", value=True) if advanced else False
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1) if advanced else None

# Session state
if "locked_circles" not in st.session_state:
    st.session_state.locked_circles = []
if "current_edit_circle" not in st.session_state:
    st.session_state.current_edit_circle = None

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
    x = left + raw_r
    y = top + raw_r
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

    # Add all locked ROIs (non-editable)
    all_inside_features = []
    for locked in st.session_state.locked_circles:
        locked_obj = locked.copy()
        locked_obj["selectable"] = False
        overlay_objects.append(locked_obj)

        x0, y0, r = extract_circle_geometry(locked_obj)
        features = detect_features(proc, diameter, minmass, separation, confidence)
        if not features.empty:
            dx = features['x'] - x0
            dy = features['y'] - y0
            inside = (dx ** 2 + dy ** 2) <= r ** 2
            inside_features = features[inside]
            all_inside_features.append((x0, y0, r, inside_features))

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

    # Editable ROI (only one at a time)
    if st.session_state.current_edit_circle is None:
        # Detect initial ROI if needed
        detected = detect_dish(gray)
        if detected is not None:
            x0, y0, r = detected
            new_circle = {
                "type": "circle",
                "left": float(x0 - r),
                "top": float(y0 - r),
                "radius": float(r),
                "fill": "rgba(255,255,255,0)",
                "stroke": "#FF0000",
                "strokeWidth": 3,
                "selectable": True
            }
            st.session_state.current_edit_circle = new_circle
    else:
        new_circle = st.session_state.current_edit_circle.copy()
        new_circle["selectable"] = True
        overlay_objects.append(new_circle)

    # Canvas
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

    # Lock button
    if st.button("Done (Lock Circle)"):
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "circle" and obj.get("selectable", False):
                st.session_state.locked_circles.append(obj)
                st.session_state.current_edit_circle = None
                st.experimental_rerun()

    # Display results
    st.markdown("### Total ROIs and Plaques Inside Each")
    if all_inside_features:
        for i, (x0, y0, r, inside_features) in enumerate(all_inside_features, start=1):
            st.success(f"ROI {i}: {len(inside_features)} plaques detected inside")
    else:
        st.info("Lock an ROI to begin counting plaques.")
