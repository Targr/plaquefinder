import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Plaque Counter with Locked ROI - Pixel-Perfect Fidelity")

# Upload + parameters
uploaded_file = st.file_uploader("Upload a petri dish photo", type=["png", "jpg", "jpeg"])
advanced = st.checkbox("Advanced Settings")

invert = st.checkbox("Invert image", value=True) if advanced else False
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1) if advanced else None

# Session state
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
    """Extract exact circle center and radius in pixels from canvas object, including scale."""
    raw_r = obj.get("radius", 50)
    left = obj.get("left", 0)
    top = obj.get("top", 0)
    scale_x = obj.get("scaleX", 1.0)
    scale_y = obj.get("scaleY", 1.0)
    # Average scale in x and y directions for radius scaling
    scale = (scale_x + scale_y) / 2
    r = raw_r * scale
    x = left + raw_r  # center x in pixels (relative to canvas)
    y = top + raw_r   # center y in pixels
    return x, y, r

if uploaded_file:
    # Load original image bytes, do NOT resize for fidelity
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Work with original resolution to maintain fidelity:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    h, w = gray.shape

    overlay_objects = []

    # Editable or locked circle
    if st.session_state.locked_circle_obj is None or st.session_state.edit_mode:
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
            circle_obj = st.session_state.locked_circle_obj.copy()
            circle_obj["selectable"] = True

        overlay_objects.append(circle_obj)
    else:
        circle_obj = st.session_state.locked_circle_obj.copy()
        circle_obj["selectable"] = False
        overlay_objects.append(circle_obj)

    # Extract exact pixel geometry from circle object (canvas coords)
    x0, y0, r = extract_circle_geometry(circle_obj)

    # Detect plaques on full image with no masking for fidelity
    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    # Post-filter to only those strictly inside the EXACT circle shown on canvas
    dx = features['x'] - x0
    dy = features['y'] - y0
    inside = (dx ** 2 + dy ** 2) <= r ** 2
    inside_features = features[inside]

    # Add green dots exactly where features are detected inside the circle
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

    # Draw the canvas at the exact original resolution and size
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

    # Lock button is always available in editing mode
    if st.session_state.locked_circle_obj is None or st.session_state.edit_mode:
        if st.button("Done (Lock Circle)"):
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "circle":
                    st.session_state.locked_circle_obj = obj
                    st.session_state.edit_mode = False
                    st.experimental_rerun()

    # Display plaque count strictly inside the exact visible ROI
    st.markdown("### Plaque Count Inside Circle")
    st.success(f"{len(inside_features)} plaques detected inside ROI")
