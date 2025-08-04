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
uploaded_file = st.file_uploader("Upload a petri dish photo", type=["png", "jpg", "jpeg"])
advanced = st.checkbox("Advanced Settings")

invert = st.checkbox("Invert image", value=True) if advanced else False
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1) if advanced else None

# Session state
if "locked_circle" not in st.session_state:
    st.session_state.locked_circle = None
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

def mask_features_in_circle(features, x0, y0, r):
    dx = features['x'] - x0
    dy = features['y'] - y0
    inside = (dx ** 2 + dy ** 2) <= r ** 2
    return features[inside]

def canvas_to_circle_data(objects):
    for obj in objects:
        if obj["type"] == "circle":
            r = obj.get("radius", 50)
            x = obj.get("left", 0) + r
            y = obj.get("top", 0) + r
            return (x, y, r)
    return None

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

    # Step 1: Set editable or locked circle
    if st.session_state.locked_circle is None:
        if st.session_state.edit_mode:
            st.info("Adjust the circle and press 'Done' to lock it.")
        # Detect initial circle if needed
        detected = detect_dish(gray)
        if detected is None:
            st.error("Could not detect dish. Please upload a clearer image.")
            st.stop()
        x0, y0, r = detected
    else:
        x0, y0, r = st.session_state.locked_circle

    # Step 2: Create overlay
    overlay_objects = []
    if st.session_state.locked_circle is None or st.session_state.edit_mode:
        # Add editable circle
        overlay_objects.append({
            "type": "circle",
            "left": float(x0 - r),
            "top": float(y0 - r),
            "radius": float(r),
            "fill": "rgba(255,255,255,0)",
            "stroke": "#FF0000",
            "strokeWidth": 3,
            "selectable": True
        })
        mode = "transform"
    else:
        # Add locked circle + green dots
        mode = "transform"  # canvas still needs this mode to avoid component error
        overlay_objects.append({
            "type": "circle",
            "left": float(x0 - r),
            "top": float(y0 - r),
            "radius": float(r),
            "fill": "rgba(255,255,255,0)",
            "stroke": "#FF0000",
            "strokeWidth": 3,
            "selectable": False
        })

    # Detect features
    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])
    inside_features = mask_features_in_circle(features, x0, y0, r)

    # Add green markers
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

    # Step 3: Draw canvas
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#00FF00",
        background_image=pil_img,
        update_streamlit=True,
        height=h,
        width=w,
        initial_drawing={"version": "4.4.0", "objects": overlay_objects},
        drawing_mode=mode,
        key="editable"
    )

    # Step 4: Lock button
    if st.session_state.locked_circle is None:
        if st.button("Done (Lock Circle)"):
            coords = canvas_to_circle_data(canvas_result.json_data["objects"])
            if coords is not None:
                st.session_state.locked_circle = coords
                st.session_state.edit_mode = False
                st.experimental_rerun()

    # Step 5: Show result
    st.markdown("### Plaque Count Inside Circle")
    st.success(f"{len(inside_features)} plaques detected inside ROI")

    # Debug info
    # st.write("Locked circle:", st.session_state.locked_circle)
