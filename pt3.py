import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Interactive Plaque Counter (Canvas-Aligned)")

# === Image Upload ===
uploaded_files = st.file_uploader("Upload plaque images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
invert = st.checkbox("Invert image", value=True)
contrast = st.checkbox("Apply contrast stretch", value=True)

# Detection parameters
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)

# Global log
def reset_log():
    return pd.DataFrame(columns=["image_title", "num_plaques"])

if "plaque_log" not in st.session_state:
    st.session_state.plaque_log = reset_log()

# === Utility Functions ===
def preprocess_image(img, invert=False, contrast=False):
    if invert:
        img = cv2.bitwise_not(img)
    if contrast:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def detect_features(gray_img, diameter, minmass, separation, confidence):
    norm_img = (gray_img / 255.0).astype(np.float32)
    try:
        features = tp.locate(
            norm_img,
            diameter=diameter,
            minmass=minmass,
            separation=separation,
            percentile=confidence,
            invert=False
        )
    except Exception:
        features = pd.DataFrame()
    return features

def resize_with_scale(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    return image, 1.0

def find_plate_regions(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray_img.shape[0]//4,
        param1=100, param2=60, minRadius=gray_img.shape[0]//10, maxRadius=0
    )
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        return [(x, y, r) for x, y, r in circles]
    return []

# === Main App ===
if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_name = st.selectbox("Select image", file_names)
    selected_file = next(file for file in uploaded_files if file.name == selected_name)

    # Load + preprocess
    file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert, contrast)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for canvas preview
    canvas_bg_resized, canvas_scale = resize_with_scale(image_rgb)

    draw_mode = st.selectbox("Drawing mode", ["transform", "rect"])
    st.subheader(selected_name)

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="green",
        background_image=Image.fromarray(canvas_bg_resized),
        update_streamlit=True,
        height=canvas_bg_resized.shape[0],
        width=canvas_bg_resized.shape[1],
        drawing_mode=draw_mode,
        key=f"canvas_{selected_name}"
    )

    display_overlay = image_rgb.copy()
    total_features = 0
    regions = []

    # Detect circular plate regions
    plate_circles = find_plate_regions(gray)
    for x, y, r in plate_circles:
        pad_r = int(r * 0.85)
        x1, y1 = max(0, x - pad_r), max(0, y - pad_r)
        x2, y2 = min(proc.shape[1], x + pad_r), min(proc.shape[0], y + pad_r)
        regions.append((x1, y1, x2, y2))
        cv2.circle(display_overlay, (x, y), pad_r, (255, 0, 0), 2)

    # Add user rectangles
    if canvas_result.json_data and canvas_result.json_data["objects"]:
        for rect in canvas_result.json_data["objects"]:
            if rect["type"] == "rect":
                left = rect["left"] / canvas_scale
                top = rect["top"] / canvas_scale
                width = rect["width"] / canvas_scale
                height = rect["height"] / canvas_scale

                x1 = int(round(left))
                y1 = int(round(top))
                x2 = int(round(left + width))
                y2 = int(round(top + height))

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(proc.shape[1], x2), min(proc.shape[0], y2)

                regions.append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in regions:
        roi_img = proc[y1:y2, x1:x2].copy()
        crop_bounds = (x1, y1)

        features = detect_features(roi_img, diameter, minmass, separation, confidence)
        if features is None or features.empty:
            features = pd.DataFrame(columns=["x", "y"])

        total_features += len(features)

        for _, row in features.iterrows():
            x, y = int(round(row["x"] + crop_bounds[0])), int(round(row["y"] + crop_bounds[1]))
            cv2.circle(display_overlay, (x, y), diameter // 2, (0, 255, 0), 1)
            cv2.circle(display_overlay, (x, y), 2, (255, 0, 0), -1)

    display_overlay_resized, _ = resize_with_scale(display_overlay)
    st.image(display_overlay_resized, caption=f"Detected plaques: {total_features}")

    st.session_state.plaque_log = st.session_state.plaque_log[st.session_state.plaque_log.image_title != selected_name]
    st.session_state.plaque_log.loc[len(st.session_state.plaque_log)] = [selected_name, total_features]

    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
