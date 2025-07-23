import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")

st.title("Interactive Plaque Counter")

# === Image Uploader ===
uploaded_files = st.file_uploader("Upload plaque images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
invert = st.checkbox("Invert image", value=True)
contrast = st.checkbox("Apply contrast stretch", value=True)
invert_detection = st.checkbox("Invert detection area logic", value=False)

# Detection parameters
diameter = st.slider("Feature Diameter", 5, 50, 15, 1)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)

# Global plaque log
def reset_log():
    return pd.DataFrame(columns=["image_title", "num_plaques"])

if "plaque_log" not in st.session_state:
    st.session_state.plaque_log = reset_log()

# === Utility functions ===
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

# === Single Image Viewer ===
if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_name = st.selectbox("Select image to analyze", file_names)
    selected_file = next(file for file in uploaded_files if file.name == selected_name)

    file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert, contrast)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    draw_mode = st.selectbox("Drawing mode", ["transform", "circle", "rect", "line", "freedraw", "polygon"])
    st.subheader(selected_name)

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="green",
        background_image=Image.fromarray(image_rgb),
        update_streamlit=True,
        height=img.shape[0],
        width=img.shape[1],
        drawing_mode=draw_mode,
        key=f"canvas_{selected_name}"
    )

    mask = np.ones(proc.shape, dtype=bool)
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "circle":
                cx = int(obj["left"] + obj["radius"])
                cy = int(obj["top"] + obj["radius"])
                r = int(obj["radius"])
                yy, xx = np.ogrid[:proc.shape[0], :proc.shape[1]]
                dist = (yy - cy)**2 + (xx - cx)**2
                if invert_detection:
                    mask = np.logical_and(mask, dist > r**2)
                else:
                    mask = np.logical_and(mask, dist <= r**2)

    features = detect_features(proc, diameter, minmass, separation, confidence)
    inside = []
    display_img = image_rgb.copy()
    if not features.empty:
        for _, row in features.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                inside.append((x, y))
                cv2.circle(display_img, (x, y), diameter // 2, (0, 255, 0), 1)
                cv2.circle(display_img, (x, y), 2, (255, 0, 0), -1)

    st.image(display_img, caption=f"Detected plaques: {len(inside)}", use_column_width=True)

    st.session_state.plaque_log = st.session_state.plaque_log[st.session_state.plaque_log.image_title != selected_name]
    st.session_state.plaque_log.loc[len(st.session_state.plaque_log)] = [selected_name, len(inside)]

    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
