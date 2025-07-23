import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from skimage.feature import blob_log

st.set_page_config(layout="wide")

st.title("Interactive Plaque Counter")

# === Image Uploader ===
uploaded_files = st.file_uploader("Upload plaque images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
invert = st.checkbox("Invert image", value=True)
contrast = st.checkbox("Apply contrast stretch", value=True)

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
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return img

def detect_features(gray_img, diameter):
    norm_img = (gray_img / 255.0).astype(np.float32)
    blobs = blob_log(norm_img, min_sigma=diameter/4, max_sigma=diameter/2, num_sigma=10, threshold=0.03)
    positions = blobs[:, :2][:, ::-1] if len(blobs) else np.empty((0, 2))
    return positions

# === Main Loop ===
if uploaded_files:
    diameter = st.slider("Feature Diameter", 5, 50, 15, 1)
    draw_mode = st.selectbox("Drawing mode", ["transform", "circle", "rect", "line", "freedraw", "polygon"])

    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proc = preprocess_image(gray, invert, contrast)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        st.subheader(file.name)
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_width=2,
            stroke_color="green",
            background_image=pil_img,
            update_streamlit=True,
            height=img.shape[0],
            width=img.shape[1],
            drawing_mode=draw_mode,
            key=f"canvas_{file.name}"
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
                    mask = dist <= r**2

        features = detect_features(proc, diameter)
        inside = []
        for f in features:
            x, y = int(f[0]), int(f[1])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                inside.append(f)
        inside = np.array(inside)

        st.image(proc, caption=f"Preprocessed: {file.name}", use_column_width=True, channels="GRAY")
        st.write(f"Detected plaques inside dish: {len(inside)}")

        # Update CSV log
        st.session_state.plaque_log = st.session_state.plaque_log[st.session_state.plaque_log.image_title != file.name]
        st.session_state.plaque_log.loc[len(st.session_state.plaque_log)] = [file.name, len(inside)]

    # === CSV Download ===
    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
