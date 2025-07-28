import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp
import io
from skimage import exposure, filters, morphology, measure

st.set_page_config(layout="wide")
st.title("Colony & Plaque Counter (Smart Contrast with Adaptive Masking)")

# === Image Upload ===
uploaded_files = st.file_uploader("Upload colony or plaque images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
image_type = st.radio("What are you detecting?", ["Plaques (dark spots)", "Colonies (light spots)"])

# === User-friendly Enhancement Options ===
st.subheader("Basic Image Adjustments")
invert = st.checkbox("Invert image", value=(image_type == "Plaques (dark spots)"))
adjust_contrast = st.checkbox("Auto contrast stretch", value=True)
gamma = st.slider("Gamma Correction", 0.1, 3.0, 1.0, 0.1)

st.subheader("Detection Parameters")
diameter = st.slider("Estimated Size of Colony/Plaque (px)", 5, 101, 15, 2)
minmass = st.slider("Ignore faint detections (sensitivity)", 1, 500, 30, 1)
separation = st.slider("Minimum distance between spots (px)", 1, 50, 5, 1)
confidence = st.slider("Detection Strictness (%)", 0, 100, 90, 1)


# === Utility Functions ===
def preprocess_image(gray, invert, contrast, gamma):
    if contrast:
        gray = exposure.rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)
    if gamma != 1.0:
        gray = exposure.adjust_gamma(gray, gamma)
    if invert:
        gray = cv2.bitwise_not(gray)
    return gray


def enhance_colony_features(gray):
    # Adaptive enhancement pipeline for colonies
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, np.ones((diameter, diameter), np.uint8))
    eq = cv2.equalizeHist(tophat)
    return eq


def locate_features(gray, mode):
    normed = (gray / 255.0).astype(np.float32)
    if mode == "Colonies (light spots)":
        enhanced = enhance_colony_features(gray)
        normed = (enhanced / 255.0).astype(np.float32)
    try:
        features = tp.locate(normed, diameter=diameter, minmass=minmass,
                             separation=separation, percentile=confidence,
                             invert=False)
    except Exception:
        features = pd.DataFrame(columns=["x", "y"])
    return features


# === Main App ===
if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_name = st.selectbox("Select image", file_names)
    selected_file = next(file for file in uploaded_files if file.name == selected_name)

    file_bytes = bytearray(selected_file.read())
    img_np = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Failed to load image.")
        st.stop()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert, adjust_contrast, gamma)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(proc, caption="Processed Grayscale Input", use_column_width=True, channels="GRAY")

    features = locate_features(proc, image_type)

    overlay = image_rgb.copy()
    if not features.empty:
        for _, row in features.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            cv2.circle(overlay, (x, y), diameter // 2, (0, 255, 0), 1)
            cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)
    
        cv2.putText(overlay, f"Count: {len(features)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(overlay, "No features detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    st.image(overlay, caption="Detected Colonies/Plaques", use_column_width=True)

    # Download CSV
    df = features[["x", "y"]].copy()
    df["image_title"] = selected_name
    df["object"] = image_type
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Detected Coordinates CSV", data=csv,
                       file_name="detections.csv", mime="text/csv")
