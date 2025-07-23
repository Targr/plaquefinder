import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
import tempfile
from PIL import Image

# === Configuration ===
SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")

st.title("Bacterial Plaque Counter")

# === Upload Folder ===
st.sidebar.header("1. Upload Image Folder")
uploaded_folder = st.sidebar.file_uploader("Upload multiple image files", accept_multiple_files=True, type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_folder:
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    for uploaded_file in uploaded_folder:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        image_paths.append(temp_path)
else:
    st.warning("Upload image files to begin.")
    st.stop()

# === Sidebar Parameters ===
st.sidebar.header("2. Detection Parameters")
diameter = st.sidebar.slider("Diameter", 5, 51, 15, step=2)
minmass = st.sidebar.slider("Min Mass", 1, 50, 1)
separation = st.sidebar.slider("Separation", 1, 30, 5)
percentile = st.sidebar.slider("Percentile", 0.0, 100.0, 64.0)
invert = st.sidebar.checkbox("Invert Image", True)
contrast = st.sidebar.checkbox("Contrast Stretch", True)
dish_width_scale = st.sidebar.slider("Dish Width Scale", 0.5, 1.5, 1.0)
dish_height_scale = st.sidebar.slider("Dish Height Scale", 0.5, 1.5, 1.0)
dish_rotation = st.sidebar.slider("Dish Rotation (degrees)", 0, 180, 0)

# === Detection Helpers ===
def preprocess_image(gray, invert, contrast):
    img = gray.copy()
    if contrast:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    if invert:
        img = cv2.bitwise_not(img)
    return img

def detect_dish_edge(gray):
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30,
        minRadius=int(min(gray.shape[:2]) * 0.4),
        maxRadius=int(min(gray.shape[:2]) * 0.6)
    )
    if circles is not None:
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    return None

def ellipse_mask_filter(features, cx, cy, rx, ry, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    dx = features[:, 0] - cx
    dy = features[:, 1] - cy
    x_rot = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
    y_rot = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    inside = (x_rot / rx)**2 + (y_rot / ry)**2 <= 1
    return features[inside]

def detect_features(gray_img):
    from skimage.feature import blob_log
    norm_img = (gray_img / 255.0).astype(np.float32)
    blobs = blob_log(norm_img, min_sigma=diameter/4, max_sigma=diameter/2, num_sigma=3, threshold=minmass/255.0)
    positions = blobs[:, :2][:, ::-1] if len(blobs) else np.empty((0, 2))
    return positions

# === Detection Loop ===
results = []
for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert, contrast)
    features = detect_features(proc)

    dish = detect_dish_edge(gray)
    if dish:
        cx, cy, cr = dish
        rx = cr * dish_width_scale * 0.95
        ry = cr * dish_height_scale * 0.95
        features = ellipse_mask_filter(features, cx, cy, rx, ry, dish_rotation)

    count = len(features)
    results.append({"image_title": os.path.basename(path), "num_plaques": count})

    # === Display Preview ===
    preview = img.copy()
    if dish:
        cv2.ellipse(preview, (cx, cy), (int(rx), int(ry)), dish_rotation, 0, 360, (255, 0, 0), 2)
    for pt in features:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(preview, (x, y), 5, (0, 255, 0), 1)
    st.image(preview, caption=f"{os.path.basename(path)} — Count: {count}", channels="BGR")

# === Download CSV ===
df = pd.DataFrame(results)
st.download_button("Download CSV", data=df.to_csv(index=False), file_name="plaque_counts.csv", mime="text/csv")
