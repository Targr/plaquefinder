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
num_dishes = st.slider("Number of dishes to detect", 1, 10, 1, 1)

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


def detect_multiple_dishes(gray, max_dishes):
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0] // (max_dishes + 1),
        param1=50, param2=30,
        minRadius=int(min(gray.shape[:2]) * 0.2),
        maxRadius=int(min(gray.shape[:2]) * 0.6)
    )
    if circles is not None:
        return np.round(circles[0, :max_dishes]).astype("int")
    return []


def ellipse_mask_filter(features, cx, cy, rx, ry, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    dx = features['x'] - cx
    dy = features['y'] - cy

    x_rot = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
    y_rot = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

    inside = (x_rot / rx)**2 + (y_rot / ry)**2 <= 1
    return features[inside]

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

    dish_circles = detect_multiple_dishes(gray, num_dishes)
    new_rows = []

    if len(dish_circles) > 0:
        st.subheader(f"Detected {len(dish_circles)} dish(es)")
        for i, (cx, cy, cr) in enumerate(dish_circles):
            margin = int(cr * 1.1)
            x1 = max(0, cx - margin)
            x2 = min(img.shape[1], cx + margin)
            y1 = max(0, cy - margin)
            y2 = min(img.shape[0], cy + margin)
            dish_crop = img[y1:y2, x1:x2]
            dish_gray = cv2.cvtColor(dish_crop, cv2.COLOR_BGR2GRAY)
            dish_proc = preprocess_image(dish_gray, invert, adjust_contrast, gamma)
            dish_overlay = cv2.cvtColor(dish_crop, cv2.COLOR_BGR2RGB)

            cx_adj = cx - x1
            cy_adj = cy - y1
            rx = ry = int(cr * 0.95)
            angle_deg = 0

            features = locate_features(dish_proc, image_type)
            if features is None or features.empty:
                features = pd.DataFrame(columns=["x", "y"])

            masked_feats = ellipse_mask_filter(features, cx_adj, cy_adj, rx, ry, angle_deg)

            cv2.ellipse(dish_overlay, (cx_adj, cy_adj), (rx, ry), angle_deg, 0, 360, (255, 0, 0), 2)
            for _, row in masked_feats.iterrows():
                x, y = int(round(row["x"])), int(round(row["y"]))
                cv2.circle(dish_overlay, (x, y), diameter // 2, (0, 255, 0), 1)
                cv2.circle(dish_overlay, (x, y), 2, (255, 0, 0), -1)

            count = len(masked_feats)
            cv2.putText(dish_overlay, f"Dish {i+1}: {count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            st.image(dish_overlay, caption=f"Dish {i+1}: {count} {image_type.lower()}", use_column_width=True)

            new_rows.append({
                "image_title": f"{selected_name} (Dish {i+1})",
                "dish_id": f"Dish {i+1}",
                "num_objects": count
            })
    else:
        features = locate_features(proc, image_type)
        if features is None or features.empty:
            features = pd.DataFrame(columns=["x", "y"])

        overlay = image_rgb.copy()
        for _, row in features.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            cv2.circle(overlay, (x, y), diameter // 2, (0, 255, 0), 1)
            cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)

        count = len(features)
        cv2.putText(overlay, f"Detected: {count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        st.image(overlay, caption=f"Detected {image_type.lower()}: {count}", use_column_width=True)

        new_rows.append({
            "image_title": selected_name,
            "dish_id": "Whole Image",
            "num_objects": count
        })

    df = pd.DataFrame(new_rows)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Count CSV", data=csv,
                       file_name="counts.csv", mime="text/csv")
