import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp
import io
from skimage import exposure

st.set_page_config(layout="wide")
st.title("Colony & Plaque Counter (Fast Mode)")

# === Image Upload ===
uploaded_files = st.file_uploader("Upload colony or plaque images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
image_type = st.radio("What are you detecting?", ["Plaques (dark spots)", "Colonies (light spots)"])

# === Basic Image Adjustments ===
st.subheader("Image Adjustments")
invert = st.checkbox("Invert image", value=(image_type == "Plaques (dark spots)"))
adjust_contrast = st.checkbox("Auto contrast stretch", value=True)
gamma = st.slider("Gamma Correction", 0.1, 3.0, 1.0, 0.1)

st.subheader("Detection Parameters")
diameter = st.slider("Estimated Size (px)", 5, 101, 15, 2)
minmass = st.slider("Ignore faint detections", 1, 500, 30, 1)
separation = st.slider("Min distance between spots (px)", 1, 50, 5, 1)
confidence = st.slider("Detection Strictness (%)", 0, 100, 90, 1)
num_dishes = st.slider("Number of dishes to detect", 1, 10, 1, 1)

# === Utility Functions ===
def preprocess_image(gray):
    if adjust_contrast:
        gray = exposure.rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)
    if gamma != 1.0:
        look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
        gray = cv2.LUT(gray, look_up)
    if invert:
        gray = cv2.bitwise_not(gray)
    return gray

def fast_enhance(gray):
    return cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0))

def locate_features(gray):
    normed = (gray / 255.0).astype(np.float32)
    try:
        return tp.locate(normed, diameter=diameter, minmass=minmass,
                         separation=separation, percentile=confidence, invert=False)
    except Exception:
        return pd.DataFrame(columns=["x", "y"])

def detect_dishes(gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=gray.shape[0] // (num_dishes + 1),
                               param1=50, param2=30,
                               minRadius=int(min(gray.shape[:2]) * 0.2),
                               maxRadius=int(min(gray.shape[:2]) * 0.6))
    return np.round(circles[0, :num_dishes]).astype("int") if circles is not None else []

def ellipse_mask(features, cx, cy, rx):
    dx, dy = features['x'] - cx, features['y'] - cy
    return features[(dx / rx) ** 2 + (dy / rx) ** 2 <= 1]

# === Main App ===
if uploaded_files:
    name_list = [f.name for f in uploaded_files]
    chosen_name = st.selectbox("Choose image", name_list)
    chosen_file = next(f for f in uploaded_files if f.name == chosen_name)

    data = bytearray(chosen_file.read())
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not load image.")
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prepped = preprocess_image(gray)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(prepped, caption="Processed Input", use_column_width=True, channels="GRAY")

    dishes = detect_dishes(gray)
    output_data = []

    if len(dishes) > 0:
        st.subheader(f"{len(dishes)} Dish(es) Found")
        for i, (cx, cy, cr) in enumerate(dishes):
            m = int(cr * 1.1)
            x1, x2 = max(0, cx - m), min(image.shape[1], cx + m)
            y1, y2 = max(0, cy - m), min(image.shape[0], cy + m)
            crop = image[y1:y2, x1:x2]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            proc = preprocess_image(crop_gray)
            if image_type == "Colonies (light spots)":
                proc = fast_enhance(proc)

            overlay = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            cx_local, cy_local = cx - x1, cy - y1
            rx = int(cr * 0.95)
            feats = locate_features(proc)
            feats = ellipse_mask(feats, cx_local, cy_local, rx)

            cv2.ellipse(overlay, (cx_local, cy_local), (rx, rx), 0, 0, 360, (255, 0, 0), 2)
            for _, r in feats.iterrows():
                px, py = int(r["x"]), int(r["y"])
                cv2.circle(overlay, (px, py), diameter // 2, (0, 255, 0), 1)
                cv2.circle(overlay, (px, py), 2, (255, 0, 0), -1)

            count = len(feats)
            cv2.putText(overlay, f"Dish {i+1}: {count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            st.image(overlay, caption=f"Dish {i+1}: {count} {image_type.lower()}", use_column_width=True)

            output_data.append({"image_title": f"{chosen_name} (Dish {i+1})", "dish_id": f"Dish {i+1}", "num_objects": count})
    else:
        feats = locate_features(prepped)
        if image_type == "Colonies (light spots)":
            feats = locate_features(fast_enhance(prepped))

        if feats.empty:
            feats = pd.DataFrame(columns=["x", "y"])

        overlay = rgb_img.copy()
        for _, r in feats.iterrows():
            px, py = int(r["x"]), int(r["y"])
            cv2.circle(overlay, (px, py), diameter // 2, (0, 255, 0), 1)
            cv2.circle(overlay, (px, py), 2, (255, 0, 0), -1)

        count = len(feats)
        cv2.putText(overlay, f"Detected: {count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        st.image(overlay, caption=f"Detected {image_type.lower()}: {count}", use_column_width=True)
        output_data.append({"image_title": chosen_name, "dish_id": "Whole Image", "num_objects": count})

    df = pd.DataFrame(output_data)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Count CSV", data=csv, file_name="counts.csv", mime="text/csv")
