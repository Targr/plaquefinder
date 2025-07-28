import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp
import io

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
num_dishes = st.slider("Number of dishes to detect", 1, 10, 1, 1)

# Global log
def reset_log():
    return pd.DataFrame(columns=["image_title", "dish_id", "num_plaques"])

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

# === Main App ===
if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_name = st.selectbox("Select image", file_names)
    selected_file = next(file for file in uploaded_files if file.name == selected_name)

    # Load and optionally compress the image
    file_bytes = bytearray(selected_file.read())
    img_np = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert, contrast)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader(selected_name)

    dish_circles = detect_multiple_dishes(gray, num_dishes)
    new_rows = []

    if len(dish_circles) > 1:
        st.subheader(f"Detected {len(dish_circles)} dish(es) – Cropping and analyzing individually")

        for i, (cx, cy, cr) in enumerate(dish_circles):
            # Define crop bounds
            margin = int(cr * 1.1)
            x1 = max(0, cx - margin)
            x2 = min(img.shape[1], cx + margin)
            y1 = max(0, cy - margin)
            y2 = min(img.shape[0], cy + margin)

            # Crop image and preprocess
            dish_img = img[y1:y2, x1:x2]
            dish_gray = cv2.cvtColor(dish_img, cv2.COLOR_BGR2GRAY)
            dish_proc = preprocess_image(dish_gray, invert, contrast)

            # Detect features in the cropped image
            dish_features = detect_features(dish_proc, diameter, minmass, separation, confidence)
            if dish_features is None or dish_features.empty:
                dish_features = pd.DataFrame(columns=["x", "y"])

            # Create display image
            dish_overlay = cv2.cvtColor(dish_img, cv2.COLOR_BGR2RGB)

            # Draw all detected features
            for _, row in dish_features.iterrows():
                x, y = int(round(row["x"])), int(round(row["y"]))
                cv2.circle(dish_overlay, (x, y), diameter // 2, (0, 255, 0), 1)
                cv2.circle(dish_overlay, (x, y), 2, (255, 0, 0), -1)

            plaque_count = len(dish_features)
            cv2.putText(dish_overlay, f"Dish {i+1}: {plaque_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show result
            st.image(dish_overlay, caption=f"Dish {i+1}: {plaque_count} plaques", use_column_width=True)

            # Store count
            new_rows.append({
                "image_title": f"{selected_name} (Dish {i+1})",
                "dish_id": f"Dish {i+1}",
                "num_plaques": plaque_count
            })

    else:
        # Fallback: treat the whole image as one plate
        features = detect_features(proc, diameter, minmass, separation, confidence)
        if features is None or features.empty:
            features = pd.DataFrame(columns=["x", "y"])
        display_overlay = image_rgb.copy()
        for _, row in features.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            cv2.circle(display_overlay, (x, y), diameter // 2, (0, 255, 0), 1)
            cv2.circle(display_overlay, (x, y), 2, (255, 0, 0), -1)

        plaque_count = len(features)
        cv2.putText(display_overlay, f"Plaques: {plaque_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        st.image(display_overlay, caption=f"Detected plaques: {plaque_count}", use_column_width=True)

        new_rows.append({
            "image_title": selected_name,
            "dish_id": "Whole Image",
            "num_plaques": plaque_count
        })

    # Update session log
    st.session_state.plaque_log = st.session_state.plaque_log[
        ~((st.session_state.plaque_log.image_title.str.startswith(selected_name)))
    ]
    st.session_state.plaque_log = pd.concat([
        st.session_state.plaque_log,
        pd.DataFrame(new_rows)
    ], ignore_index=True)

    # CSV export
    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
