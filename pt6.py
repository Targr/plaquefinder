import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp
import io

st.set_page_config(layout="wide")
st.title("Interactive Plaque Counter (Dish-Cropped with Ellipse Masking)")

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

    if len(dish_circles) > 0:
        st.subheader(f"Detected {len(dish_circles)} dish(es)")

        for i, (cx, cy, cr) in enumerate(dish_circles):
            # Crop a square region around the dish
            margin = int(cr * 1.1)
            x1 = max(0, cx - margin)
            x2 = min(img.shape[1], cx + margin)
            y1 = max(0, cy - margin)
            y2 = min(img.shape[0], cy + margin)
            dish_crop = img[y1:y2, x1:x2]
            dish_gray = cv2.cvtColor(dish_crop, cv2.COLOR_BGR2GRAY)
            dish_proc = preprocess_image(dish_gray, invert, contrast)
            dish_overlay = cv2.cvtColor(dish_crop, cv2.COLOR_BGR2RGB)

            # Adjust ellipse center for the cropped image
            cx_adj = cx - x1
            cy_adj = cy - y1
            rx = ry = int(cr * 0.95)
            angle_deg = 0

            # Detect all features in crop
            dish_features = detect_features(dish_proc, diameter, minmass, separation, confidence)
            if dish_features is None or dish_features.empty:
                dish_features = pd.DataFrame(columns=["x", "y"])

            # Ellipse mask
            masked_feats = ellipse_mask_filter(dish_features, cx_adj, cy_adj, rx, ry, angle_deg)

            # Draw ellipse and features
            cv2.ellipse(dish_overlay, (cx_adj, cy_adj), (rx, ry), angle_deg, 0, 360, (255, 0, 0), 2)
            for _, row in masked_feats.iterrows():
                x, y = int(round(row["x"])), int(round(row["y"]))
                cv2.circle(dish_overlay, (x, y), diameter // 2, (0, 255, 0), 1)
                cv2.circle(dish_overlay, (x, y), 2, (255, 0, 0), -1)

            count = len(masked_feats)
            cv2.putText(dish_overlay, f"Dish {i+1}: {count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display
            st.image(dish_overlay, caption=f"Dish {i+1}: {count} plaques", use_column_width=True)

            # Log
            new_rows.append({
                "image_title": f"{selected_name} (Dish {i+1})",
                "dish_id": f"Dish {i+1}",
                "num_plaques": count
            })
    else:
        # No dish detected — fallback to whole image
        features = detect_features(proc, diameter, minmass, separation, confidence)
        if features is None or features.empty:
            features = pd.DataFrame(columns=["x", "y"])
        display_overlay = image_rgb.copy()
        for _, row in features.iterrows():
            x, y = int(round(row["x"])), int(round(row["y"]))
            cv2.circle(display_overlay, (x, y), diameter // 2, (0, 255, 0), 1)
            cv2.circle(display_overlay, (x, y), 2, (255, 0, 0), -1)

        count = len(features)
        cv2.putText(display_overlay, f"Plaques: {count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        st.image(display_overlay, caption=f"Detected plaques: {count}", use_column_width=True)

        new_rows.append({
            "image_title": selected_name,
            "dish_id": "Whole Image",
            "num_plaques": count
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
