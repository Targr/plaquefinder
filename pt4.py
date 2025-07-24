import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp
import io

st.set_page_config(layout="wide")
st.title("Interactive Plaque Counter (Canvas-Aligned with Multi-Dish Support)")

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

    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert, contrast)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    canvas_bg_resized, canvas_scale = resize_with_scale(image_rgb)
    canvas_height, canvas_width = canvas_bg_resized.shape[:2]

    st.subheader("Draw ellipses for each Petri dish")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=Image.fromarray(canvas_bg_resized),
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="transform",
        key="canvas",
    )

    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    display_overlay = image_rgb.copy()
    dish_results = []

    if canvas_result.json_data is not None:
        for i, obj in enumerate(canvas_result.json_data["objects"]):
            if obj["type"] == "ellipse":
                cx = obj["left"] + obj["rx"]
                cy = obj["top"] + obj["ry"]
                rx = obj["rx"]
                ry = obj["ry"]
                cx_scaled = int(cx / canvas_scale)
                cy_scaled = int(cy / canvas_scale)
                rx_scaled = int(rx / canvas_scale)
                ry_scaled = int(ry / canvas_scale)

                mask_feats = ellipse_mask_filter(features, cx_scaled, cy_scaled, rx_scaled, ry_scaled, angle_deg=0)
                count = len(mask_feats)
                dish_results.append((f"Dish {i+1}", count))

                for _, row in mask_feats.iterrows():
                    x, y = int(round(row["x"])), int(round(row["y"]))
                    cv2.circle(display_overlay, (x, y), diameter // 2, (0, 255, 0), 1)
                    cv2.circle(display_overlay, (x, y), 2, (255, 0, 0), -1)

                cv2.ellipse(display_overlay, (cx_scaled, cy_scaled), (rx_scaled, ry_scaled), 0, 0, 360, (255, 0, 0), 2)
                cv2.putText(display_overlay, f"#{i+1}: {count}", (cx_scaled - 20, cy_scaled),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    display_overlay_resized, _ = resize_with_scale(display_overlay)
    st.image(display_overlay_resized, caption="Detected plaques per drawn region")

    st.session_state.plaque_log = st.session_state.plaque_log[st.session_state.plaque_log.image_title != selected_name]
    for dish_label, count in dish_results:
        st.session_state.plaque_log.loc[len(st.session_state.plaque_log)] = [selected_name, dish_label, count]

    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
