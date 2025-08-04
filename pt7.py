import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Interactive Plaque Counter (Touch-Friendly)")

# === Mobile Upload ===
mobile_file = st.file_uploader("Take a photo of a dish", type=["png", "jpg", "jpeg"])

# === Advanced Settings Toggle ===
advanced = st.checkbox("Advanced Settings")

# === Detection Parameters ===
invert = st.checkbox("Invert image", value=True) if advanced else False
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)

if advanced:
    separation = st.slider("Minimum Separation", 1, 30, 5, 1)
else:
    separation = None  # auto/default

minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)

# === Utility ===
def reset_log():
    return pd.DataFrame(columns=["image_title", "ellipse_id", "num_plaques"])

if "plaque_log" not in st.session_state:
    st.session_state.plaque_log = reset_log()

def preprocess_image(img, invert=False):
    if invert:
        img = cv2.bitwise_not(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def detect_features(gray, diameter, minmass, separation, confidence):
    norm_img = (gray / 255.0).astype(np.float32)
    try:
        features = tp.locate(
            norm_img,
            diameter=diameter,
            minmass=minmass,
            separation=separation or diameter,
            percentile=confidence,
            invert=False
        )
    except Exception:
        features = pd.DataFrame()
    return features

def ellipse_mask_filter(features, x, y, rx, ry):
    dx = features['x'] - x
    dy = features['y'] - y
    inside = ((dx / rx) ** 2 + (dy / ry) ** 2) <= 1
    return features[inside]

# === Main Logic ===
if mobile_file:
    file_bytes = np.asarray(bytearray(mobile_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    canvas_img = Image.fromarray(image_rgb)
    canvas_w, canvas_h = canvas_img.size

    st.markdown("Draw or resize the ellipse around a dish region.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=canvas_img,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="transform",
        key="ellipse_canvas",
    )

    if canvas_result.json_data is not None:
        objs = canvas_result.json_data["objects"]
        if objs:
            features = detect_features(proc, diameter, minmass, separation, confidence)
            features = features if features is not None else pd.DataFrame(columns=["x", "y"])
            new_rows = []
            for i, obj in enumerate(objs):
                if obj["type"] == "ellipse":
                    left = obj["left"]
                    top = obj["top"]
                    rx = obj["rx"]
                    ry = obj["ry"]
                    cx = left + obj.get("width", 2 * rx) / 2
                    cy = top + obj.get("height", 2 * ry) / 2

                    mask_feats = ellipse_mask_filter(features, cx, cy, rx, ry)
                    count = len(mask_feats)
                    new_rows.append({
                        "image_title": mobile_file.name,
                        "ellipse_id": f"Ellipse {i + 1}",
                        "num_plaques": count
                    })

            st.session_state.plaque_log = pd.concat([
                st.session_state.plaque_log,
                pd.DataFrame(new_rows)
            ], ignore_index=True)

            st.success(f"{len(new_rows)} region(s) analyzed. See CSV below.")

    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
