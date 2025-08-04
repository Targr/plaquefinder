import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Touch-Friendly Plaque Counter")

mobile_file = st.file_uploader("Take a photo of a dish", type=["png", "jpg", "jpeg"])
advanced = st.checkbox("Advanced Settings")

invert = st.checkbox("Invert image", value=True) if advanced else False
diameter = st.slider("Feature Diameter", 5, 51, 15, 2)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1) if advanced else None

def preprocess_image(img, invert=False):
    if invert:
        img = cv2.bitwise_not(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

proc = preprocess_image(gray, invert)

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

def ellipse_mask_filter(features, cx, cy, rx, ry):
    dx = features['x'] - cx
    dy = features['y'] - cy
    inside = ((dx / rx) ** 2 + (dy / ry) ** 2) <= 1
    return features[inside]

if "plaque_log" not in st.session_state:
    st.session_state.plaque_log = pd.DataFrame(columns=["image_title", "ellipse_id", "num_plaques"])

if mobile_file:
    file_bytes = np.asarray(bytearray(mobile_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    canvas_w, canvas_h = pil_image.size

# === Detect features
features = detect_features(proc, diameter, minmass, separation, confidence)
if features is None or features.empty:
    features = pd.DataFrame(columns=["x", "y"])

# === Initialize ellipse in session state if missing
if "ellipse_state" not in st.session_state:
    st.session_state.ellipse_state = {
        "type": "ellipse",
        "left": canvas_w // 4,
        "top": canvas_h // 4,
        "rx": canvas_w // 6,
        "ry": canvas_h // 6,
        "fill": "rgba(255,255,255,0)",
        "stroke": "#FF0000",
        "strokeWidth": 3,
        "angle": 0
    }

# === Pull user-modified ellipse if it exists in canvas_result
if st.session_state.get("ellipse_canvas") and st.session_state.ellipse_canvas.get("json_data"):
    json_data = st.session_state.ellipse_canvas["json_data"]
    for obj in json_data.get("objects", []):
        if obj["type"] == "ellipse":
            # Save updated shape to session state
            st.session_state.ellipse_state = obj
            break

ellipse = st.session_state.ellipse_state

# === Compute ellipse center and size
left = ellipse.get("left", canvas_w // 4)
top = ellipse.get("top", canvas_h // 4)
rx = ellipse.get("rx", canvas_w // 6)
ry = ellipse.get("ry", canvas_h // 6)
width = ellipse.get("width", 2 * rx)
height = ellipse.get("height", 2 * ry)
cx = left + width / 2
cy = top + height / 2

# === Filter features inside ellipse
mask_feats = ellipse_mask_filter(features, cx, cy, rx, ry)

# === Green overlay markers
canvas_objects = [ellipse]
for _, row in mask_feats.iterrows():
    canvas_objects.append({
        "type": "circle",
        "left": float(row["x"]) - 3,
        "top": float(row["y"]) - 3,
        "radius": 3,
        "fill": "rgba(0,255,0,0.8)",
        "stroke": "#000000",
        "strokeWidth": 1,
        "selectable": False
    })

initial_shape = {
    "version": "4.4.0",
    "objects": canvas_objects
}

# === Render canvas
st.markdown("Move and resize the red ellipse to select a dish region.")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=pil_image,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode="transform",
    initial_drawing=initial_shape,
    key="ellipse_canvas"
)


    # === Live Count Display
st.markdown("### Live Plaque Count")
st.write(f"Ellipse 1: **{len(mask_feats)} plaques detected**")

    # === Update Session Log and Offer CSV
st.session_state.plaque_log = st.session_state.plaque_log[
    st.session_state.plaque_log.image_title != mobile_file.name
    ]
st.session_state.plaque_log = pd.concat([
    st.session_state.plaque_log,
    pd.DataFrame([{
        "image_title": mobile_file.name,
        "ellipse_id": "Ellipse 1",
        "num_plaques": len(mask_feats)
        }])
    ], ignore_index=True)

csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
