import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import trackpy as tp

st.set_page_config(layout="wide")
st.title("Interactive Plaque Counter (Canvas-Aligned)")

# === Image Upload ===
uploaded_files = st.file_uploader("Upload plaque images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
invert = st.checkbox("Invert image", value=True)
contrast = st.checkbox("Apply contrast stretch", value=True)

# Detection parameters
diameter = st.slider("Feature Diameter", 5, 50, 15, 1)
minmass = st.slider("Minimum Mass", 1, 100, 10, 1)
separation = st.slider("Minimum Separation", 1, 30, 5, 1)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1)

# Global log
def reset_log():
    return pd.DataFrame(columns=["image_title", "num_plaques"])

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

# === Main App ===
if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_name = st.selectbox("Select image", file_names)
    selected_file = next(file for file in uploaded_files if file.name == selected_name)

    # Load + preprocess
    file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert, contrast)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw
    draw_mode = st.selectbox("Drawing mode", ["transform", "circle"])
    st.subheader(selected_name)

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="green",
        background_image=Image.fromarray(image_rgb),
        update_streamlit=True,
        height=image_rgb.shape[0],
        width=image_rgb.shape[1],
        drawing_mode=draw_mode,
        key=f"canvas_{selected_name}"
    )

    # --- Transform circle coords to image space ---
    roi_img = proc
    region_defined = False
    crop_bounds = None
    display_overlay = image_rgb.copy()

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        circle = canvas_result.json_data["objects"][0]
        if circle["type"] == "circle":
            left = circle["left"]
            top = circle["top"]
            r = circle["radius"]

            cx = int(round(left + r))
            cy = int(round(top + r))
            r = int(round(r))

            y1, y2 = max(0, cy - r), min(proc.shape[0], cy + r)
            x1, x2 = max(0, cx - r), min(proc.shape[1], cx + r)
            roi_img = proc[y1:y2, x1:x2].copy()
            crop_bounds = (x1, y1)
            region_defined = True

            # Draw the ROI circle on the image for display
            cv2.circle(display_overlay, (cx, cy), r, (0, 255, 0), 2)
        else:
            st.warning("Please draw a circle to define ROI.")
    else:
        st.warning("Draw a circle to define the region of interest.")

    # Detect
    features = detect_features(roi_img, diameter, minmass, separation, confidence)

    if features is None or features.empty:
        st.info("No features found.")
        features = pd.DataFrame(columns=["x", "y"])

    # Draw detection results
    det_img = cv2.cvtColor(roi_img.copy(), cv2.COLOR_GRAY2RGB)
    for _, row in features.iterrows():
        x, y = int(round(row["x"])), int(round(row["y"]))
        cv2.circle(det_img, (x, y), diameter // 2, (0, 255, 0), 1)
        cv2.circle(det_img, (x, y), 2, (255, 0, 0), -1)

    def resize_for_display(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

    # Resize before display
    display_overlay_resized = resize_for_display(display_overlay)
    det_img_resized = resize_for_display(det_img)

    st.image(display_overlay_resized, caption="Original + ROI Overlay")
    st.image(det_img_resized, caption=f"Detected plaques: {len(features)}")

    # Log + Export
    st.session_state.plaque_log = st.session_state.plaque_log[st.session_state.plaque_log.image_title != selected_name]
    st.session_state.plaque_log.loc[len(st.session_state.plaque_log)] = [selected_name, len(features)]

    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")
