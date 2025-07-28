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

    # Auto-tune detection parameters based on number of dishes
    scale_factor = 1.0 / num_dishes
    auto_diameter = max(5, int(diameter * scale_factor))
    auto_separation = max(1, int(separation * scale_factor))
    auto_minmass = max(1, int(minmass * scale_factor))

    canvas_bg_resized, canvas_scale = resize_with_scale(image_rgb)

    st.subheader(selected_name)

    display_overlay = image_rgb.copy()
    features = detect_features(proc, auto_diameter, auto_minmass, auto_separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    dish_circles = detect_multiple_dishes(gray, num_dishes)
    new_rows = []
    for i, (cx, cy, cr) in enumerate(dish_circles):
        rx, ry = int(cr * 0.95), int(cr * 0.95)
        cv2.ellipse(display_overlay, (cx, cy), (rx, ry), 0, 0, 360, (255, 0, 0), 2)
        dish_feats = ellipse_mask_filter(features, cx, cy, rx, ry, angle_deg=0)

        for j, (_, row) in enumerate(dish_feats.iterrows()):
            x, y = int(round(row["x"])), int(round(row["y"]))
            cv2.circle(display_overlay, (x, y), auto_diameter // 2, (0, 255, 0), 1)
            cv2.circle(display_overlay, (x, y), 2, (255, 0, 0), -1)
    
    plaque_id = f"D{i+1}_P{j+1}"
    cv2.putText(display_overlay, plaque_id, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(display_overlay, plaque_id, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


        plaque_count = len(dish_feats)
        new_rows.append({"image_title": selected_name, "dish_id": f"Dish {i+1}", "num_plaques": plaque_count})

        # === AUTOMATED REFINEMENT FOR >2 DISHES ===
        if num_dishes > 2:
            pad = int(cr * 1.1)
            x1, x2 = max(cx - pad, 0), min(cx + pad, gray.shape[1])
            y1, y2 = max(cy - pad, 0), min(cy + pad, gray.shape[0])
            dish_crop = proc[y1:y2, x1:x2]

            mask = np.zeros_like(dish_crop)
            ellipse_center = (dish_crop.shape[1] // 2, dish_crop.shape[0] // 2)
            ellipse_axes = (int(rx * 0.9), int(ry * 0.9))
            cv2.ellipse(mask, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)
            dish_crop_masked = cv2.bitwise_and(dish_crop, dish_crop, mask=mask)

            refined_feats = detect_features(
                dish_crop_masked,
                diameter=diameter,
                minmass=minmass,
                separation=separation,
                confidence=confidence
            )
            refined_count = len(refined_feats) if refined_feats is not None else 0
            new_rows[-1]["num_plaques"] = refined_count  # overwrite with better estimate

        cv2.putText(display_overlay, f"Dish {i+1}: {new_rows[-1]['num_plaques']}", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    display_overlay_resized, _ = resize_with_scale(display_overlay)
    st.image(display_overlay_resized, caption="Detected plaques per dish")

    st.session_state.plaque_log = st.session_state.plaque_log[
        ~((st.session_state.plaque_log.image_title == selected_name) &
          (st.session_state.plaque_log.dish_id.str.startswith("Dish")))
    ]
    st.session_state.plaque_log = pd.concat([
        st.session_state.plaque_log,
        pd.DataFrame(new_rows)
    ], ignore_index=True)

    csv = st.session_state.plaque_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="plaque_counts.csv", mime="text/csv")

# === COMPLETELY INDEPENDENT COLOR MATCH TOOL ===
st.markdown("---")
st.subheader("Color Finder (Independent)")

enable_color_tool = st.checkbox("Enable standalone color picker", value=False)
color_tolerance_slider = st.slider("Color distance tolerance", 0, 255, 40, 1)

if enable_color_tool:
    if uploaded_files:
        # Resize original image for canvas
        rgb_for_canvas, _ = resize_with_scale(image_rgb.copy())
        h, w = rgb_for_canvas.shape[:2]

        # Draw canvas for user to click a point
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",
            stroke_width=5,
            background_image=Image.fromarray(rgb_for_canvas),
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="point",
            key="color_picker_canvas"
        )

        # Process the click
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            point = canvas_result.json_data["objects"][-1]
            x_click = int(point["left"])
            y_click = int(point["top"])

            # Convert canvas coordinates to full-res image coordinates
            x_orig = int(x_click * image_rgb.shape[1] / w)
            y_orig = int(y_click * image_rgb.shape[0] / h)

            if 0 <= x_orig < image_rgb.shape[1] and 0 <= y_orig < image_rgb.shape[0]:
                selected_color = image_rgb[y_orig, x_orig].astype(np.int16)

                # Show selected color
                st.markdown(f"**Selected color:** RGB({selected_color[0]}, {selected_color[1]}, {selected_color[2]})")
                st.color_picker("Preview", value=f"#{selected_color[0]:02x}{selected_color[1]:02x}{selected_color[2]:02x}".upper(), label_visibility="collapsed", disabled=True)

                # Find all matching pixels
                diff_img = np.linalg.norm(image_rgb.astype(np.int16) - selected_color, axis=2)
                match_mask = diff_img <= color_tolerance_slider

                # Overlay yellow on matched pixels
                highlight_img = image_rgb.copy()
                highlight_img[match_mask] = [255, 255, 0]  # Yellow highlight

                # Show result
                highlight_resized, _ = resize_with_scale(highlight_img)
                st.image(highlight_resized, caption="Pixels matching selected color")

    else:
        st.info("Upload an image first to use the color picker.")
