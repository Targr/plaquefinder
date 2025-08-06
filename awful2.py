import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import trackpy as tp
import json
import io
import base64
import zipfile

# ========================== FUNCTIONS ==========================

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

def detect_dish(gray_img):
    gray_blur = cv2.medianBlur(gray_img, 5)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray_img.shape[0] // 4,
        param1=100,
        param2=30,
        minRadius=gray_img.shape[0] // 5,
        maxRadius=gray_img.shape[0] // 2
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]
    return None

def extract_circle_geometry(obj):
    raw_r = obj.get("radius", 50)
    left = obj.get("left", 0)
    top = obj.get("top", 0)
    scale_x = obj.get("scaleX", 1.0)
    scale_y = obj.get("scaleY", 1.0)
    scale = (scale_x + scale_y) / 2
    r = raw_r * scale
    x = left + raw_r * scale_x
    y = top + raw_r * scale_y
    return x, y, r

def draw_features_on_image(image, features):
    draw = ImageDraw.Draw(image)
    for _, row in features.iterrows():
        x, y = row["x"], row["y"]
        r = 5
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="lime", width=2)
    return image

def pil_to_base64_thumbnail(pil_img, size=(100, 100)):
    img_copy = pil_img.copy()
    img_copy.thumbnail(size)
    buffered = io.BytesIO()
    img_copy.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# ========================== STREAMLIT UI ==========================

st.set_page_config(layout="wide")
st.title("Plaque/Colony Counter")

uploaded_file = st.file_uploader("Upload a petri dish photo", type=["png", "jpg", "jpeg", "tiff", "tif"])
advanced = st.checkbox("Advanced Settings")

mode = st.radio("Feature Type", options=["Colony", "Plaque"], index=1, horizontal=True)
invert = mode == "Colony"

# Parameter sliders
slider_kwargs = dict(label_visibility="visible" if advanced else "visible")
diameter = st.slider("Feature Diameter (px)", 5, 51, 15, 2, **slider_kwargs)
minmass = st.slider("Minimum Mass (signal:noise)", 1, 100, 10, 1, **slider_kwargs)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1, **slider_kwargs)
separation = st.slider("Minimum Separation (px)", 1, 30, 5, 1, **slider_kwargs) if advanced else None

# Parameter sets
if "saved_param_sets" not in st.session_state:
    st.session_state.saved_param_sets = {}

st.markdown("#### Save or Load Parameter Sets")
with st.expander("Manage Parameter Sets", expanded=False):
    param_name = st.text_input("Save current parameters as:")
    if st.button("Save Parameters"):
        if param_name:
            st.session_state.saved_param_sets[param_name] = {
                "invert": invert,
                "diameter": diameter,
                "minmass": minmass,
                "confidence": confidence,
                "separation": separation,
            }
            st.success(f"Saved as '{param_name}'")
        else:
            st.warning("Please enter a name to save the parameters.")

    if st.session_state.saved_param_sets:
        selected = st.selectbox("Load saved parameters", options=["Select..."] + list(st.session_state.saved_param_sets.keys()))
        if selected != "Select...":
            params = st.session_state.saved_param_sets[selected]
            invert = params["invert"]
            diameter = params["diameter"]
            minmass = params["minmass"]
            confidence = params["confidence"]
            separation = params["separation"]
            mode = "Colony" if invert else "Plaque"
            st.success(f"Loaded parameters: '{selected}'")

        param_json = json.dumps(st.session_state.saved_param_sets, indent=2)
        st.download_button("Download All Saved Parameters (.txt)", data=param_json, file_name="saved_parameters.txt", mime="text/plain")

    uploaded_param_file = st.file_uploader("Load parameters from .txt", type=["txt"])
    if uploaded_param_file:
        try:
            loaded_params = json.load(uploaded_param_file)
            if isinstance(loaded_params, dict):
                st.session_state.saved_param_sets.update(loaded_params)
                st.success("Parameters successfully loaded and added to memory.")
            else:
                st.error("Invalid parameter file format.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Multi-plate ROI handling
num_plates = st.number_input("How many plates are in the image?", min_value=1, max_value=10, value=1, step=1)

if "locked_circle_objs" not in st.session_state:
    st.session_state.locked_circle_objs = [None] * num_plates
if "edit_mode_multi" not in st.session_state:
    st.session_state.edit_mode_multi = True

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = preprocess_image(gray, invert)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    h, w = gray.shape

    features = detect_features(proc, diameter, minmass, separation, confidence)
    if features is None or features.empty:
        features = pd.DataFrame(columns=["x", "y"])

    # ROI UI
    overlay_objects = []
    canvas_key = f"canvas_{num_plates}_{st.session_state.edit_mode_multi}"

    if st.session_state.edit_mode_multi or any(obj is None for obj in st.session_state.locked_circle_objs[:num_plates]):
        default_circles = []
        for i in range(num_plates):
            if st.session_state.locked_circle_objs[i] is None:
                detected = detect_dish(gray)
                if detected is None:
                    st.error(f"Could not auto-detect dish #{i+1}.")
                    continue
                x0, y0, r = detected
                circle_obj = {
                    "type": "circle",
                    "left": float(x0 - r),
                    "top": float(y0 - r),
                    "radius": float(r),
                    "fill": "rgba(255,255,255,0)",
                    "stroke": "#FF0000",
                    "strokeWidth": 3,
                    "selectable": True,
                    "name": f"circle_{i}"
                }
            else:
                circle_obj = st.session_state.locked_circle_objs[i].copy()
                circle_obj["selectable"] = True
                circle_obj["name"] = f"circle_{i}"
            default_circles.append(circle_obj)
        overlay_objects = default_circles
    else:
        for i in range(num_plates):
            circle_obj = st.session_state.locked_circle_objs[i].copy()
            circle_obj["selectable"] = False
            overlay_objects.append(circle_obj)

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#00FF00",
        background_image=pil_img,
        update_streamlit=True,
        height=h,
        width=w,
        initial_drawing={"version": "4.4.0", "objects": overlay_objects},
        drawing_mode="transform",
        key=canvas_key,
    )

    if st.session_state.edit_mode_multi or any(obj is None for obj in st.session_state.locked_circle_objs[:num_plates]):
        if st.button("Done (Lock All Circles)"):
            locked = [None] * num_plates
            found_count = 0
            for obj in canvas_result.json_data.get("objects", []):
                if obj["type"] == "circle" and obj.get("selectable", True):
                    name = obj.get("name", "")
                    if name.startswith("circle_"):
                        idx = int(name.split("_")[1])
                        if idx < num_plates:
                            locked[idx] = obj
                            found_count += 1
            if found_count == num_plates:
                st.session_state.locked_circle_objs = locked
                st.session_state.edit_mode_multi = False
                st.rerun()
            else:
                st.error("Please draw/select all circles.")
    else:
        if st.button("Reset All ROIs"):
            st.session_state.locked_circle_objs = [None] * num_plates
            st.session_state.edit_mode_multi = True
            st.rerun()

    # Count features inside each ROI
    st.markdown("### Plaque/Colony Count Inside Each ROI")
    total_count = 0
    for idx, circle_obj in enumerate(st.session_state.locked_circle_objs[:num_plates]):
        x0, y0, r = extract_circle_geometry(circle_obj)
        yy, xx = np.ogrid[:h, :w]
        circle_mask = (xx - x0)**2 + (yy - y0)**2 <= r**2

        fx = features["x"].astype(int)
        fy = features["y"].astype(int)
        inside = circle_mask[fy.clip(0, h-1), fx.clip(0, w-1)]
        inside_features = features[inside]

        count = len(inside_features)
        total_count += count

        st.markdown(f"**Plate #{idx + 1}**: {count} features")
    st.success(f"Total across all ROIs: {total_count}")
