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

st.set_page_config(layout="wide")
st.title("Multi-Plate Colony/Plaque Counter")

# File uploader
uploaded_file = st.file_uploader("Upload a petri dish photo", type=["png", "jpg", "jpeg", "tiff", "tif"])
advanced = st.checkbox("Advanced Settings")

mode = st.radio("Feature Type", options=["Colony", "Plaque"], index=1, horizontal=True)
invert = mode == "Colony"
slider_kwargs = dict(label_visibility="visible" if advanced else "visible")

diameter = st.slider("Feature Diameter (px)", 5, 51, 15, 2, **slider_kwargs)
minmass = st.slider("Minimum Mass (signal:noise)", 1, 100, 10, 1, **slider_kwargs)
confidence = st.slider("Percentile Confidence to Keep", 0, 100, 90, 1, **slider_kwargs)
separation = st.slider("Minimum Separation (px)", 1, 30, 5, 1, **slider_kwargs) if advanced else None

# Multi-plate toggle and slider
multi_plate_mode = st.checkbox("Enable Multi-Plate Detection")
max_plates = 1
if multi_plate_mode:
    max_plates = st.slider("Number of Plates to Detect", 1, 20, 2)

# --- Utilities ---
def preprocess_image(img, invert=False):
    if invert:
        img = cv2.bitwise_not(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def subtract_background(img):
    blurred = cv2.GaussianBlur(img, (31, 31), 0)
    return cv2.subtract(img, blurred)

def detect_dishes(gray_img, max_count=1):
    gray_blur = cv2.medianBlur(gray_img, 5)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray_img.shape[0] // 6,
        param1=100,
        param2=30,
        minRadius=gray_img.shape[0] // 7,
        maxRadius=gray_img.shape[0] // 2
    )
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        return circles[:max_count]
    return []

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

def analyze_colors(image, features):
    color_labels = []
    for _, row in features.iterrows():
        x, y = int(row['x']), int(row['y'])
        color = image[y, x]
        color_labels.append(tuple(color))
    return color_labels

def cluster_colors(color_list):
    from sklearn.cluster import KMeans
    X = np.array(color_list)
    k = min(len(color_list), 5)
    if k < 1:
        return {}
    kmeans = KMeans(n_clusters=k, n_init=10).fit(X)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    result = {}
    for i, count in zip(labels, counts):
        rgb = tuple(map(int, kmeans.cluster_centers_[i]))
        result[str(rgb)] = count
    return result

def draw_detected_plates(image, circles):
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (255, 0, 0), 2)

def draw_features_on_image(image, features):
    draw = ImageDraw.Draw(image)
    for _, row in features.iterrows():
        x, y = row["x"], row["y"]
        r = 5
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="lime", width=2)
    return image

def report_color_counts(image, features):
    colors = analyze_colors(image, features)
    clustered = cluster_colors(colors)
    return clustered

# --- Main Processing ---
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    while img.nbytes > 1_000_000:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = subtract_background(preprocess_image(gray, invert))
    h, w = gray.shape

    circles = detect_dishes(gray, max_count=max_plates)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    all_results = []

    if circles is None or len(circles) == 0:
        st.error("No plates detected.")
    else:
        for idx, (x, y, r) in enumerate(circles):
            yy, xx = np.ogrid[:h, :w]
            circle_mask = (xx - x)**2 + (yy - y)**2 <= r**2

            roi_proc = proc.copy()
roi_proc[~circle_mask] = 128
roi_proc = cv2.convertScaleAbs(roi_proc, alpha=3.0, beta=-128)
features = detect_features(roi_proc, diameter, minmass, separation, confidence)
if features is None or features.empty:
    features = pd.DataFrame(columns=["x", "y"])

            fx = features["x"].astype(int)
            fy = features["y"].astype(int)
            inside = circle_mask[fy.clip(0, h-1), fx.clip(0, w-1)]
            inside_features = features[inside]

            draw_features_on_image(pil_img, inside_features)
            draw = ImageDraw.Draw(pil_img)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="red", width=3)

            color_counts = report_color_counts(img, inside_features)
            color_summary = ", ".join([f"{count} of color {color}" for color, count in color_counts.items()])

            st.subheader(f"Plate {idx + 1}")
            st.success(f"{len(inside_features)} features detected inside Plate {idx + 1}")
            if color_counts:
                st.info(f"Breakdown: {color_summary}")

            all_results.append((f"Plate {idx + 1}", len(inside_features), color_counts))

        st.image(pil_img, caption="Detected Colonies/Plaques with Annotations", use_column_width=True)

        # CSV Export
        df = pd.DataFrame([{
            "plate": name,
            "count": count,
            **{f"color_{i}": f"{v}" for i, (v, _) in enumerate(colors.items())}
        } for name, count, colors in all_results])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv, "plate_colony_results.csv", "text/csv")



# Multi-Plate Colony/Plaque Counter (continued full integration)

# --- Parameter Saving & Loading ---
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
                "max_plates": max_plates,
                "multi": multi_plate_mode
            }
            st.success(f"Saved as '{param_name}'")
        else:
            st.warning("Please enter a name to save the parameters.")

    if st.session_state.saved_param_sets:
        selected = st.selectbox("Load saved parameters", options=["Select..."] + list(st.session_state.saved_param_sets.keys()))
        if selected != "Select...":
            params = st.session_state.saved_param_sets[selected]
            invert = params.get("invert", False)
            diameter = params.get("diameter", 15)
            minmass = params.get("minmass", 10)
            confidence = params.get("confidence", 90)
            separation = params.get("separation", 5)
            max_plates = params.get("max_plates", 1)
            multi_plate_mode = params.get("multi", False)
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

# --- Optional Batch Upload ---
st.markdown("---")
st.header("Batch Process a Folder of Images")

batch_files = st.file_uploader("Upload multiple dish images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
add_thumbnails = st.checkbox("Add image thumbnails to CSV")
add_zip_download = st.checkbox("Include ZIP download of overlaid images")

if batch_files and st.button("Process Folder and Export CSV"):
    results = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file in batch_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            while img.nbytes > 1_000_000:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            proc = subtract_background(preprocess_image(gray, invert))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            detected = detect_dishes(gray, max_count=max_plates)
            for idx, (x, y, r) in enumerate(detected):
                h, w = gray.shape
                yy, xx = np.ogrid[:h, :w]
                circle_mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2

                features = detect_features(proc, diameter, minmass, separation, confidence)
                if features.empty:
                    continue

                fx = features["x"].astype(int)
                fy = features["y"].astype(int)
                inside = circle_mask[fy.clip(0, h-1), fx.clip(0, w-1)]
                inside_features = features[inside]

                color_counts = report_color_counts(img, inside_features)
                color_summary = ", ".join([f"{count} of color {color}" for color, count in color_counts.items()])
                draw_features_on_image(pil_img, inside_features)
                draw = ImageDraw.Draw(pil_img)
                draw.ellipse([(x - r, y - r), (x + r, y + r)], outline="red", width=3)

                # Save overlay image
                if add_zip_download:
                    img_io = io.BytesIO()
                    pil_img.save(img_io, format="PNG")
                    zip_file.writestr(f"{file.name}_plate_{idx+1}.png", img_io.getvalue())

                row = {
                    "image_name": file.name,
                    "plate_number": idx+1,
                    "feature_count": len(inside_features),
                    **{f"color_{k}": v for k, v in enumerate(color_counts.values())}
                }
                if add_thumbnails:
                    thumb = pil_to_base64_thumbnail(pil_img)
                    row["thumbnail"] = thumb

                results.append(row)

    df = pd.DataFrame(results)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Batch CSV", csv, "batch_feature_counts.csv", "text/csv")

    if add_zip_download:
        zip_buffer.seek(0)
        st.download_button("Download ZIP of Overlays", zip_buffer, "overlays.zip", "application/zip")

