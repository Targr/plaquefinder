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
