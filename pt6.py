import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import os

st.set_page_config(layout="wide")
st.title("Auto Plate Analysis: Plaques vs. Colonies")

def load_image(image_file):
    img = Image.open(image_file)
    if image_file.size > 2 * 1024 * 1024:  # If file > 2MB, downscale
        img.thumbnail((img.width // 2, img.height // 2))
    return np.array(img)

def detect_plate_type(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio < 0.5:
        return "plaque"
    else:
        return "colony"

def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=100, maxRadius=0)
    return circles

def crop_plate(image, circle):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    x, y, r = circle
    return masked[y - r:y + r, x - r:x + r]

def detect_colonies(gray_img, blur_kernel=7, block_size=31, C=5, min_area=5, max_area=500, enable_micro_colony_boost=True):
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    block_size = block_size if block_size % 2 == 1 else block_size + 1

    blurred = cv2.GaussianBlur(gray_img, (blur_kernel, blur_kernel), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                features.append((cx, cy))

    if enable_micro_colony_boost:
        micro_thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block_size // 2 | 1,
            C + 5
        )
        micro_contours, _ = cv2.findContours(micro_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in micro_contours:
            area = cv2.contourArea(cnt)
            if 2 <= area < min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    features.append((cx, cy))

    df = pd.DataFrame(features, columns=["x", "y"])
    if not df.empty:
        df = df.drop_duplicates(subset=["x", "y"])
        df = df.reset_index(drop=True)

    return df

def plot_points(image, points):
    for (x, y) in points:
        cv2.circle(image, (x, y), 5, (0, 255, 0), 1)
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    return image

uploaded_files = st.file_uploader("Upload Plate Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    for image_file in uploaded_files:
        image = load_image(image_file)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        plate_type = detect_plate_type(image)
        circles = detect_circles(image)

        st.subheader(f"File: {image_file.name} | Detected as: {plate_type.title()} Plate")

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, c in enumerate(circles[0, :]):
                cropped = crop_plate(image, c)
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                features = detect_colonies(gray)

                result = plot_points(cropped.copy(), features.to_numpy())

                col1, col2 = st.columns(2)
                with col1:
                    st.image(cropped, caption=f"Cropped Plate #{i+1}", channels="BGR")
                with col2:
                    st.image(result, caption=f"Detected ({len(features)} {plate_type}s)", channels="BGR")

        else:
            st.warning("No circular plates detected in image.")
