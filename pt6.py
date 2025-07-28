import streamlit as st
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import io

st.set_page_config(layout="wide")
st.title("Colony Detection App")

def preprocess_image(image, blur_kernel):
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return gray, blurred

def detect_dish_circle(gray_img):
    circles = cv2.HoughCircles(
        gray_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray_img.shape[0]//8,
        param1=50,
        param2=30,
        minRadius=gray_img.shape[0]//4,
        maxRadius=gray_img.shape[0]//2
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[0]  # Use the first detected circle
    return None

def mask_outside_circle(image, circle):
    mask = np.zeros_like(image[:, :, 0])
    cv2.circle(mask, (circle[0], circle[1]), circle[2], (255), thickness=-1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

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

def draw_colonies(image, colonies, dish_circle=None):
    output = image.copy()
    for _, row in colonies.iterrows():
        cv2.circle(output, (int(row['x']), int(row['y'])), 10, (0, 255, 0), 2)
        cv2.circle(output, (int(row['x']), int(row['y'])), 3, (0, 0, 255), -1)
    if dish_circle is not None:
        cv2.circle(output, (dish_circle[0], dish_circle[1]), dish_circle[2], (0, 0, 255), 4)
    return output

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.sidebar.header("Parameters")
    colony_blur = st.sidebar.slider("Blur Kernel Size", 1, 25, 7, 2)
    colony_thresh_block = st.sidebar.slider("Adaptive Threshold Block Size", 3, 51, 31, 2)
    colony_thresh_C = st.sidebar.slider("Adaptive Threshold C", -10, 10, 5)
    colony_min_area = st.sidebar.slider("Min Area (px) for Colonies", 1, 100, 5, 1)
    colony_max_area = st.sidebar.slider("Max Area (px) for Colonies", 10, 1000, 500, 10)
    micro_boost = st.sidebar.checkbox("Boost detection of tiny faint colonies", value=True)

    gray, _ = preprocess_image(image, colony_blur)
    circle = detect_dish_circle(gray)

    if circle is not None:
        masked = mask_outside_circle(image, circle)
        dish_gray, _ = preprocess_image(masked, colony_blur)
        dish_features = detect_colonies(
            dish_gray,
            colony_blur,
            colony_thresh_block,
            colony_thresh_C,
            min_area=colony_min_area,
            max_area=colony_max_area,
            enable_micro_colony_boost=micro_boost
        )
        vis_img = draw_colonies(image, dish_features, dish_circle=circle)
        st.image(vis_img, caption=f"Detected Colonies: {len(dish_features)}", channels="BGR")
        st.dataframe(dish_features)
    else:
        st.warning("No petri dish circle detected. Showing full image results.")
        full_gray, _ = preprocess_image(image, colony_blur)
        features = detect_colonies(
            full_gray,
            colony_blur,
            colony_thresh_block,
            colony_thresh_C,
            min_area=colony_min_area,
            max_area=colony_max_area,
            enable_micro_colony_boost=micro_boost
        )
        vis_img = draw_colonies(image, features)
        st.image(vis_img, caption=f"Detected Colonies (Full Image): {len(features)}", channels="BGR")
        st.dataframe(features)
