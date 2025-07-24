import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Tiny Plaque Detector")

# Upload image
uploaded_file = st.file_uploader("Upload Petri Dish Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # Invert and blur
    blurred = cv2.GaussianBlur(255 - gray_enhanced, (7, 7), 0)

    # Threshold to segment plaques
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

    # Blob detection for small plaques
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 150
    params.filterByCircularity = True
    params.minCircularity = 0.2
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)

    # Draw keypoints
    output = cv2.drawKeypoints(img_array, keypoints, np.array([]), (0, 255, 0),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    st.image(output, caption=f"Detected {len(keypoints)} Tiny Plaques", use_column_width=True)
