import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="wide")
st.title("Tiny Plaque Detector with Per-Dish Counts")

uploaded_file = st.file_uploader("Upload Petri Dish Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    original_copy = img_array.copy()
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect dishes using Hough Circles
    blurred_gray = cv2.medianBlur(gray, 9)
    circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=100, param2=30, minRadius=150, maxRadius=300)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        dish_counts = []

        for i, (x, y, r) in enumerate(circles[0, :], start=1):
            # Extract ROI
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            roi_color = cv2.bitwise_and(original_copy, original_copy, mask=mask)

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(roi)

            # Invert and blur
            inv = 255 - enhanced
            blur = cv2.GaussianBlur(inv, (7, 7), 0)

            # Threshold
            _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)

            # Blob detection
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

            # Draw detected plaques as green circles
            for kp in keypoints:
                cx, cy = int(kp.pt[0]), int(kp.pt[1])
                if (cx - x)**2 + (cy - y)**2 <= r**2:  # Stay within the dish
                    cv2.circle(original_copy, (cx, cy), int(kp.size/2), (0, 255, 0), 1)

            # Label dish with plaque count
            cv2.putText(original_copy, f"D{i}: {len(keypoints)}", (x - 50, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            dish_counts.append((f"Dish {i}", len(keypoints)))

        st.image(original_copy, caption="Detected Plaques per Dish", use_column_width=True)

        st.subheader("Plaque Counts Per Dish")
        for label, count in dish_counts:
            st.write(f"{label}: {count} plaques")
    else:
        st.error("No dishes detected. Try adjusting the image or lighting.")
