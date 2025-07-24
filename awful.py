import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import trackpy as tp
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Plaque Counter Using Trackpy with Manual Plate Count")

# === Upload image ===
uploaded_file = st.file_uploader("Upload image of plates", type=["jpg", "jpeg", "png", "tif"])
num_plates = st.number_input("How many plates are in the image?", min_value=1, max_value=10, value=5)

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # === Detect circular plates using HoughCircles ===
    img_blur = cv2.medianBlur(img_array, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150,
                               param1=50, param2=30, minRadius=100, maxRadius=300)

    if circles is None or len(circles[0]) < num_plates:
        st.warning("Detected fewer plates than expected. Try adjusting image quality or count.")
    else:
        circles = np.uint16(np.around(circles[0]))
        # Sort by y (then x) to stabilize plate ordering
        circles = sorted(circles, key=lambda c: (c[1], c[0]))[:num_plates]

        results = []
        output_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        for i, (x, y, r) in enumerate(circles):
            # Crop circular ROI
            mask = np.zeros_like(img_array)
            cv2.circle(mask, (x, y), r, 255, -1)
            roi = cv2.bitwise_and(img_array, img_array, mask=mask)

            # Extract bounding box for ROI to reduce trackpy load
            x1, x2 = max(0, x - r), min(img_array.shape[1], x + r)
            y1, y2 = max(0, y - r), min(img_array.shape[0], y + r)
            roi_crop = roi[y1:y2, x1:x2]

            # Normalize ROI for trackpy (optional, enhances light plaques)
            roi_norm = (roi_crop - np.min(roi_crop)) / (np.max(roi_crop) - np.min(roi_crop) + 1e-8)
            roi_norm = (roi_norm * 255).astype(np.uint8)

            # Detect bright spots (plaques)
            f = tp.locate(roi_norm, diameter=7, minmass=30, invert=False)

            # Offset to global image coords
            f['x'] += x1
            f['y'] += y1

            # Draw plaques
            for _, row in f.iterrows():
                cx, cy = int(row['x']), int(row['y'])
                cv2.circle(output_rgb, (cx, cy), 4, (0, 255, 0), 1)

            # Label plate
            cv2.putText(output_rgb, f"P{i+1}: {len(f)}", (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            results.append((f"Plate {i+1}", len(f)))

        # === Output ===
        st.image(output_rgb, caption="Detected Plaques per Plate", use_column_width=True)

        st.subheader("Plaque Counts Per Plate")
        for plate_label, count in results:
            st.write(f"{plate_label}: {count} plaques")
