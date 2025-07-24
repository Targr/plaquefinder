import streamlit as st
import numpy as np
import pandas as pd
import cv2
import trackpy as tp
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Trackpy-Based Plaque Counter")

# === Sidebar parameters ===
st.sidebar.header("Detection Parameters")
num_plates = st.sidebar.number_input("Number of Plates", 1, 10, value=5)
diameter = st.sidebar.slider("Blob Diameter (px)", 3, 15, value=7)
minmass = st.sidebar.slider("Minimum Mass", 10, 200, value=30)
invert = st.sidebar.checkbox("Invert Contrast (for light plaques)", value=False)

# === Upload image ===
uploaded_file = st.file_uploader("Upload image of plates", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to blur for circle detection
    img_blur = cv2.medianBlur(img_array, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150,
                               param1=50, param2=30, minRadius=100, maxRadius=300)

    if circles is None or len(circles[0]) < num_plates:
        st.warning("Detected fewer plates than expected.")
    else:
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda c: (c[1], c[0]))[:num_plates]

        output_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        results = []

        for i, (x, y, r) in enumerate(circles):
            # Define safe ROI bounds
            x1, x2 = max(0, x - r), min(img_array.shape[1], x + r)
            y1, y2 = max(0, y - r), min(img_array.shape[0], y + r)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_crop = img_array[y1:y2, x1:x2]
            if roi_crop.size == 0 or np.max(roi_crop) == np.min(roi_crop):
                continue

            # Mask outside of circle in cropped ROI
            mask = np.zeros_like(roi_crop)
            cv2.circle(mask, (r, r), r, 255, -1)
            roi_crop = cv2.bitwise_and(roi_crop, roi_crop, mask=mask)

            # Normalize
            norm = (roi_crop - roi_crop.min()) / (roi_crop.max() - roi_crop.min() + 1e-8)
            norm = (norm * 255).astype(np.uint8)

            if invert:
                norm = 255 - norm

            # Use Trackpy to detect plaques
            try:
                features = tp.locate(norm, diameter=diameter, minmass=minmass, invert=False)
            except Exception:
                continue

            if features.empty:
                count = 0
            else:
                # Shift local coords to global image
                features['x'] += x1
                features['y'] += y1

                for _, row in features.iterrows():
                    cx, cy = int(row['x']), int(row['y'])
                    cv2.circle(output_img, (cx, cy), 4, (0, 255, 0), 1)

                count = len(features)

            # Label plate
            cv2.putText(output_img, f"P{i+1}: {count}", (x - 50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            results.append({"Plate": f"Plate {i+1}", "Count": count})

        st.image(output_img, caption="Detected Plaques per Plate", use_column_width=True)

        df = pd.DataFrame(results)
        st.subheader("Plaque Counts")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", csv, "plaque_counts.csv", "text/csv")
