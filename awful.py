import streamlit as st
import numpy as np
import pandas as pd
import cv2
import trackpy as tp
from PIL import Image
import datetime

st.set_page_config(page_title="Plaque Counter Adaptive", layout="wide")
st.title("🧫 Plaque Counter — Adaptive Threshold Mode")
st.caption("Plaque detection using adaptive Gaussian thresholding + Trackpy on cropped dishes.")

# Sidebar controls
with st.sidebar:
    st.header("🔧 Image Settings")
    scale_factor = st.slider("Scale Image (0.05–1.0)", 0.05, 1.0, 0.5, 0.05)
    invert = st.checkbox("Invert Image", value=True)
    overlay_on_original = st.checkbox("Overlay on Original Image", value=True)

    st.header("🍩 Plate Detection (Hough Circles)")
    num_plates = st.number_input("Expected Plates", 1, 20, 4)
    blur_k = st.slider("Blur Kernel", 3, 15, 5, step=2)
    param1 = st.slider("Edge Sensitivity (param1)", 10, 100, 50)
    param2 = st.slider("Circle Threshold (param2)", 10, 100, 30)
    min_radius = st.slider("Min Radius", 30, 200, 50)
    max_radius = st.slider("Max Radius", 100, 400, 150)

    st.header("🧪 Detection Parameters")
    block_size = st.slider("Block Size (odd)", 3, 51, 11, step=2)
    c_value = st.slider("C (mean subtraction)", -20, 20, 2)
    diameter = st.slider("Blob Diameter (px)", 3, 15, 7)
    minmass = st.slider("Minimum Brightness (mass)", 10, 500, 40)

# Upload image
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)

    # Resize
    new_size = (int(img_array.shape[1] * scale_factor), int(img_array.shape[0] * scale_factor))
    img_resized = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)
    img_for_overlay = img_resized.copy()

    # Blur + detect plates
    img_blur = cv2.GaussianBlur(img_resized, (blur_k, blur_k), 0)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    if circles is None or len(circles[0]) < num_plates:
        st.error("⚠️ Not enough plates detected. Try adjusting parameters.")
    else:
        circles = np.uint16(np.around(circles[0]))[:num_plates]
        circles = sorted(circles, key=lambda c: (c[1], c[0]))

        annotated = cv2.cvtColor(
            255 - img_for_overlay if invert and overlay_on_original else img_for_overlay,
            cv2.COLOR_GRAY2BGR
        ) if overlay_on_original else cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        results = []

        for i, (x, y, r) in enumerate(circles):
            x1, x2 = max(0, x - r), min(img_resized.shape[1], x + r)
            y1, y2 = max(0, y - r), min(img_resized.shape[0], y + r)
            roi = img_resized[y1:y2, x1:x2]

            if roi.size == 0 or roi.max() == roi.min():
                continue

            # Mask and normalize
            mask = np.zeros_like(roi)
            cv2.circle(mask, (r, r), r, 255, -1)
            masked = cv2.bitwise_and(roi, roi, mask=mask)

            if invert:
                masked = 255 - masked

            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(masked, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, c_value)

            try:
                spots = tp.locate(thresh, diameter=diameter, minmass=minmass, invert=False)
            except Exception:
                continue

            count = len(spots)
            results.append({"Plate": f"Plate {i+1}", "Count": count})

            # Draw overlays
            cv2.circle(annotated, (x, y), r, (0, 0, 255), 1)
            cv2.putText(annotated, f"P{i+1}: {count}", (x - 40, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not spots.empty:
                spots["x"] += x1
                spots["y"] += y1
                for _, s in spots.iterrows():
                    cx, cy = int(s["x"]), int(s["y"])
                    cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), 1)

        # Show image
        st.image(annotated, caption="🧬 Annotated Image with Adaptive Thresholding", use_column_width=True)

        # Show results
        df = pd.DataFrame(results)
        st.subheader("📊 Plaque Counts")
        st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        # Download
        filename = f"plaque_counts_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=filename, mime="text/csv")
