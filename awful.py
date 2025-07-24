import streamlit as st
import numpy as np
import pandas as pd
import cv2
import trackpy as tp
from PIL import Image
import datetime

st.set_page_config(page_title="Plaque Counter Pro+", layout="wide")
st.title("🧫 Plaque Counter Pro+")
st.caption("Detect and count plaques using blob or white-pixel detection, with optional overlays on the original image.")

# Sidebar — Parameters
with st.sidebar:
    st.header("🔧 Detection Settings")
    detection_mode = st.selectbox("Detection Mode", ["Blob Detection (Trackpy)", "White Pixel Counting"])
    num_plates = st.number_input("Estimated # of Plates", 1, 20, 4)
    scale_factor = st.slider("Scale Image (0.05–1.0)", 0.05, 1.0, 0.5, 0.05)

    st.subheader("Image Preprocessing")
    invert = st.checkbox("Invert Image", value=True)
    overlay_on_original = st.checkbox("Overlay on Original Image", value=False)

    st.markdown("---")
    if detection_mode == "Blob Detection (Trackpy)":
        st.subheader("Blob Detection Parameters")
        diameter = st.slider("Blob Diameter (px)", 3, 15, 7)
        minmass = st.slider("Minimum Brightness (mass)", 10, 500, 40)
    else:
        st.subheader("White Pixel Thresholding")
        pixel_thresh = st.slider("White Threshold (0–255)", 0, 255, 200)

    st.markdown("---")
    st.subheader("Plate Circle Detection (Hough)")
    blur_k = st.slider("Blur Kernel", 3, 15, 5, step=2)
    param1 = st.slider("Edge Sensitivity (param1)", 10, 100, 50)
    param2 = st.slider("Circle Threshold (param2)", 10, 100, 30)
    min_radius = st.slider("Min Radius", 30, 200, 50)
    max_radius = st.slider("Max Radius", 100, 400, 150)

# Upload image
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)

    # Resize
    new_size = (int(img_array.shape[1] * scale_factor), int(img_array.shape[0] * scale_factor))
    img_resized = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)
    original_resized = img_resized.copy()  # for overlay later

    # Blur + Hough Circles
    img_blur = cv2.GaussianBlur(img_resized, (blur_k, blur_k), 0)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    if circles is None or len(circles[0]) < num_plates:
        st.error("⚠️ Not enough plates detected. Try adjusting detection parameters.")
    else:
        circles = np.uint16(np.around(circles[0]))[:num_plates]
        circles = sorted(circles, key=lambda c: (c[1], c[0]))

        # Choose background for annotation
        annotated = cv2.cvtColor(original_resized if overlay_on_original else img_resized, cv2.COLOR_GRAY2BGR)
        if invert and overlay_on_original:
            annotated = cv2.cvtColor(255 - original_resized, cv2.COLOR_GRAY2BGR)

        results = []

        for i, (x, y, r) in enumerate(circles):
            x1, x2 = max(0, x - r), min(img_resized.shape[1], x + r)
            y1, y2 = max(0, y - r), min(img_resized.shape[0], y + r)
            roi = img_resized[y1:y2, x1:x2]

            if roi.size == 0 or roi.max() == roi.min():
                continue

            # Mask
            mask = np.zeros_like(roi)
            cv2.circle(mask, (r, r), r, 255, -1)
            masked = cv2.bitwise_and(roi, roi, mask=mask)

            # Normalize
            norm = (masked.astype(np.float32) - masked.min()) / (masked.max() - masked.min() + 1e-8)
            norm = (norm * 255).astype(np.uint8)
            if invert:
                norm = 255 - norm

            count = 0

            if detection_mode == "Blob Detection (Trackpy)":
                try:
                    spots = tp.locate(norm, diameter=diameter, minmass=minmass, invert=False)
                    count = len(spots)
                    if not spots.empty:
                        spots["x"] += x1
                        spots["y"] += y1
                        for _, s in spots.iterrows():
                            cx, cy = int(s["x"]), int(s["y"])
                            cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), 1)
                except Exception:
                    pass
            else:
                # White pixel detection
                _, binary = cv2.threshold(norm, pixel_thresh, 255, cv2.THRESH_BINARY)
                count = int(np.sum(binary == 255))

            results.append({"Plate": f"Plate {i+1}", "Count": count})

            # Overlay
            cv2.circle(annotated, (x, y), r, (0, 0, 255), 1)
            cv2.putText(annotated, f"P{i+1}: {count}", (x - 40, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display annotated image
        st.image(annotated, caption="📍 Annotated Image", use_column_width=True)

        df = pd.DataFrame(results)
        st.subheader("📊 Plaque Counts")
        st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        # CSV export
        filename = f"plaque_counts_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=filename, mime="text/csv")
