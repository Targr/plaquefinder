import streamlit as st
import numpy as np
import pandas as pd
import cv2
import trackpy as tp
from PIL import Image
import datetime

st.set_page_config(page_title="Plaque Counter", layout="wide")
st.title("🧫 Plaque Counter Pro")
st.caption("Upload an image of Petri dishes. Automatically detect and count plaques using blob detection.")

# Sidebar — Detection parameters
with st.sidebar:
    st.header("🔧 Detection Settings")

    num_plates = st.number_input("Estimated # of Plates", 1, 20, 4)
    scale_factor = st.slider("Scale Image (0.05–1.0)", 0.05, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.subheader("Blob Detection")
    diameter = st.slider("Blob Diameter (px)", 3, 15, 7)
    minmass = st.slider("Minimum Brightness (mass)", 10, 500, 40)
    invert = st.checkbox("Invert Image", value=True)

    st.markdown("---")
    st.subheader("Circle Detection (Hough)")
    blur_k = st.slider("Blur Kernel (odd)", 3, 15, 5, step=2)
    param1 = st.slider("Edge Sensitivity (param1)", 10, 100, 50)
    param2 = st.slider("Circle Threshold (param2)", 10, 100, 30)
    min_radius = st.slider("Min Radius", 30, 200, 50)
    max_radius = st.slider("Max Radius", 100, 400, 150)

    st.markdown("---")
    st.subheader("💡 Visual Settings")
    draw_rings = st.checkbox("Draw Dish Rings", value=True)
    theme = st.radio("Color Theme", ["Light", "Dark"], horizontal=True)

# File uploader
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)

    # Resize image
    new_size = (int(img_array.shape[1] * scale_factor), int(img_array.shape[0] * scale_factor))
    img_resized = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)

    # Preprocessing
    img_blur = cv2.GaussianBlur(img_resized, (blur_k, blur_k), 0)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    if circles is None or len(circles[0]) < num_plates:
        st.error("⚠️ Not enough plates detected. Try adjusting Hough settings or image scale.")
    else:
        circles = np.uint16(np.around(circles[0]))[:num_plates]
        circles = sorted(circles, key=lambda c: (c[1], c[0]))

        # Prepare image
        if theme == "Dark":
            annotated = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        else:
            inverted = 255 - img_resized if invert else img_resized
            annotated = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

        results = []

        for i, (x, y, r) in enumerate(circles):
            # Crop and mask ROI
            x1, x2 = max(0, x - r), min(img_resized.shape[1], x + r)
            y1, y2 = max(0, y - r), min(img_resized.shape[0], y + r)
            roi = img_resized[y1:y2, x1:x2]

            if roi.size == 0 or roi.max() == roi.min():
                continue

            mask = np.zeros_like(roi)
            cv2.circle(mask, (r, r), r, 255, -1)
            masked = cv2.bitwise_and(roi, roi, mask=mask)

            norm = (masked.astype(np.float32) - masked.min()) / (masked.max() - masked.min() + 1e-8)
            norm = (norm * 255).astype(np.uint8)
            if invert:
                norm = 255 - norm

            try:
                spots = tp.locate(norm, diameter=diameter, minmass=minmass, invert=False)
            except Exception:
                continue

            count = len(spots)
            results.append({"Plate": f"Plate {i+1}", "Count": count})

            # Draw ring and count
            if draw_rings:
                cv2.circle(annotated, (x, y), r, (0, 0, 255), 1)
                cv2.circle(annotated, (x, y), int(r * 0.95), (0, 255, 255), 1)

            cv2.putText(annotated, f"P{i+1}: {count}", (x - 40, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Mark blobs
            if not spots.empty:
                spots["x"] += x1
                spots["y"] += y1
                for _, s in spots.iterrows():
                    cx, cy = int(s["x"]), int(s["y"])
                    cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), 1)

        st.image(annotated, caption="🧪 Annotated Plate Image", use_column_width=True)

        df = pd.DataFrame(results)
        st.subheader("📊 Plaque Counts")
        st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        filename = f"plaque_counts_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=filename, mime="text/csv")
