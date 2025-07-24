import streamlit as st
import numpy as np
import pandas as pd
import cv2
import trackpy as tp
from PIL import Image
import datetime

st.set_page_config(page_title="Plaque Counter", layout="wide")
st.title("🧫 Plaque Counter")
st.caption("Detect plaques inside Petri dishes using adaptive thresholding. Only counts inside detected dish circles.")

# Sidebar: Detection & Display Settings
with st.sidebar:
    st.header("🖼️ Image Settings")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tif"])
    scale = st.slider("Scale Image for Speed", 0.05, 1.0, 0.5, 0.05, help="Reduce size for faster detection")
    invert = st.checkbox("Invert Grayscale", True, help="Enable if plaques are brighter than background")
    show_overlay = st.checkbox("Overlay Results on Original Image", True)

    st.markdown("---")
    st.header("🍽️ Dish Detection (Circles)")
    plates_expected = st.number_input("Expected # of Dishes", 1, 20, 4)
    blur_k = st.slider("Blur Strength (odd)", 3, 15, 5, step=2)
    min_radius = st.slider("Min Dish Radius", 30, 200, 50)
    max_radius = st.slider("Max Dish Radius", 100, 400, 150)
    hough_p1 = st.slider("Hough Edge Param (param1)", 10, 100, 50)
    hough_p2 = st.slider("Hough Circle Param (param2)", 10, 100, 30)

    st.markdown("---")
    st.header("🔬 Plaque Detection (Inside Circles Only)")
    block_size = st.slider("Local Threshold Block Size", 3, 51, 11, step=2)
    c_value = st.slider("C: Offset from Mean", -20, 20, 2)
    diameter = st.slider("Spot Diameter", 3, 15, 7)
    minmass = st.slider("Min Brightness (Mass)", 10, 500, 40)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)

    # Resize
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    img_orig = img.copy()

    # Blur + circle detection
    img_blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=hough_p1, param2=hough_p2,
                               minRadius=min_radius, maxRadius=max_radius)

    if circles is None or len(circles[0]) < plates_expected:
        st.error("❌ Not enough dishes found. Try adjusting radius or edge thresholds.")
    else:
        circles = np.uint16(np.around(circles[0]))[:plates_expected]
        circles = sorted(circles, key=lambda c: (c[1], c[0]))  # sort top-left to bottom-right

        # Annotated image
        display_img = img_orig if show_overlay else img
        if invert and show_overlay:
            display_img = 255 - display_img
        annotated = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        results = []

        for i, (x, y, r) in enumerate(circles):
            # Crop + mask region
            x1, x2 = max(0, x - r), min(img.shape[1], x + r)
            y1, y2 = max(0, y - r), min(img.shape[0], y + r)
            roi = img[y1:y2, x1:x2]

            if roi.size == 0 or roi.max() == roi.min():
                continue

            # Create circular mask
            mask = np.zeros_like(roi)
            cv2.circle(mask, (r, r), r, 255, -1)
            roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

            # Optionally invert
            if invert:
                roi_masked = 255 - roi_masked

            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(roi_masked, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, c_value)

            # Detect spots
            try:
                spots = tp.locate(thresh, diameter=diameter, minmass=minmass, invert=False)
            except Exception:
                continue

            count = len(spots)
            results.append({"Dish": f"Dish {i+1}", "Plaques": count})

            # Draw overlays
            cv2.circle(annotated, (x, y), r, (0, 0, 255), 1)
            cv2.putText(annotated, f"{count}", (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if not spots.empty:
                spots["x"] += x1
                spots["y"] += y1
                for _, s in spots.iterrows():
                    cx, cy = int(s["x"]), int(s["y"])
                    cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), 1)

        # Show output
        st.image(annotated, caption="🔍 Detected Plaques (Inside Dishes Only)", use_column_width=True)

        # Show table
        df = pd.DataFrame(results)
        st.subheader("📊 Results")
        st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        # Download CSV
        filename = f"plaque_counts_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=filename, mime="text/csv")
