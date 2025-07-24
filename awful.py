import streamlit as st
import numpy as np
import pandas as pd
import cv2
import trackpy as tp
from PIL import Image
import datetime

st.set_page_config(page_title="Plaque Counter", layout="wide")
st.title("🧫 Plaque Counter")
st.caption("Upload a petri dish image to count plaques/colonies with adaptive thresholding and Trackpy.")

# Sidebar Parameters
with st.sidebar:
    st.header("🔧 Detection Settings")
    num_plates = st.number_input("Estimated # of Plates", 1, 12, 4)
    diameter = st.slider("Feature Diameter (px)", 3, 15, 7)
    minmass = st.slider("Trackpy Minmass", 10, 500, 60)
    brightness_filter = st.slider("Minimum Brightness After Detection", 0, 10000, 1000)
    invert = st.checkbox("Invert Image", value=True)
    
    st.markdown("### 🧪 Thresholding")
    adaptive_blocksize = st.slider("Adaptive Threshold Block Size", 3, 101, 51, step=2)
    adaptive_C = st.slider("Adaptive Threshold C", -20, 20, 5)
    
    st.markdown("### 🖼️ Output Options")
    overlay_on_original = st.checkbox("Overlay Circles on Original", value=True)
    downscale_factor = st.slider("Downscale Factor", 0.1, 1.0, 0.25, step=0.05)

# Upload
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    orig_array = np.array(image)

    # Downscale image
    img_array = cv2.resize(orig_array, (0, 0), fx=downscale_factor, fy=downscale_factor)

    # Blur for circle detection
    blur = cv2.medianBlur(img_array, 5)

    # Detect plates
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150,
                               param1=50, param2=30, minRadius=100, maxRadius=300)

    if circles is None or len(circles[0]) < num_plates:
        st.error("⚠️ Fewer plates detected than expected.")
    else:
        circles = np.uint16(np.around(circles[0]))[:num_plates]
        circles = sorted(circles, key=lambda c: (c[1], c[0]))
        
        # Overlay image: either original or grayscale copy
        overlay_base = cv2.cvtColor(orig_array, cv2.COLOR_GRAY2BGR) if overlay_on_original else cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        results = []

        for i, (x, y, r) in enumerate(circles):
            # Scale back to original if needed
            scale = 1 / downscale_factor
            x_full, y_full, r_full = int(x * scale), int(y * scale), int(r * scale)

            # Crop ROI
            x1, x2 = max(0, x_full - r_full), min(orig_array.shape[1], x_full + r_full)
            y1, y2 = max(0, y_full - r_full), min(orig_array.shape[0], y_full + r_full)
            roi = orig_array[y1:y2, x1:x2]

            if roi.size == 0 or roi.max() == roi.min():
                continue

            # Circular mask
            mask = np.zeros_like(roi)
            cv2.circle(mask, (r_full, r_full), r_full, 255, -1)
            roi = cv2.bitwise_and(roi, roi, mask=mask)

            # Invert
            if invert:
                roi = 255 - roi

            # Adaptive thresholding
            blocksize = adaptive_blocksize if adaptive_blocksize % 2 == 1 else adaptive_blocksize + 1
            roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, blocksize, adaptive_C)

            # Normalize
            norm = roi_thresh.astype(np.uint8)

            # Detect features
            try:
                spots = tp.locate(norm, diameter=diameter, minmass=minmass, invert=False)
            except Exception:
                continue

            if not spots.empty:
                # Filter by brightness (mass)
                spots = spots[spots["mass"] >= brightness_filter]

                spots["x"] += x1
                spots["y"] += y1

                for _, s in spots.iterrows():
                    cx, cy = int(s["x"]), int(s["y"])
                    cv2.circle(overlay_base, (cx, cy), 2, (0, 255, 0), 1)

            count = len(spots)
            results.append({"Plate": f"Plate {i+1}", "Count": count})

            # Draw plate ring and label
            cv2.circle(overlay_base, (x_full, y_full), r_full, (255, 0, 255), 2)
            cv2.putText(overlay_base, f"P{i+1}: {count}", (x_full - 40, y_full - r_full - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        st.image(overlay_base, caption="Detected Plaques", use_column_width=True)

        # Table
        df = pd.DataFrame(results)
        st.subheader("📊 Plaque Counts")
        st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        # Download CSV
        filename = f"plaque_counts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=filename, mime="text/csv")
