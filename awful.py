import streamlit as st
import numpy as np
import pandas as pd
import cv2
import trackpy as tp
from PIL import Image
import datetime

st.set_page_config(page_title="Plaque Counter", layout="wide")
st.title("🧫 Plaque Counter")
st.caption("Upload an image of petri dishes and count plaques using Trackpy.")

# Sidebar: parameters
with st.sidebar:
    st.header("🔧 Detection Settings")
    num_plates = st.number_input("Estimated # of Plates", 1, 12, 4, help="Adjust if detection seems off")
    diameter = st.slider("Blob Diameter (px)", 3, 15, 7, help="Estimated size of a plaque in pixels")
    minmass = st.slider("Minimum Brightness (mass)", 10, 300, 40, help="Lower = more sensitivity")
    invert = st.checkbox("Invert Contrast", value=True, help="Enable if plaques are brighter than background")

    st.markdown("---")
    st.caption("Tip: default settings usually work well.")

# Upload image
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect plates with HoughCircles
    img_blur = cv2.medianBlur(img_array, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150,
                               param1=50, param2=30, minRadius=100, maxRadius=300)

    if circles is None or len(circles[0]) < num_plates:
        st.error("⚠️ Fewer plates detected than expected. Try adjusting blur, lighting, or plate count.")
    else:
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda c: (c[1], c[0]))[:num_plates]
        annotated = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        plate_data = []

        for i, (x, y, r) in enumerate(circles):
            # Bound crop region safely
            x1, x2 = max(0, x - r), min(img_array.shape[1], x + r)
            y1, y2 = max(0, y - r), min(img_array.shape[0], y + r)
            roi = img_array[y1:y2, x1:x2]

            if roi.size == 0 or roi.max() == roi.min():
                continue

            # Mask out circular region
            mask = np.zeros_like(roi)
            cv2.circle(mask, (r, r), r, 255, -1)
            roi = cv2.bitwise_and(roi, roi, mask=mask)

            # Normalize
            norm = roi.astype(np.float32)
            norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
            norm = (norm * 255).astype(np.uint8)

            if invert:
                norm = 255 - norm

            try:
                spots = tp.locate(norm, diameter=diameter, minmass=minmass, invert=False)
            except Exception:
                continue

            if not spots.empty:
                spots["x"] += x1
                spots["y"] += y1
                for _, s in spots.iterrows():
                    cx, cy = int(s["x"]), int(s["y"])
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), 1)

            count = len(spots)
            cv2.putText(annotated, f"P{i+1}: {count}", (x - 50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            plate_data.append({"Plate": f"Plate {i+1}", "Count": count})

        # Display annotated image
        st.image(annotated, caption="Detected Plaques", use_column_width=True)

        # Results table
        df = pd.DataFrame(plate_data)
        st.subheader("📊 Plaque Counts per Plate")
        st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        # Export CSV
        filename = f"plaque_counts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=filename, mime="text/csv")
