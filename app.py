import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2
import tempfile
import re
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher
import os

logo_path = os.path.join("assets", "logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=200, use_container_width=False)
else:
    st.warning("⚠️ Logo not found. Please check assets/logo.png")

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="wide"
)

# ----------------------------
# Helpers
# ----------------------------
def clean_text(text):
    """Keep only A-Z and 0-9"""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def preprocess_crop(crop):
    """Improve OCR accuracy by resizing + grayscale + threshold"""
    if crop is None or crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def similar(a, b):
    """Check similarity ratio between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def finalize_plates(plates):
    """Remove duplicates & noisy OCR results"""
    cleaned = [clean_text(p) for p in plates if len(clean_text(p)) >= 4]
    counts = Counter(cleaned)
    final = []

    for plate, count in counts.items():
        if count < 2:  # must appear in at least 2 frames
            continue

        is_duplicate = False
        for f in final:
            if similar(plate, f) > 0.8:  # 80% similarity = duplicate
                is_duplicate = True
                break

        if not is_duplicate:
            final.append(plate)

    return final

# ----------------------------
# Load YOLO model & OCR
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.title("⚙️ Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Store detected plates in session
if "plate_list" not in st.session_state:
    st.session_state.plate_list = []

# ----------------------------
# Header (Logo + Title)
# ----------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.image("assets/logo.png", width=200, use_container_width=False)  # Bigger logo, no border
with col2:
    st.markdown("""
        <div class="header-title">
            <h1>🚘 License Plate Recognition</h1>
            <p>Upload images or videos, detect plates, and extract numbers with YOLO + EasyOCR.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Tabs
# ----------------------------
tab1, = st.tabs(["📤 Upload"])  # Only Upload tab visible at start

with tab1:
    file_type = st.radio("Choose input type", ["Image", "Video"])

    # ----------------------------
    # Image Upload
    # ----------------------------
    if file_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Detecting license plate..."):
                results = model.predict(np.array(image), conf=conf)

            for r in results:
                im_array = r.plot(show=False)
                im_rgb = Image.fromarray(im_array[..., ::-1])
                st.image(im_rgb, caption="Detected Plates", use_container_width=True)

                if r.boxes:
                    st.subheader("📋 Detection Details")
                    for i, box in enumerate(r.boxes):
                        cls = model.names[int(box.cls)]
                        conf_score = float(box.conf)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        crop = np.array(image)[y1:y2, x1:x2]
                        crop = preprocess_crop(crop)

                        ocr_result = reader.readtext(crop)
                        if ocr_result:
                            plate_number = clean_text(ocr_result[0][1])
                        else:
                            plate_number = None

                        st.write(f"🔲 **{cls}** — Confidence: {conf_score:.2f}")
                        if plate_number:
                            st.write(f"📖 **Plate Number:** {plate_number}")
                            st.image(crop, caption=f"Cropped Plate {i+1}", use_container_width=False)
                            st.session_state.plate_list.append(plate_number)
                        else:
                            st.warning("⚠️ No readable plate text detected.")

    # ----------------------------
    # Video Upload
    # ----------------------------
    else:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            frame_idx = 0
            with st.spinner("🎥 Processing video..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % 20 != 0:  # process every 20th frame
                        continue

                    results = model.predict(frame, conf=conf)

                    for r in results:
                        im_array = r.plot(show=False)
                        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                        stframe.image(im_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

                        if r.boxes:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                crop = frame[y1:y2, x1:x2]
                                crop = preprocess_crop(crop)

                                ocr_result = reader.readtext(crop)
                                if ocr_result:
                                    plate_number = clean_text(ocr_result[0][1])
                                    st.session_state.plate_list.append(plate_number)

            cap.release()

# ----------------------------
# Results Section
# ----------------------------
if st.session_state.plate_list:
    # Show Results tab only if plates exist
    tab2, = st.tabs(["📊 Results"])   # <-- fixed unpacking

    with tab2:
        st.subheader("✅ Final Detected Plate Numbers")
        final_plates = finalize_plates(st.session_state.plate_list)

        if final_plates:
            for i, plate in enumerate(final_plates, 1):
                st.success(f"🚘 Plate {i}: {plate}")
        else:
            st.warning("⚠️ No valid plate numbers detected.")

        # Save to CSV
        df = pd.DataFrame(final_plates, columns=["Detected Plates"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Detected Plates (CSV)",
            data=csv,
            file_name="detected_plates.csv",
            mime="text/csv",
        )

