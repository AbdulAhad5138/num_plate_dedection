import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr
import numpy as np

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="centered"
)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Load OCR Reader
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])  # English OCR

reader = load_reader()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.image("assets/logo.png", use_container_width=True)  # ✅ Logo added back
st.sidebar.title("⚙️ Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# ----------------------------
# Main UI
# ----------------------------
st.title("🚘 License Plate Recognition")
st.write("Upload an image of a vehicle, the model will detect the license plate and extract its number.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run YOLO inference
    with st.spinner("🔍 Detecting license plate..."):
        results = model.predict(image, conf=conf)

    # Process results
    for r in results:
        im_array = r.plot(show=False)  # Draw bounding boxes
        im_rgb = Image.fromarray(im_array[..., ::-1])  # Convert BGR → RGB
        st.image(im_rgb, caption="Detected Plates", use_container_width=True)

        # OCR on detected plates
        if r.boxes:
            st.subheader("📋 Detection Details")
            for i, box in enumerate(r.boxes):
                cls = model.names[int(box.cls)]
                conf_score = float(box.conf)

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                crop = np.array(image)[int(y1):int(y2), int(x1):int(x2)]

                # OCR
                ocr_result = reader.readtext(crop)
                plate_number = ocr_result[0][1] if len(ocr_result) > 0 else "Not detected"

                st.write(f"🔲 **{cls}** — Confidence: {conf_score:.2f}")
                st.write(f"📖 **Plate Number:** {plate_number}")

                # Show cropped plate
                st.image(crop, caption=f"Cropped Plate {i+1}", use_container_width=False)
        else:
            st.warning("⚠️ No license plate detected.")
else:
    st.info("👆 Please upload an image to start.")
