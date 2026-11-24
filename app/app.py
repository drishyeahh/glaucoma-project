import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

st.set_page_config(page_title="Glaucoma Screening", layout="centered")
st.title("🩺 Glaucoma Screening System")
st.write("Upload a fundus image to detect the Optic Disc and screen for Glaucoma.")

@st.cache_resource
def load_models():
    yolo_model = YOLO("models/od_localizer/best.pt")
    classifier_model = tf.keras.models.load_model("models/classifier_ULTIMATE_MASTER.keras")
    return yolo_model, classifier_model

yolo_model, classifier_model = load_models()

# NEW: Checkbox to skip YOLO for pre-cropped images
is_cropped = st.checkbox("Image is already cropped to the Optic Disc (Bypass YOLO)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Run Screening"):
        if is_cropped:
            st.info("Bypassing YOLO. Sending full image to classifier.")
            resized = cv2.resize(image_rgb, (224, 224))
            display_img = image_rgb.copy()
        else:
            with st.spinner("Detecting Optic Disc..."):
                # Increased confidence to 0.6 to avoid false positives
                results = yolo_model(image_rgb, conf=0.6, verbose=False)
                
                if len(results[0].boxes) == 0:
                    st.warning("⚠️ YOLO could not detect the Optic Disc. Using the full image as the ROI.")
                    resized = cv2.resize(image_rgb, (224, 224))
                    display_img = image_rgb.copy()
                else:
                    box = results[0].boxes[0].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    
                    w, h = x2 - x1, y2 - y1
                    pad_x, pad_y = int(w * 0.3), int(h * 0.3)
                    
                    x1_pad = max(0, x1 - pad_x)
                    y1_pad = max(0, y1 - pad_y)
                    x2_pad = min(image_rgb.shape[1], x2 + pad_x)
                    y2_pad = min(image_rgb.shape[0], y2 + pad_y)
                    
                    cropped = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
                    resized = cv2.resize(cropped, (224, 224))
                    
                    display_img = image_rgb.copy()
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    st.image(display_img, caption="Detected Optic Disc", use_container_width=True)
        
        st.image(resized, caption="ROI sent to Classifier (224x224)", width=224)
        
        with st.spinner("Analyzing ROI..."):
            input_tensor = np.expand_dims(resized, axis=0)
            prediction = classifier_model.predict(input_tensor)[0][0]
            
            if prediction < 0.5:
                st.error(f"⚠️ **Result: Glaucoma Suspected** (Confidence: {(1 - prediction)*100:.2f}%)")
            else:
                st.success(f"✅ **Result: Healthy** (Confidence: {prediction*100:.2f}%)")