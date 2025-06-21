import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp
from tensorflow.keras.models import load_model

# ========== Load Model ==========
@st.cache_resource
def load_cnn_model():
    return load_model("cnn1_model.h5")

cnn_model = load_cnn_model()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ========== UI ==========
st.title("💁‍♂️ Face Classification App - Handsome vs Ugly")
st.markdown("Upload gambar wajah atau gunakan kamera, lalu model akan mengklasifikasikan wajah sebagai Handsome atau Ugly.")

input_type = st.radio("Pilih sumber gambar:", ["Upload File", "Kamera"])

if input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif input_type == "Kamera":
    camera_image = st.camera_input("Ambil gambar dari kamera")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
else:
    image = None

# ========== Proses Gambar ==========
if 'image' in locals() and image is not None:
    img_np = np.array(image)
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        results = face_detector.process(img_np)

        if not results.detections:
            st.warning("❌ Tidak ada wajah terdeteksi.")
        else:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img_np.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                box_w, box_h = int(bbox.width * w), int(bbox.height * h)

                face_img = img_np[y:y+box_h, x:x+box_w]
                if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                    continue

                face_resized = np.array(Image.fromarray(face_img).resize((128, 128))) / 255.0
                input_tensor = np.expand_dims(face_resized.astype(np.float32), axis=0)

                pred = cnn_model.predict(input_tensor)[0][0]
                label = "Handsome" if pred >= 0.5 else "Ugly"
                color = "green" if pred >= 0.5 else "red"

                draw.rectangle([x, y, x+box_w, y+box_h], outline=color, width=3)
                draw.text((x, y - 10), f"{label} ({pred:.2f})", fill=color)

            st.image(img_copy, caption="Hasil Klasifikasi", use_column_width=True)
