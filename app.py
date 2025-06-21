import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ===================
# Load model & assets
# ===================

@st.cache_resource
def load_cnn_model():
    return load_model("cnn1_model.h5")

@st.cache_resource
def load_landmark_predictor():
    return dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cnn_model = load_cnn_model()
predictor = load_landmark_predictor()
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ===================
# UI
# ===================
st.title("💁‍♂️ Face Classification App - Handsome vs Ugly")
st.markdown("Upload gambar wajah atau gunakan kamera, lalu model akan mengklasifikasikan wajah sebagai Handsome atau Ugly.")

# Pilihan input
input_type = st.radio("Pilih sumber gambar:", ["Upload File", "Kamera"])

# Ambil gambar
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

# Proses jika ada gambar
if 'image' in locals() and image is not None:
    img_rgb = np.array(image)
    img_copy = img_rgb.copy()

    faces = haar_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("❌ Tidak ada wajah terdeteksi.")
    else:
        for (x, y, w, h) in faces:
            try:
                rect = dlib.rectangle(x, y, x + w, y + h)
                shape = predictor(img_rgb, rect)
                landmarks = [(pt.x, pt.y) for pt in shape.parts()]
                landmarks_flat = np.array(landmarks).flatten()
                landmark_input = np.expand_dims(landmarks_flat.astype(np.float32), axis=0)

                face_img = img_rgb[y:y+h, x:x+w]
                if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                    continue
                face_res = cv2.resize(face_img, (128, 128)) / 255.0
                img_input = np.expand_dims(face_res.astype(np.float32), axis=0)

                pred = cnn_model.predict([img_input, landmark_input])[0][0]
                label = "Handsome" if pred >= 0.5 else "Ugly"
                color = (0,255,0) if pred >= 0.5 else (255,0,0)

                # Visualisasi
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_copy, f"{label} ({pred:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                st.error(f"⚠️ Gagal proses wajah di ({x},{y}): {e}")
                continue

        st.image(img_copy, caption="Hasil Klasifikasi", use_column_width=True)
