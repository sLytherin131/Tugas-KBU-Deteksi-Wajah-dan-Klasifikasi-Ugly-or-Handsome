import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from tensorflow.keras.models import load_model

# ===================
# Load model
# ===================
@st.cache_resource
def load_cnn_model():
    return load_model("cnn1_model.h5")

cnn_model = load_cnn_model()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# ===================
# UI
# ===================
st.title("💁‍♂️ Face Classification App - Handsome vs Ugly (with Mediapipe)")
st.markdown("Upload gambar wajah atau gunakan kamera. Model akan mengklasifikasikan wajah sebagai *Handsome* atau *Ugly*.")

input_type = st.radio("Pilih sumber gambar:", ["Upload File", "Kamera"])

image = None
if input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif input_type == "Kamera":
    camera_image = st.camera_input("Ambil gambar dari kamera")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

if image:
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        st.warning("❌ Tidak ada wajah terdeteksi.")
    else:
        annotated_img = img_rgb.copy()
        h, w, _ = img_rgb.shape
        for face_landmarks in results.multi_face_landmarks:
            # Ambil 468 titik landmark wajah dari mediapipe
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
            landmarks_flat = np.array(landmarks).flatten()
            landmark_input = np.expand_dims(landmarks_flat.astype(np.float32), axis=0)

            # Bounding box kasar dari wajah
            x_coords = [pt[0] for pt in landmarks]
            y_coords = [pt[1] for pt in landmarks]
            x_min, y_min = max(min(x_coords), 0), max(min(y_coords), 0)
            x_max, y_max = min(max(x_coords), w), min(max(y_coords), h)
            face_img = img_rgb[y_min:y_max, x_min:x_max]

            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                continue

            face_res = cv2.resize(face_img, (128, 128)) / 255.0
            img_input = np.expand_dims(face_res.astype(np.float32), axis=0)

            pred = cnn_model.predict([img_input, landmark_input])[0][0]
            label = "Handsome" if pred >= 0.5 else "Ugly"
            color = (0, 255, 0) if pred >= 0.5 else (255, 0, 0)

            cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(annotated_img, f"{label} ({pred:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        st.image(annotated_img, caption="Hasil Klasifikasi", use_column_width=True)
