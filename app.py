from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# =======================
# Load model & aset once
# =======================
model = load_model('cnn1_model.h5')

predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError("File shape_predictor_68_face_landmarks.dat tidak ditemukan!")

predictor = dlib.shape_predictor(predictor_path)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ====================
# Kamera real-time loop
# ====================
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("❌ Gagal membuka kamera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_copy = frame.copy()

        faces = haar_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            try:
                rect = dlib.rectangle(x, y, x + w, y + h)
                shape = predictor(img_rgb, rect)
                landmarks = [(pt.x, pt.y) for pt in shape.parts()]
                landmarks_flat = np.array(landmarks).flatten()

                face_img = img_rgb[y:y+h, x:x+w]
                if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                    continue

                face_res = cv2.resize(face_img, (128, 128)) / 255.0
                img_input = np.expand_dims(face_res.astype(np.float32), axis=0)

                img_h, img_w = face_img.shape[:2]
                xs = (landmarks_flat[0::2] - x) * 128.0 / max(img_w, 1)
                ys = (landmarks_flat[1::2] - y) * 128.0 / max(img_h, 1)
                landmark_input = np.expand_dims(np.vstack([xs, ys]).T.flatten().astype(np.float32), axis=0)

                pred = model.predict([img_input, landmark_input])[0][0]
                label = "Handsome" if pred >= 0.5 else "Ugly"
                color = (0, 255, 0) if pred >= 0.5 else (0, 0, 255)

                cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_copy, f"{label} ({pred:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except:
                continue

        # Encode frame jadi JPEG dan kirim ke browser
        ret, buffer = cv2.imencode('.jpg', img_copy)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ====================
# Route
# ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ====================
# Run
# ====================
if __name__ == '__main__':
    app.run(debug=True)
