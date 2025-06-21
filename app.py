from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# =======================
# Load model & assets
# =======================
model = load_model('cnn1_model.h5')
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

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
            face_roi = img_rgb[y:y+h, x:x+w]
            face_res = cv2.resize(face_roi, (128, 128)) / 255.0
            img_input = np.expand_dims(face_res.astype(np.float32), axis=0)

            results = face_mesh.process(face_roi)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                xs, ys = [], []
                for lm in face_landmarks.landmark:
                    xs.append((lm.x * w))
                    ys.append((lm.y * h))

                if len(xs) != len(ys) or len(xs) == 0:
                    continue

                xs = np.array(xs)
                ys = np.array(ys)

                xs_norm = xs * 128.0 / max(w, 1)
                ys_norm = ys * 128.0 / max(h, 1)
                landmark_input = np.expand_dims(np.vstack([xs_norm, ys_norm]).T.flatten().astype(np.float32), axis=0)

                pred = model.predict([img_input, landmark_input])[0][0]
                label = "Handsome" if pred >= 0.5 else "Ugly"
                color = (0, 255, 0) if pred >= 0.5 else (0, 0, 255)

                cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_copy, f"{label} ({pred:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', img_copy)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
