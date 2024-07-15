from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

sunglasses = {
    'sunglasses1': cv2.imread('static/images/sunglasses1.png', cv2.IMREAD_UNCHANGED),
    'sunglasses2': cv2.imread('static/images/sunglasses2.png', cv2.IMREAD_UNCHANGED)
}

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]

    img_crop[:] = alpha * img_overlay_crop[:, :, :3] + (1 - alpha) * img_crop

def generate_frames(selected_sunglasses):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda eye: eye[0])
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[1]

                eye_center1 = (ex1 + ew1 // 2, ey1 + eh1 // 2)
                eye_center2 = (ex2 + ew2 // 2, ey2 + eh2 // 2)

                sunglasses_width = int(2.2 * (eye_center2[0] - eye_center1[0]))
                scale_factor = sunglasses_width / sunglasses[selected_sunglasses].shape[1]
                sunglasses_height = int(sunglasses[selected_sunglasses].shape[0] * scale_factor)

                x1 = eye_center1[0] - sunglasses_width // 4
                y1 = eye_center1[1] - sunglasses_height // 2

                resized_sunglasses = cv2.resize(sunglasses[selected_sunglasses], (sunglasses_width, sunglasses_height))

                alpha_mask = resized_sunglasses[:, :, 3] / 255.0

                overlay_image_alpha(roi_color, resized_sunglasses[:, :, :3], x1, y1, alpha_mask)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/basic_feed')
def basic_feed():
    def generate_basic_frames():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_basic_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<selected_sunglasses>')
def video_feed(selected_sunglasses):
    return Response(generate_frames(selected_sunglasses),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
