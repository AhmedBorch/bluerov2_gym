import cv2
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Open default webcam (0 = default camera)
camera = cv2.VideoCapture(0)

def generate_webcam_feed():
    while True:
        success, frame = camera.read()
        if not success:
            continue

        # Encode the frame in JPEG format
        success, jpeg = cv2.imencode('.jpg', frame)
        if not success:
            continue

        # Yield the frame as part of MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template_string(open('templates/index.html').read())

@app.route('/video_feed')
def video_feed():
    return Response(generate_webcam_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, threaded=True)
