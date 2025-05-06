from flask import Flask, Response, render_template_string
import cv2
import time
import requests
import webbrowser
import threading
import numpy as np
from camera_simulator import CameraSimulator

app = Flask(__name__)
camera = CameraSimulator()

# MJPEG stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve the HTML
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_camera_frame():
    orientation = [0, -np.pi/4, 0]  # yaw, pitch, roll
    while True:
        position = np.array([0, 0, 5])
        orientation[0] += 0.5

        frame = camera.render_camera_view(position, orientation)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.1)

def open_camera_tab():
    max_retries = 20
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:5050")
            if response.status_code == 200:
                webbrowser.open("http://127.0.0.1:5050")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

# HTML content directly embedded
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Live Camera Feed</title>
    <style>
        body { font-family: Arial; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .camera-container { width: 80vw; height: 80vh; border: 2px solid #ccc; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        #camera-feed { width: 100%; height: 100%; object-fit: contain; }
    </style>
</head>
<body>
    <h1>Live Camera Feed</h1>
    <div class="camera-container">
        <img id="camera-feed" src="http://127.0.0.1:5050/video_feed" alt="Camera Feed">
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    threading.Thread(target=open_camera_tab, daemon=True).start()
    app.run(port=5050)
