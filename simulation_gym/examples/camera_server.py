from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import time
import requests
import webbrowser
import threading
import numpy as np
from bluerov2_gym.envs.core.camera.camera_simulator import CameraSimulator
import os
import signal

app = Flask(__name__)

# Shutdown function
def shutdown():
    print("Shutting down the server...")
    os.kill(os.getpid(), signal.SIGINT)

camera = CameraSimulator()

latest_state = {
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 0.0],  # Euler angles
    "target_buoy": [1.0, 0.0, 0.0],
    "trajectory_markers": [[0.0, 0.0, 0.0]]
}

@app.route("/update_state", methods=["POST"])
def update_state():
    global latest_state
    data = request.get_json()
    latest_state["position"] = data.get("position", latest_state["position"])
    latest_state["orientation"] = data.get("orientation", latest_state["orientation"])
    latest_state["target_buoy"] = data.get("target_buoy", latest_state["target_buoy"])
    latest_state["trajectory_markers"] = data.get("trajectory_markers", latest_state["trajectory_markers"] )
    return jsonify({"status": "ok"})


# MJPEG stream
@app.route('/video_feed')
def video_feed():
    print(" /video_feed requested")
    return Response(generate_camera_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve the HTML
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    """Handle server shutdown manually"""
    shutdown()
    return 'Server shutting down...'

def generate_camera_frame():
    print("🎥 Starting to generate frames")
    
    while True:
        #edited here the latest_state
        position = np.array(latest_state["position"])
        orientation = np.array(latest_state["orientation"])  # -roll,-pitch,-yaw
        target_buoy = latest_state["target_buoy"]
        trajectory_markers = np.array(latest_state["trajectory_markers"])
        #orientation[0] += 0.5

        frame = camera.render_camera_view(position, orientation,new_buoy_positions=target_buoy,trajectory_markers=trajectory_markers)#change later to multiple target buoys
        if frame is None or frame.size == 0:
            print(" Invalid frame received!")

        if frame.shape != (480, 640, 3):  # Or whatever shape you're expecting
            print(f"Unexpected frame shape: {frame.shape}")
        # If the image is in RGB format and you want to ensure it's in BGR before encoding
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, png = cv2.imencode('.png', frame_bgr)

        if not ret:
            print(" Failed to encode frame to JPEG")
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + png.tobytes() + b'\r\n')
        time.sleep(0.1)

    print("Ending camera server...")
    shutdown()  # Call the shutdown function

def open_camera_tab():
    max_retries = 20
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:5050")
            if response.status_code == 200:
                print("Server is ready, opening browser...")
                webbrowser.open("http://127.0.0.1:5050")
                return
        except requests.exceptions.ConnectionError:
            print(f"Retrying... {i + 1}/{max_retries}")
        time.sleep(0.5)  # Wait a bit before retrying
    print(" Failed to open browser tab: server didn't respond.")


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
