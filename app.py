"""
Cattle Measurement System - Backend
Description: AI-based Animal Type Classification System for cattle and buffaloes
Uses MediaPipe for pose estimation and reference object for accurate measurements
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
from imutils import perspective, contours
import imutils
import json
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import math

class CattleMeasurementSystem:
    def __init__(self, reference_dimension_cm: float = 30.0):
        """
        Initialize the cattle measurement system
        
        Args:
            reference_dimension_cm: Known dimension of reference object in cm (default: 30cm)
        """
        self.reference_dimension_cm = reference_dimension_cm
        self.pixel_per_cm = None
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        return blur, edged
    
    def detect_reference_object(self, image: np.ndarray, min_area: int = 1000) -> Optional[float]:
        _, edged = self.preprocess_image(image)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [x for x in cnts if cv2.contourArea(x) > min_area]
        if len(cnts) == 0:
            return None
        (cnts, _) = contours.sort_contours(cnts)
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        self.pixel_per_cm = dist_in_pixel / self.reference_dimension_cm
        return self.pixel_per_cm
    
    def detect_pose_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = {}
        h, w, _ = image.shape
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x * w,
                'y': landmark.y * h,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        return landmarks
    
    def calculate_cattle_height(self, landmarks: Dict, image_shape: Tuple) -> float:
        if self.pixel_per_cm is None:
            raise ValueError("Pixel per cm ratio not calibrated. Run detect_reference_object first.")
        y_coords = [landmarks[i]['y'] for i in landmarks.keys() if landmarks[i]['visibility'] > 0.5]
        if len(y_coords) < 2:
            return 0.0
        top_y = min(y_coords)
        bottom_y = max(y_coords)
        height_pixels = bottom_y - top_y
        height_cm = height_pixels / self.pixel_per_cm
        return height_cm
    
    def calculate_wither_height(self, landmarks: Dict) -> float:
        if self.pixel_per_cm is None:
            raise ValueError("Pixel per cm ratio not calibrated.")
        left_shoulder = landmarks.get(11)
        right_shoulder = landmarks.get(12)
        if not left_shoulder or not right_shoulder:
            return 0.0
        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        bottom_landmarks = [landmarks.get(i) for i in [27, 28, 29, 30, 31, 32] if landmarks.get(i)]
        if not bottom_landmarks:
            return 0.0
        bottom_y = max([lm['y'] for lm in bottom_landmarks if lm['visibility'] > 0.5])
        wither_height_pixels = bottom_y - shoulder_y
        wither_height_cm = wither_height_pixels / self.pixel_per_cm
        return wither_height_cm
    
    def calculate_chest_width(self, landmarks: Dict) -> float:
        if self.pixel_per_cm is None:
            raise ValueError("Pixel per cm ratio not calibrated.")
        left_shoulder = landmarks.get(11)
        right_shoulder = landmarks.get(12)
        if not left_shoulder or not right_shoulder:
            return 0.0
        chest_width_pixels = euclidean(
            [left_shoulder['x'], left_shoulder['y']],
            [right_shoulder['x'], right_shoulder['y']]
        )
        chest_width_cm = chest_width_pixels / self.pixel_per_cm
        return chest_width_cm
    
    def calculate_rump_angle(self, landmarks: Dict) -> float:
        left_hip = landmarks.get(23)
        right_hip = landmarks.get(24)
        if not left_hip or not right_hip:
            return 0.0
        hip_mid_x = (left_hip['x'] + right_hip['x']) / 2
        hip_mid_y = (left_hip['y'] + right_hip['y']) / 2
        left_ankle = landmarks.get(27)
        right_ankle = landmarks.get(28)
        if not left_ankle or not right_ankle:
            return 0.0
        tail_base_x = hip_mid_x
        tail_base_y = hip_mid_y + abs(hip_mid_y - (left_ankle['y'] + right_ankle['y']) / 2) * 0.3
        dx = tail_base_x - hip_mid_x
        dy = tail_base_y - hip_mid_y
        angle_rad = math.atan2(dy, dx) if dx != 0 else 0
        angle_deg = abs(math.degrees(angle_rad))
        rump_angle = 90 - angle_deg if angle_deg < 90 else angle_deg - 90
        return rump_angle
    
    def calculate_body_length(self, landmarks: Dict) -> float:
        if self.pixel_per_cm is None:
            raise ValueError("Pixel per cm ratio not calibrated.")
        left_shoulder = landmarks.get(11)
        right_shoulder = landmarks.get(12)
        left_hip = landmarks.get(23)
        right_hip = landmarks.get(24)
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0
        shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        shoulder_mid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        hip_mid_x = (left_hip['x'] + right_hip['x']) / 2
        hip_mid_y = (left_hip['y'] + right_hip['y']) / 2
        body_length_pixels = euclidean([shoulder_mid_x, shoulder_mid_y],[hip_mid_x, hip_mid_y])
        body_length_cm = body_length_pixels / self.pixel_per_cm
        return body_length_cm
    
    def process_cattle_image(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        pixel_ratio = self.detect_reference_object(image)
        if pixel_ratio is None:
            raise ValueError("Reference object not detected. Ensure reference object is leftmost in image.")
        landmarks = self.detect_pose_landmarks(image)
        if landmarks is None:
            raise ValueError("Could not detect cattle pose landmarks. Ensure cattle is clearly visible.")
        measurements = {
            'total_height_cm': round(self.calculate_cattle_height(landmarks, image.shape), 2),
            'wither_height_cm': round(self.calculate_wither_height(landmarks), 2),
            'chest_width_cm': round(self.calculate_chest_width(landmarks), 2),
            'rump_angle_deg': round(self.calculate_rump_angle(landmarks), 2),
            'body_length_cm': round(self.calculate_body_length(landmarks), 2),
            'pixel_per_cm': round(self.pixel_per_cm, 4),
            'reference_dimension_cm': self.reference_dimension_cm
        }
        result = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'image_dimensions': {'height': image.shape[0],'width': image.shape[1]},
            'measurements': measurements,
            'status': 'success'
        }
        return result
    
    def draw_measurements(self, image_path: str, output_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        result = self.process_cattle_image(image_path)
        measurements = result['measurements']
        landmarks = self.detect_pose_landmarks(image)
        if landmarks:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)
        thickness = 2
        text_lines = [
            f"Total Height: {measurements['total_height_cm']} cm",
            f"Wither Height: {measurements['wither_height_cm']} cm",
            f"Chest Width: {measurements['chest_width_cm']} cm",
            f"Body Length: {measurements['body_length_cm']} cm",
            f"Rump Angle: {measurements['rump_angle_deg']} deg"
        ]
        for i, line in enumerate(text_lines):
            cv2.putText(image, line, (10, y_offset + i * 30), font, font_scale, color, thickness)
        cv2.imwrite(output_path, image)
        return output_path
    
    def save_results(self, result: Dict, output_json_path: str) -> str:
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=4)
        return output_json_path
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


# Flask API
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os

# NEW: enable CORS
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # <---- FIXED: allow frontend requests

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

measurement_system = CattleMeasurementSystem(reference_dimension_cm=30.0)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Cattle Measurement API'})

@app.route('/api/measure', methods=['POST'])
def measure_cattle():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        reference_dim = float(request.form.get('reference_dimension', 30.0))
        measurement_system.reference_dimension_cm = reference_dim
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        result = measurement_system.process_cattle_image(filepath)
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"annotated_{unique_filename}")
        measurement_system.draw_measurements(filepath, output_image_path)
        output_json_path = os.path.join(app.config['OUTPUT_FOLDER'], f"result_{timestamp}.json")
        measurement_system.save_results(result, output_json_path)
        result['annotated_image_path'] = output_image_path
        result['json_result_path'] = output_json_path
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/set-reference', methods=['POST'])
def set_reference_dimension():
    try:
        data = request.get_json()
        reference_dim = float(data.get('reference_dimension', 30.0))
        measurement_system.reference_dimension_cm = reference_dim
        return jsonify({'status': 'success','reference_dimension_cm': reference_dim}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Cattle Measurement System - Backend")
    print("=" * 50)
    print("\nStarting Flask API server...")
    print("API Endpoints:")
    print("  - GET  /api/health")
    print("  - POST /api/measure")
    print("  - POST /api/set-reference")
    print("  - GET  /api/download/<filename>")
    app.run(debug=True, host='0.0.0.0', port=5000)
