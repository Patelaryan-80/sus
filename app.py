import os
import cv2
import numpy as np
import mediapipe as mp
import time
import datetime as dt
from datetime import datetime
import logging
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, render_template, redirect, url_for, session, send_from_directory
from flask_socketio import SocketIO
import threading
import json
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import bcrypt

# Initialize Flask application and SocketIO
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'survion_secure_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///survion.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
detection_thread = None
is_detecting = False
alert_history = []

# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))
    
    
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    camera_id = db.Column(db.String(50), nullable=False)
    
    def __repr__(self):
        return f'<Alert {self.timestamp} - {self.alert_type}>'
    
    def __init__(self, alert_type, message, camera_id):
        self.alert_type = alert_type
        self.message = message
        self.camera_id = camera_id
        self.timestamp = datetime.now(dt.UTC)
    
    

# Create the database and tables
with app.app_context():
    db.create_all()

# Configuration parameters
CONFIG = {
    "confidence_threshold": 0.4,      # Lowered MediaPipe pose detection confidence for better detection
    "yolo_confidence": 0.35,          # Lowered YOLO detection confidence for more detections
    "alert_cooldown": 5,              # Reduced cooldown between alerts
    "suspicious_pose_threshold": 0.6,  # Lowered threshold for classifying pose as suspicious
    "frame_processing_interval": 0.05, # Process frames more frequently
    "camera_index": 0,                # Default to built-in webcam (changed from 1)
    "video_source": None,             # Path to video file, None for live camera
    "suspicious_objects": ["knife", "gun", "scissors", "bottle", "cell phone"],
    "theft_related_objects": ["laptop", "cell phone", "handbag", "backpack", "wallet", "book", "remote"],
    "resolution": (1280, 720),         # Camera resolution
    "movement_threshold": 20,         # Lowered movement threshold for better detection
    "pose_confidence_required": 0.3   # Minimum confidence for pose detection
}

# Load YOLO model
try:
    yolo_model = YOLO("yolov8n.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    yolo_model = None

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=CONFIG["confidence_threshold"],
    min_tracking_confidence=CONFIG["confidence_threshold"]
)

# Detection functions
def detect_rapid_movement(landmarks_history, threshold=None):
    """Detect rapid movements in pose landmarks."""
    if len(landmarks_history) < 2:
        return False, 0
    
    if threshold is None:
        threshold = CONFIG["movement_threshold"]
    
    prev_landmarks = landmarks_history[-2]
    curr_landmarks = landmarks_history[-1]
    
    body_indices = [
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE
    ]
    
    total_movement = 0
    valid_points = 0
    
    for idx in body_indices:
        if (prev_landmarks[idx].visibility > CONFIG["pose_confidence_required"] and 
            curr_landmarks[idx].visibility > CONFIG["pose_confidence_required"]):
            
            movement = np.sqrt(
                (curr_landmarks[idx].x - prev_landmarks[idx].x) ** 2 +
                (curr_landmarks[idx].y - prev_landmarks[idx].y) ** 2
            )
            if idx in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                      mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]:
                movement *= 1.5
            
            total_movement += movement
            valid_points += 1
    
    if valid_points == 0:
        return False, 0
    
    avg_movement = total_movement / valid_points
    return avg_movement > threshold, avg_movement

def detect_unusual_pose(landmarks):
    """Enhanced unusual pose detection."""
    if not landmarks:
        return False, "No pose detected"
    
    suspicious_reasons = []
    
    if detect_crouching(landmarks):
        suspicious_reasons.append("Suspicious crouching detected")
    
    if detect_arms_raised(landmarks):
        suspicious_reasons.append("Suspicious arm position detected")
    
    if detect_bending(landmarks):
        suspicious_reasons.append("Suspicious bending detected")
    
    if detect_unusual_head_position(landmarks):
        suspicious_reasons.append("Unusual head position detected")
    
    return len(suspicious_reasons) > 0, "; ".join(suspicious_reasons)

def detect_crouching(landmarks):
    """Detect if person is in crouching position."""
    if (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility > CONFIG["pose_confidence_required"]):
        
        left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        
        if (left_knee_y > left_hip_y + 0.15 and right_knee_y > right_hip_y + 0.15):
            return True
    
    return False

def detect_arms_raised(landmarks):
    """Detect if arms are raised above shoulders."""
    if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > CONFIG["pose_confidence_required"]):
        
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        
        if (left_wrist_y < left_shoulder_y - 0.1 or right_wrist_y < right_shoulder_y - 0.1):
            return True
    
    return False

def detect_bending(landmarks):
    """Detect if person is bending suspiciously."""
    if (landmarks[mp_pose.PoseLandmark.NOSE].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility > CONFIG["pose_confidence_required"]):
        
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        
        if (nose_y > left_hip_y + 0.1 and nose_y > right_hip_y + 0.1):
            return True
    
    return False

def detect_unusual_head_position(landmarks):
    """Detect if head position is unusual (looking around suspiciously)."""
    if (landmarks[mp_pose.PoseLandmark.NOSE].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.LEFT_EAR].visibility > CONFIG["pose_confidence_required"] and
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR].visibility > CONFIG["pose_confidence_required"]):
        
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        
        ear_diff = abs(left_ear.x - right_ear.x)
        
        if ear_diff < 0.05:
            return True
        
        if nose.y < left_ear.y - 0.1 or nose.y < right_ear.y - 0.1:
            return True
    
    return False

def analyze_person_object_interaction(person_bbox, objects):
    """Enhanced analysis of person-object interactions."""
    if not objects:
        return False, ""
    
    suspicious_interactions = []
    
    p_xmin, p_ymin, p_width, p_height = person_bbox
    p_xmax = p_xmin + p_width
    p_ymax = p_ymin + p_height
    
    person_area = p_width * p_height
    
    for obj in objects:
        obj_name = obj['name']
        obj_conf = obj['confidence']
        obj_bbox = obj['bbox']
        
        o_xmin, o_ymin, o_width, o_height = obj_bbox
        o_xmax = o_xmin + o_width
        o_ymax = o_ymin + o_height
        
        inter_xmin = max(p_xmin, o_xmin)
        inter_ymin = max(p_ymin, o_ymin)
        inter_xmax = min(p_xmax, o_xmax)
        inter_ymax = min(p_ymax, o_ymax)
        
        close_proximity = False
        if not (inter_xmax > inter_xmin and inter_ymax > inter_ymin):
            p_center_x = p_xmin + p_width/2
            p_center_y = p_ymin + p_height/2
            o_center_x = o_xmin + o_width/2
            o_center_y = o_ymin + o_height/2
            
            distance = np.sqrt((p_center_x - o_center_x)**2 + (p_center_y - o_center_y)**2)
            proximity_threshold = np.sqrt(person_area) * 0.5
            
            if distance < proximity_threshold:
                close_proximity = True
        
        if (inter_xmax > inter_xmin and inter_ymax > inter_ymin) or close_proximity:
            if obj_name in CONFIG["suspicious_objects"]:
                suspicious_interactions.append(f"Person interacting with suspicious object: {obj_name} (conf: {obj_conf:.2f})")
            elif obj_name in CONFIG["theft_related_objects"]:
                suspicious_interactions.append(f"Potential theft: Person interacting with {obj_name} (conf: {obj_conf:.2f})")
            
            if len(suspicious_interactions) > 1:
                suspicious_interactions.append("Multiple suspicious interactions detected")
    
    return len(suspicious_interactions) > 0, "; ".join(suspicious_interactions)

def get_person_bbox_from_landmarks(landmarks, frame_shape):
    """Get bounding box of person from pose landmarks."""
    if not landmarks:
        return None
    
    x_coords = []
    y_coords = []
    
    for landmark in landmarks:
        if landmark.visibility > CONFIG["pose_confidence_required"]:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
    
    if not x_coords or not y_coords:
        return None
    
    frame_height, frame_width = frame_shape[:2]
    
    xmin = max(0, int(min(x_coords) * frame_width))
    ymin = max(0, int(min(y_coords) * frame_height))
    xmax = min(frame_width, int(max(x_coords) * frame_width))
    ymax = min(frame_height, int(max(y_coords) * frame_height))
    
    width = xmax - xmin
    height = ymax - ymin
    
    return [xmin, ymin, width, height]

def setup_camera():
    """Set up video capture from camera or file."""
    global camera
    
    try:
        if camera is not None:
            camera.release()
        
        if CONFIG["video_source"] and os.path.exists(CONFIG["video_source"]):
            camera = cv2.VideoCapture(CONFIG["video_source"])
        else:
            # Try the configured camera index first
            camera = cv2.VideoCapture(CONFIG["camera_index"])
            
            # If that fails, try alternative camera indices
            if not camera.isOpened():
                logger.info(f"Camera index {CONFIG['camera_index']} failed, trying alternative indices")
                alternative_indices = [0, 1]  # Try both common webcam indices
                
                for idx in alternative_indices:
                    if idx == CONFIG["camera_index"]:
                        continue  # Skip the one we already tried
                        
                    logger.info(f"Trying camera index {idx}")
                    camera = cv2.VideoCapture(idx)
                    if camera.isOpened():
                        CONFIG["camera_index"] = idx  # Update the config with the working index
                        logger.info(f"Successfully connected to camera at index {idx}")
                        break
        
        # Try to set resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["resolution"][0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["resolution"][1])
        
        # Verify camera opened successfully
        if not camera.isOpened():
            logger.error("Error: Could not open video source")
            return False
            
        # Try reading a test frame
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.error("Error: Could not read frame from video source")
            camera.release()
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Camera setup error: {e}")
        if camera is not None:
            camera.release()
        return False

def generate_frames():
    global output_frame, camera, lock, is_detecting
    
    # Create a fallback frame (black image with text)
    fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(fallback_frame, "No Camera Feed Available", (100, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while True:
        try:
            # Read frame directly from camera if output_frame is not available
            with lock:
                if output_frame is not None:
                    frame_to_send = output_frame.copy()
                else:
                    if not is_detecting or camera is None or not camera.isOpened():
                        frame_to_send = fallback_frame.copy()
                    else:
                        ret, frame = camera.read()
                        if not ret:
                            frame_to_send = fallback_frame.copy()
                        else:
                            frame_to_send = frame
            
            # Encode and send the frame
            flag, encoded_image = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not flag:
                continue
            
            frame_data = b'--frame\r\n' \
                        b'Content-Type: image/jpeg\r\n\r\n' + \
                        bytearray(encoded_image) + \
                        b'\r\n'
            yield frame_data
            
            # Add a short delay to reduce CPU usage
            time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
            
            # Send fallback frame on error
            try:
                flag, encoded_image = cv2.imencode('.jpg', fallback_frame)
                if flag:
                    frame_data = b'--frame\r\n' \
                                b'Content-Type: image/jpeg\r\n\r\n' + \
                                bytearray(encoded_image) + \
                                b'\r\n'
                    yield frame_data
            except:
                pass
                
            # Add delay before retrying
            time.sleep(0.5)
            continue

def detect_suspicious_activities():
    """Main detection function that runs in a separate thread."""
    global camera, output_frame, lock, is_detecting, alert_history  # Added alert_history to globals
    
    landmarks_history = []
    last_alert_time = 0
    last_process_time = time.time()
    suspicious_activity_counter = 0
    
    is_detecting = True
    
    try:
        while is_detecting and camera is not None:  # Added camera check
            current_time = time.time()
            
            if current_time - last_process_time >= CONFIG["frame_processing_interval"]:
                last_process_time = current_time
                
                ret, frame = camera.read()
                if not ret or frame is None:  # Added frame check
                    logger.error("Failed to grab frame")
                    continue  # Changed from break to continue to make it more resilient
                
                try:
                    output_img = frame.copy()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(frame_rgb)
                    
                    detected_objects = []
                    if yolo_model is not None:
                        try:
                            yolo_results = yolo_model(frame, conf=CONFIG["yolo_confidence"])
                            
                            for result in yolo_results:
                                for i, (box, score, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                                    x1, y1, x2, y2 = box.tolist()
                                    cls_id = int(cls.item())
                                    conf = score.item()
                                    label = result.names[cls_id]
                                    
                                    x, y = int(x1), int(y1)
                                    w, h = int(x2 - x1), int(y2 - y1)
                                    
                                    detected_objects.append({
                                        'name': label,
                                        'confidence': conf,
                                        'bbox': [x, y, w, h],
                                        'class_id': cls_id
                                    })

                                    color = (0, 0, 255) if label in CONFIG["suspicious_objects"] else (0, 255, 0)
                                    theft_color = (255, 0, 255) if label in CONFIG["theft_related_objects"] else color
                                    cv2.rectangle(output_img, (x, y), (x+w, y+h), theft_color, 2)
                                    cv2.putText(output_img, f"{label} {conf:.2f}", (x, y-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, theft_color, 2)
                                    
                                    socketio.emit('detected_object', {
                                        'label': label,
                                        'confidence': conf,
                                        'is_suspicious': label in CONFIG["suspicious_objects"],
                                        'is_theft_related': label in CONFIG["theft_related_objects"]
                                    })
                        except Exception as e:
                            logger.error(f"YOLO processing error: {e}")
                    
                    is_suspicious = False
                    suspicious_reasons = []
                    
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            output_img, 
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )
                        
                        landmarks_history.append(pose_results.pose_landmarks.landmark)
                        if len(landmarks_history) > 10:
                            landmarks_history.pop(0)
                        
                        person_bbox = get_person_bbox_from_landmarks(
                            pose_results.pose_landmarks.landmark, 
                            frame.shape
                        )
                        
                        rapid_movement, movement_score = detect_rapid_movement(landmarks_history)
                        if rapid_movement:
                            is_suspicious = True
                            suspicious_reasons.append(f"Rapid movement detected (score: {movement_score:.2f})")
                        
                        unusual_pose, pose_reason = detect_unusual_pose(pose_results.pose_landmarks.landmark)
                        if unusual_pose:
                            is_suspicious = True
                            suspicious_reasons.append(pose_reason)
                        
                        if person_bbox and detected_objects:
                            interaction, interaction_reason = analyze_person_object_interaction(
                                person_bbox, 
                                detected_objects
                            )
                            if interaction:
                                is_suspicious = True
                                suspicious_reasons.append(interaction_reason)
                        
                        if is_suspicious:
                            suspicious_activity_counter += 1
                        else:
                            suspicious_activity_counter = max(0, suspicious_activity_counter - 1)
                        
                        if (suspicious_activity_counter >= 3 or len(suspicious_reasons) >= 2) and \
                           (current_time - last_alert_time > CONFIG["alert_cooldown"]):
                            confidence_level = min(100, (suspicious_activity_counter * 20))
                            alert_message = "Suspicious activity detected: " + "; ".join(suspicious_reasons)
                            
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            alert = {
                                "timestamp": timestamp,
                                "message": alert_message,
                                "confidence": confidence_level
                            }
                            
                            with lock:  # Added lock for thread safety
                                alert_history.append(alert)
                                if len(alert_history) > 100:
                                    alert_history = alert_history[-100:]
                            
                            # Save alert to database
                            try:
                                with app.app_context():
                                    alert_type = "Suspicious Activity"
                                    camera_id = f"Camera_{CONFIG['camera_index']}"
                                    db_alert = Alert(alert_type=alert_type, message=alert_message, camera_id=camera_id)
                                    db.session.add(db_alert)
                                    db.session.commit()
                                    logger.info(f"Alert saved to database: {alert_message}")
                            except Exception as e:
                                logger.error(f"Failed to save alert to database: {e}")
                            
                            socketio.emit('new_alert', alert)
                            last_alert_time = current_time
                            
                            cv2.putText(output_img, f"ALERT ({confidence_level}% confidence): {alert_message}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (0, 0, 255), 2)
                            
                            logger.info(f"ALERT: {alert_message}")
                            
                            # Automatically capture screenshot when suspicious activity detected
                            try:
                                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                screenshot_dir = os.path.join(app.static_folder, 'screenshots')
                                
                                if not os.path.exists(screenshot_dir):
                                    os.makedirs(screenshot_dir)
                                    
                                screenshot_path = os.path.join(screenshot_dir, f'alert_screenshot_{timestamp_str}.jpg')
                                
                                with lock:
                                    screenshot = output_img.copy()
                                    success = cv2.imwrite(screenshot_path, screenshot)
                                    
                                    if success:
                                        logger.info(f"Alert screenshot captured automatically: {screenshot_path}")
                                    else:
                                        logger.error("Failed to save automatic alert screenshot")
                                        
                            except Exception as e:
                                logger.error(f"Error capturing alert screenshot: {e}")
                    
                    with lock:
                        output_frame = output_img.copy()
                        
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    continue  # Skip this frame but continue processing
                
    except Exception as e:
        logger.error(f"Detection error: {e}")
    finally:
        is_detecting = False
        if camera is not None:
            camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/servilance')
def servilance():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('servilance.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global detection_thread, is_detecting
    
    if is_detecting:
        return jsonify({"status": "error", "message": "Detection already running"})
    
    if request.json:
        for key, value in request.json.items():
            if key in CONFIG:
                CONFIG[key] = value
    
    if not setup_camera():
        return jsonify({"status": "error", "message": "Failed to setup camera"})
    
    detection_thread = threading.Thread(target=detect_suspicious_activities)
    detection_thread.daemon = True
    detection_thread.start()
    
    return jsonify({"status": "success", "message": "Detection started", "config": CONFIG})

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    global is_detecting, camera, output_frame
    
    is_detecting = False
    
    # Clean up resources
    if camera is not None:
        camera.release()
        camera = None
    
    # Clear the output frame
    with lock:
        output_frame = None
    
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/get_alerts')
def get_alerts_route():
    # Get the most recent alerts from the database
    try:
        db_alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(100).all()
        alerts_data = []
        
        for alert in db_alerts:
            alerts_data.append({
                "timestamp": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "message": alert.message,
                "confidence": 85  # Default confidence value
            })
        
        # Add in-memory alerts too (for backward compatibility)
        for alert in alert_history:
            if "timestamp" in alert and "message" in alert:
                # Check if this alert is already in the database results
                if not any(db_alert["message"] == alert["message"] and 
                        db_alert["timestamp"] == alert["timestamp"] for db_alert in alerts_data):
                    alerts_data.append(alert)
        
        # Sort combined alerts by timestamp (newest first)
        alerts_data.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"), reverse=True)
        
        return jsonify({"alerts": alerts_data})
    except Exception as e:
        logger.error(f"Error retrieving alerts from database: {e}")
        # Fall back to in-memory alerts if database fails
        return jsonify({"alerts": alert_history})

@app.route('/clear_alerts', methods=['POST'])
def clear_alerts_route():
    global alert_history
    alert_history = []
    
    # Also clear alerts from the database
    try:
        with app.app_context():
            Alert.query.delete()
            db.session.commit()
            logger.info("All alerts cleared from database")
    except Exception as e:
        logger.error(f"Failed to clear alerts from database: {e}")
    
    return jsonify({"status": "success", "message": "Alerts cleared"})

@app.route('/get_config')
def get_config_route():
    return jsonify({"status": "success", "config": CONFIG})

@app.route('/update_config', methods=['POST'])
def update_config_route():
    if not request.json:
        return jsonify({"status": "error", "message": "No JSON data provided"})
    
    for key, value in request.json.items():
        if key in CONFIG:
            CONFIG[key] = value
    
    return jsonify({"status": "success", "message": "Configuration updated", "config": CONFIG})

@app.route('/get_latest_alert')
def get_latest_alert_route():
    if not alert_history:
        return jsonify({"alert_active": False})
    
    latest_alert = alert_history[-1]
    current_time = time.time()
    alert_time = datetime.strptime(latest_alert["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
    
    # Only show alert if it's within the last 5 seconds
    if current_time - alert_time <= 5:
        return jsonify({
            "alert_active": True,
            "alert_message": latest_alert["message"]
        })
    
    return jsonify({"alert_active": False})

@app.route('/get_status')
def get_status_route():
    global camera, is_detecting
    
    camera_status = False
    if camera is not None:
        camera_status = camera.isOpened()
    
    return jsonify({
        "status": "success",
        "system_status": {
            "is_detecting": is_detecting,
            "camera_status": camera_status
        }
    })

@app.route('/capture_screenshot', methods=['POST'])
def capture_screenshot():
    global output_frame, camera, is_detecting
    
    # Check if detection is active
    if not is_detecting:
        return jsonify({
            "status": "error", 
            "message": "Detection is not active. Start detection first to capture a screenshot."
        }), 400
    
    # Check if camera is available
    if camera is None or not camera.isOpened():
        return jsonify({
            "status": "error", 
            "message": "Camera is not available. Check your camera connection."
        }), 400
    
    # Check if we have a frame to capture
    if output_frame is None:
        return jsonify({
            "status": "error", 
            "message": "No frame available to capture. Wait for video to initialize."
        }), 400
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = os.path.join(app.static_folder, 'screenshots')
        
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
            
        screenshot_path = os.path.join(screenshot_dir, f'screenshot_{timestamp}.jpg')
        
        with lock:
            screenshot = output_frame.copy()
            success = cv2.imwrite(screenshot_path, screenshot)
            
            if not success:
                return jsonify({
                    "status": "error", 
                    "message": "Failed to save screenshot file. Check permissions and disk space."
                }), 500
        
        # Convert image to base64 for web display
        ret, buffer = cv2.imencode('.jpg', screenshot)
        if not ret:
            return jsonify({
                "status": "error", 
                "message": "Failed to encode screenshot image."
            }), 500
            
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Use the relative path for the response
        screenshot_filename = f'screenshot_{timestamp}.jpg'
        relative_path = f'screenshots/{screenshot_filename}'
        
        logger.info(f"Screenshot captured successfully: {screenshot_path}")
        return jsonify({
            "status": "success", 
            "message": "Screenshot captured successfully", 
            "filename": screenshot_filename,
            "path": relative_path,
            "image_data": jpg_as_text
        })
        
    except Exception as e:
        logger.error(f"Screenshot error: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Screenshot error: {str(e)}"
        }), 500

@app.route('/screenshots')
def screenshots():
    """Display a list of all alert screenshots."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    screenshot_dir = os.path.join(app.static_folder, 'screenshots')
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    
    # Get all screenshots with "alert_screenshot" in filename (auto-captured ones)
    screenshots = []
    for filename in os.listdir(screenshot_dir):
        if 'alert_screenshot' in filename and (filename.endswith('.jpg') or filename.endswith('.png')):
            file_path = os.path.join(screenshot_dir, filename)
            file_stat = os.stat(file_path)
            
            # Get creation date
            creation_time = datetime.fromtimestamp(file_stat.st_ctime)
            
            # Try to extract timestamp from filename
            timestamp = None
            try:
                date_part = filename.replace('alert_screenshot_', '').replace('.jpg', '').replace('.png', '')
                timestamp = datetime.strptime(date_part, '%Y%m%d_%H%M%S')
            except ValueError:
                timestamp = creation_time
            
            screenshots.append({
                'filename': filename,
                'path': f'screenshots/{filename}',
                'timestamp': timestamp,
                'size': file_stat.st_size
            })
    
    # Sort by timestamp (newest first)
    screenshots.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Get alerts from database to match with screenshots
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    
    return render_template('screenshots.html', screenshots=screenshots, alerts=alerts)

@app.route('/delete_screenshot/<filename>', methods=['POST'])
def delete_screenshot(filename):
    """Delete a screenshot file."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    screenshot_dir = os.path.join(app.static_folder, 'screenshots')
    file_path = os.path.join(screenshot_dir, filename)
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Screenshot deleted: {filename}")
            return jsonify({"status": "success", "message": f"Screenshot {filename} deleted"})
        else:
            return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting screenshot: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple validation
        if not all([username, password]):
            error = 'Username and password are required'
        else:
            user = User.query.filter_by(username=username).first() or User.query.filter_by(email=username).first()
            if user and user.check_password(password):
                session['user_id'] = user.username
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid username or password'
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        # Simple validation
        if not all([username, email, password]):
            error = 'All fields are required'
        elif password != confirm_password:
            error = 'Passwords do not match'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters long'
        elif not any(char.isdigit() for char in password):
            error = 'Password must contain at least one digit'
        elif not any(char.isalpha() for char in password):
            error = 'Password must contain at least one letter'
        elif not any(char in '!@#$%^&*()_+' for char in password):
            error = 'Password must contain at least one special character'
        elif not any(char.isupper() for char in password):
            error = 'Password must contain at least one uppercase letter'
        else:
            existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                error = 'Username or email already exists'
            else:
                new_user = User(username, email, password)
                db.session.add(new_user)
                db.session.commit()
                return redirect(url_for('login'))
            
    return render_template('register.html', error=error)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
