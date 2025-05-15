import os
import cv2
import numpy as np
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
import mediapipe as mp
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import bcrypt
import torch
from collections import deque # For efficient history
import pickle
from tensorflow.keras.models import load_model
import tempfile
import uuid
from werkzeug.utils import secure_filename

# Custom Layer for Keras Model
class ModifiedDepthwiseConv2D:
    def __init__(self, *args, groups=None, **kwargs):
        pass

# Initialize Flask application and SocketIO
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'survion_secure_key_change_this_in_prod' # IMPORTANT: Change this to a strong, unique key in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///survion.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
db = SQLAlchemy(app)
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
camera = None # cv2.VideoCapture object
output_frame = None # Frame to be displayed in the web feed
lock = threading.Lock() # Lock for accessing shared resources
detection_thread = None
is_detecting = False # Flag to control the detection loop
alert_history = deque(maxlen=100) # Use deque for efficient history of recent alerts in memory

# Detection state variables
last_alert_time = 0
suspicious_activity_counter = 0
suspicion_score = 0.0

# Object tracking history
object_tracking_history = {}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a subfolder for processed videos
processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
os.makedirs(processed_folder, exist_ok=True)

# For activity recognition model
keras_model = None
lb = None
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
activity_Q = deque(maxlen=10)  # Default size 10 for activity prediction queue


# User model for authentication (Keep as is)
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

# Alert model for logging (Keep as is)
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(255), nullable=False) # Increased size for longer messages
    camera_id = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<Alert {self.timestamp} - {self.alert_type}>'

    def __init__(self, alert_type, message, camera_id):
        self.alert_type = alert_type
        self.message = message[:255] # Ensure message fits DB column
        self.camera_id = camera_id
        # timestamp defaults in column definition are handled by SQLAlchemy


# Create the database and tables
with app.app_context():
    db.create_all()
    # Create a default admin user if none exists (Optional, for easy first-time setup)
    # try:
    #     if not User.query.filter_by(username='admin').first():
    #         admin_user = User('admin', 'admin@example.com', 'admin_password') # CHANGE THIS PASSWORD!
    #         db.session.add(admin_user)
    #         db.session.commit()
    #         logger.info("Default 'admin' user created. CHANGE THE PASSWORD!")
    # except Exception as e:
    #     logger.error(f"Error creating default user: {e}", exc_info=True)
    #     db.session.rollback()


# Configuration parameters
CONFIG = {
    "confidence_threshold": 0.4,      # YOLO confidence threshold
    "alert_cooldown": 3,              # Cooldown between alerts (seconds)
    "frame_processing_interval": 0.03, # Target frame processing interval (seconds)
    "frame_skip": 5,                  # Process every N frames. 1 means process every frame.
    "camera_index": 1,                # Default to built-in webcam index 0
    "video_source": None,             # Path to video file, None for live camera
    "suspicious_objects": ["knife", "gun", "scissors"], # Focused on weapons/tools
    "theft_related_objects": ["laptop", "cell phone", "handbag", "backpack", "wallet", "purse", "briefcase", "jewelry", "book"], # Objects to track
    "resolution": (640, 480),         # Processing resolution
    "enable_gpu": True,               # Enable GPU acceleration if available
    "alert_threshold": 3.0,           # Threshold for total suspicion_activity_counter for regular alerts
    "activity_prediction_size": 10,   # Size of queue for activity prediction averaging
    "keras_model_path": "saved_model.keras", # Path to keras model
    "label_bin_path": "lb.pickle",    # Path to label binarizer
    "activity_threshold": 0.80        # Threshold for activity detection confidence
}

# Load YOLO model
yolo_model = None
try:
    if CONFIG["enable_gpu"] and torch.cuda.is_available():
        logger.info("Attempting to load YOLOv8n on GPU...")
        yolo_model = YOLO("yolov8n.pt")
        yolo_model.to('cuda')
        logger.info("YOLOv8n loaded successfully on GPU.")
    else:
        logger.info("Loading YOLOv8n on CPU...")
        yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n loaded successfully on CPU.")

except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}. Object detection will be limited.", exc_info=True)
    yolo_model = None

# Load Keras Activity Recognition Model
try:
    logger.info("Loading Keras activity recognition model...")
    if os.path.exists(CONFIG["keras_model_path"]):
        # Use tf.keras.utils.custom_object_scope if needed
        keras_model = load_model(CONFIG["keras_model_path"], compile=False)
        logger.info(f"Keras model loaded successfully from {CONFIG['keras_model_path']}")
    else:
        logger.error(f"Keras model file not found at {CONFIG['keras_model_path']}")
    
    # Load label binarizer
    if os.path.exists(CONFIG["label_bin_path"]):
        lb = pickle.loads(open(CONFIG["label_bin_path"], "rb").read())
        logger.info(f"Label binarizer loaded successfully from {CONFIG['label_bin_path']}")
        logger.info(f"Activity labels: {lb.classes_}")
    else:
        logger.error(f"Label binarizer file not found at {CONFIG['label_bin_path']}")
except Exception as e:
    logger.error(f"Failed to load Keras model: {e}", exc_info=True)
    keras_model = None
    lb = None

# Initialize MediaPipe Pose
pose = None
mp_pose = None
mp_drawing = None
try:
    logger.info("Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # Use 1 for balance, 0 for speed, 2 for accuracy
        smooth_landmarks=True,
        enable_segmentation=False, # Disable segmentation for speed
        smooth_segmentation=False, # Disable smoothing for speed
        min_detection_confidence=CONFIG["confidence_threshold"],
        min_tracking_confidence=CONFIG["confidence_threshold"]
    )
    logger.info("MediaPipe Pose initialized.")
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe Pose: {e}. Pose detection will not be available.", exc_info=True)
    pose = None
    mp_pose = None
    mp_drawing = None


# Use deque for pose history
landmarks_history = deque(maxlen=CONFIG["detection_history_size"])


# --- Camera Setup Function (Integrated) ---
def setup_camera(index=CONFIG['camera_index'], source=CONFIG['video_source'], resolution=CONFIG['resolution']):
    """Set up video capture from camera index or file."""
    global camera

    if camera is not None:
        logger.info("Releasing existing camera instance.")
        try:
             camera.release()
        except Exception as e:
             logger.error(f"Error releasing camera: {e}", exc_info=True)
        camera = None # Ensure camera is None after releasing

    try:
        if source and os.path.exists(source):
            logger.info(f"Attempting to open video file: {source}")
            camera = cv2.VideoCapture(source)
        else:
            # Try the configured camera index first
            logger.info(f"Attempting to open camera index: {index}")
            camera = cv2.VideoCapture(index)

            # If that fails, try alternative common indices
            if not camera.isOpened():
                logger.warning(f"Camera index {index} failed, trying alternative indices (0, 1)...")
                alternative_indices = [0, 1]
                for idx in alternative_indices:
                    if idx == index:
                        continue
                    logger.info(f"Trying camera index {idx}")
                    camera = cv2.VideoCapture(idx)
                    if camera.isOpened():
                        CONFIG["camera_index"] = idx # Update config if successful
                        logger.info(f"Successfully connected to camera at index {idx}")
                        break

        # Verify camera opened successfully
        if not camera or not camera.isOpened():
            logger.error("Error: Could not open video source after trying options.")
            return False

        # Try to set resolution
        try:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Requested resolution: {resolution}, Actual resolution: ({int(actual_width)}, {int(actual_height)})")
            # Update config with actual resolution if significantly different? Or log warning?
            # CONFIG["resolution"] = (int(actual_width), int(actual_height)) # Optional: update config with actual resolution

        except Exception as e:
             logger.warning(f"Could not set resolution {resolution}: {e}", exc_info=True)


        # Try reading a test frame
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.error("Error: Could not read initial frame from video source.")
            camera.release()
            camera = None
            return False

        logger.info("Camera setup successful.")
        return True

    except Exception as e:
        logger.error(f"Camera setup error: {e}", exc_info=True)
        if camera is not None:
            camera.release()
        camera = None
        return False


# --- Helper Functions (from previous response, keep as is) ---

# Adjusted movement detection to return normalized score
def detect_rapid_movement(landmarks_history, frame_diag_norm):
    """Detect rapid movements in pose landmarks with smoothing, return normalized score."""
    if not mp_pose or len(landmarks_history) < 2:
        return False, 0.0

    prev_landmarks = landmarks_history[-2]
    curr_landmarks = landmarks_history[-1]

    body_indices = [ # Focus on upper body for potential theft indicators
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_HIP, # Include hips for overall body movement
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    total_movement_norm = 0
    valid_points = 0

    for idx in body_indices:
        # Check if landmark index exists and is valid
        if idx.value >= len(prev_landmarks.landmark) or idx.value >= len(curr_landmarks.landmark):
             continue

        prev_lm = prev_landmarks.landmark[idx]
        curr_lm = curr_landmarks.landmark[idx]

        if (prev_lm.visibility > CONFIG["pose_confidence_required"] and
            curr_lm.visibility > CONFIG["pose_confidence_required"]):

            # Movement is calculated from normalized coordinates (0 to 1)
            movement_norm = np.sqrt(
                (curr_lm.x - prev_lm.x) ** 2 +
                (curr_lm.y - prev_lm.y) ** 2
            )

            total_movement_norm += movement_norm
            valid_points += 1

    if valid_points == 0:
        return False, 0.0

    avg_movement_norm = total_movement_norm / valid_points

    # Threshold is now normalized
    is_rapid = avg_movement_norm > CONFIG["movement_threshold"]

    return is_rapid, avg_movement_norm

# Keep unusual pose detection, maybe refine thresholds/reasons slightly
def detect_unusual_pose(landmarks):
    """Enhanced unusual pose detection."""
    if not landmarks or not mp_pose:
        return False, ""

    suspicious_reasons = set() # Use a set to avoid duplicate reasons

    # Require minimum number of visible key points for pose analysis
    # Check if landmark list is not empty before accessing elements
    if not landmarks.landmark:
        return False, "No pose landmarks detected"

    visible_points_count = sum(1 for lm in landmarks.landmark if lm.visibility > CONFIG["pose_confidence_required"])
    if visible_points_count < 10: # Require at least 10 visible points (head, shoulders, arms, hips)
         return False, "Insufficient visible pose points"

    # Add checks for required landmarks before accessing
    required_lms = [
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR
    ]
    if not all(idx.value < len(landmarks.landmark) for idx in required_lms):
         return False, "Missing required pose landmarks"


    if detect_crouching(landmarks):
        suspicious_reasons.add("Suspicious crouching detected")

    if detect_arms_raised(landmarks):
        suspicious_reasons.add("Suspicious arm position detected")

    if detect_bending(landmarks):
        suspicious_reasons.add("Suspicious bending detected")

    if detect_unusual_head_position(landmarks):
         suspicious_reasons.add("Unusual head position detected")

    if check_hands_near_body_center(landmarks):
         suspicious_reasons.add("Hands near body center (potentially hiding object)")

    return len(suspicious_reasons) > 0, "; ".join(suspicious_reasons)

# Helper functions for specific poses (ensure they check landmark visibility)
def detect_crouching(landmarks):
    if not mp_pose or not landmarks or len(landmarks.landmark) <= max(mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value): return False
    l_knee, r_knee, l_hip, r_hip = [landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]]
    if (l_knee.visibility > CONFIG["pose_confidence_required"] and r_knee.visibility > CONFIG["pose_confidence_required"] and
        l_hip.visibility > CONFIG["pose_confidence_required"] and r_hip.visibility > CONFIG["pose_confidence_required"]):
        # Knee y-coordinate is below hip y-coordinate (higher in image means lower physically)
        if (l_knee.y > l_hip.y + 0.15 and r_knee.y > r_hip.y + 0.15):
            return True
    return False

def detect_arms_raised(landmarks):
    if not mp_pose or not landmarks or len(landmarks.landmark) <= max(mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value): return False
    l_wrist, r_wrist, l_shoulder, r_shoulder = [landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]]
    if (l_wrist.visibility > CONFIG["pose_confidence_required"] and r_wrist.visibility > CONFIG["pose_confidence_required"] and
        l_shoulder.visibility > CONFIG["pose_confidence_required"] and r_shoulder.visibility > CONFIG["pose_confidence_required"]):
        # Wrist y-coordinate is above shoulder y-coordinate (lower in image means higher physically)
        if (l_wrist.y < l_shoulder.y - 0.15 or r_wrist.y < r_shoulder.y - 0.15):
            return True
    return False

def detect_bending(landmarks):
    if not mp_pose or not landmarks or len(landmarks.landmark) <= max(mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value): return False
    nose, l_hip, r_hip = [landmarks.landmark[i] for i in [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]]
    if (nose.visibility > CONFIG["pose_confidence_required"] and
        l_hip.visibility > CONFIG["pose_confidence_required"] and r_hip.visibility > CONFIG["pose_confidence_required"]):
        # Nose y-coordinate is below hip y-coordinate significantly
        if (nose.y > l_hip.y + 0.15 or nose.y > r_hip.y + 0.15):
            return True
    return False

def detect_unusual_head_position(landmarks):
    """Detect if head is tilted or twisted unusually."""
    if not mp_pose or not landmarks or len(landmarks.landmark) <= max(mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value): return False

    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if not all(lm.visibility > CONFIG["pose_confidence_required"] for lm in [nose, left_ear, right_ear, left_shoulder, right_shoulder]): return False

    # Check for significant horizontal difference between ears (head turned sideways)
    ear_diff_x_norm = abs(left_ear.x - right_ear.x)
    shoulder_diff_x_norm = abs(left_shoulder.x - right_shoulder.x)
    # If shoulders are relatively wide but ears are close, head is turned sideways
    if shoulder_diff_x_norm > 0.1 and ear_diff_x_norm < 0.03: # 0.1 shoulder diff is heuristic for facing somewhat forward
        return True

    # Check for head significantly lower than shoulders (looking down)
    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    if nose.y > shoulder_mid_y + 0.1: # Nose is significantly lower than mid-shoulder
         return True

    # Check for head tilted sideways (nose significantly off-center horizontally relative to ears)
    ear_mid_x = (left_ear.x + right_ear.x) / 2
    if abs(nose.x - ear_mid_x) > 0.05: # Nose is horizontally distant from mid-ear
         return True

    return False

def check_hands_near_body_center(landmarks):
     if not mp_pose or not landmarks or len(landmarks.landmark) <= max(mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value): return False
     l_wrist, r_wrist, l_shoulder, r_shoulder = [landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]]

     if not all(lm.visibility > CONFIG["pose_confidence_required"] for lm in [l_wrist, r_wrist, l_shoulder, r_shoulder]): return False

     body_center_x = (l_shoulder.x + r_shoulder.x) / 2
     # Check if wrists are below shoulders and close to the horizontal center of the body (normalized coords)
     if (l_wrist.y > l_shoulder.y and r_wrist.y > r_shoulder.y and
         abs(l_wrist.x - body_center_x) < 0.08 and abs(r_wrist.x - body_center_x) < 0.08): # Reduced threshold slightly
         return True

     return False


def get_person_bbox_from_landmarks(landmarks, frame_shape):
    """Get bounding box of person from pose landmarks."""
    if not landmarks:
        return None

    x_coords = []
    y_coords = []

    # Use all visible landmarks with sufficient confidence
    for landmark in landmarks.landmark: # Access .landmark
        if landmark.visibility > CONFIG["pose_confidence_required"]:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)

    if not x_coords or not y_coords:
        return None

    frame_height, frame_width = frame_shape[:2]

    # Add padding to the bounding box
    # Calculate range in normalized coordinates
    x_min_norm = min(x_coords)
    y_min_norm = min(y_coords)
    x_max_norm = max(x_coords)
    y_max_norm = max(y_coords)

    range_x_norm = x_max_norm - x_min_norm
    range_y_norm = y_max_norm - y_min_norm

    # Use a fixed relative padding or adaptive based on range
    padding_x_norm = range_x_norm * 0.15 # Increased padding
    padding_y_norm = range_y_norm * 0.15 # Increased padding


    xmin_norm_padded = max(0.0, x_min_norm - padding_x_norm)
    ymin_norm_padded = max(0.0, y_min_norm - padding_y_norm)
    xmax_norm_padded = min(1.0, x_max_norm + padding_x_norm)
    ymax_norm_padded = min(1.0, y_max_norm + padding_y_norm)


    xmin = max(0, int(xmin_norm_padded * frame_width))
    ymin = max(0, int(ymin_norm_padded * frame_height))
    xmax = min(frame_width, int(xmax_norm_padded * frame_width))
    ymax = min(frame_height, int(ymax_norm_padded * frame_height))

    width = xmax - xmin
    height = ymax - ymin

    # Return [x, y, w, h] format consistent with YOLO
    return [xmin, ymin, width, height]


# --- Enhanced Theft Analysis Function (from previous response, keep as is) ---

def analyze_potential_theft(current_frame_objects, current_landmarks, frame_shape):
    """
    Analyzes specific patterns for potential theft based on objects and pose.
    Returns (is_theft_indicator, reason_list, theft_score_contribution)
    """
    theft_indicators = []
    theft_score_contribution = 0.0
    frame_height, frame_width = frame_shape[:2]
    frame_diag_norm_factor = np.sqrt(frame_width**2 + frame_height**2) # Factor to normalize pixel distances


    # --- 1. Hand-Object Proximity Check ---
    if current_landmarks and mp_pose and current_frame_objects:
        hand_landmarks_indices = [
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_INDEX, # Include finger tips
            mp_pose.PoseLandmark.RIGHT_INDEX
        ]
        # Filter for indices that exist in the landmark list
        available_hand_lms = [
            current_landmarks.landmark[idx] for idx in hand_landmarks_indices
            if idx.value < len(current_landmarks.landmark) # Check if index is valid
        ]


        for obj in current_frame_objects:
            if obj['name'] in CONFIG["theft_related_objects"] and obj['confidence'] > CONFIG["yolo_confidence"]:
                o_xmin, o_ymin, o_width, o_height = obj['bbox']
                obj_center_x = o_xmin + o_width // 2
                obj_center_y = o_ymin + o_height // 2
                obj_radius_pix = max(o_width, o_height) // 2 # Approximate object size

                closest_dist_to_hand_norm = float('inf')
                hand_near_obj = False # Flag if *any* hand landmark is near

                for hand_lm in available_hand_lms:
                    if hand_lm.visibility > CONFIG["pose_confidence_required"]:
                        hand_x_pix = int(hand_lm.x * frame_width)
                        hand_y_pix = int(hand_lm.y * frame_height)

                        distance_pix = np.sqrt((hand_x_pix - obj_center_x)**2 + (hand_y_pix - obj_center_y)**2)
                        # Consider "near" if distance is less than object radius + a margin (in pixels)
                        proximity_margin_pix = 30 # Pixels margin (tune this)
                        if distance_pix < obj_radius_pix + proximity_margin_pix:
                             hand_near_obj = True
                             # Normalize distance for scoring
                             dist_norm = distance_pix / frame_diag_norm_factor
                             closest_dist_to_hand_norm = min(closest_dist_to_hand_norm, dist_norm)

                if hand_near_obj:
                    theft_indicators.append(f"Hand close to {obj['name']}")
                    # Score contribution based on how close and how confident YOLO was
                    # Inverse relationship: closer = higher score
                    proximity_score = max(0, 1 - (closest_dist_to_hand_norm / CONFIG["hand_object_proximity_threshold_normalized"])) # Capped score
                    theft_score_contribution += proximity_score * 20 # Max 20 for proximity alone


    # --- 2. Object Movement Tracking (Refined) ---
    # Update tracking history based on current objects
    current_obj_keys = set()
    for obj in current_frame_objects:
         if obj['name'] in CONFIG["theft_related_objects"] and obj['confidence'] > CONFIG["yolo_confidence"]:
             # Create a more stable ID based on class and *coarse* initial position/size
             # This helps track the "same" object instance across slight jitters
             obj_key = (obj['name'], int(obj['bbox'][0]/50)*50, int(obj['bbox'][1]/50)*50) # Use coarser grid for key stability
             current_obj_keys.add(obj_key)

             center_x_norm = (obj['bbox'][0] + obj['bbox'][2] // 2) / frame_width
             center_y_norm = (obj['bbox'][1] + obj['bbox'][3] // 2) / frame_height

             if obj_key not in object_tracking_history:
                 object_tracking_history[obj_key] = deque(maxlen=CONFIG["object_tracking_history"])

             object_tracking_history[obj_key].append((center_x_norm, center_y_norm))

    # Clean up tracking for objects not seen recently
    keys_to_remove = [key for key in object_tracking_history if key not in current_obj_keys]
    for key in keys_to_remove:
         # Keep for a few frames even if not seen to detect disappearance later if needed
         if len(object_tracking_history[key]) > CONFIG["object_tracking_history"] / 2:
             object_tracking_history.pop(key)


    # Check for rapid movement of tracked theft-related objects
    for obj in current_frame_objects:
        if obj['name'] in CONFIG["theft_related_objects"] and obj['confidence'] > CONFIG["yolo_confidence"]:
             obj_key = (obj['name'], int(obj['bbox'][0]/50)*50, int(obj['bbox'][1]/50)*50) # Use the same coarse key

             if obj_key in object_tracking_history and len(object_tracking_history[obj_key]) > 1:
                 # Calculate movement between the last two *recorded* positions
                 curr_center_norm_x, curr_center_norm_y = object_tracking_history[obj_key][-1]
                 prev_center_norm_x, prev_center_norm_y = object_tracking_history[obj_key][-2]

                 movement_norm = np.sqrt((curr_center_norm_x - prev_center_norm_x)**2 + (curr_center_norm_y - prev_center_norm_y)**2)

                 if movement_norm > CONFIG["theft_movement_threshold"]:
                     # Check if a person was close during the movement (using current frame pose/bbox)
                     is_person_close = False
                     if current_landmarks and mp_pose:
                         person_bbox = get_person_bbox_from_landmarks(current_landmarks, frame_shape)
                         if person_bbox:
                             p_xmin, p_ymin, p_width, p_height = person_bbox
                             # Check overlap or proximity with person bbox
                             obj_bbox = obj['bbox']
                             o_xmin, o_ymin, o_width, o_height = obj_bbox

                             inter_xmin = max(p_xmin, o_xmin)
                             inter_ymin = max(p_ymin, o_ymin)
                             inter_xmax = min(p_xmin + p_width, o_xmin + o_width)
                             inter_ymax = min(p_ymin + p_height, o_ymin + o_height)

                             overlap_area = 0
                             if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
                                 overlap_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                                 obj_area = o_width * o_height
                                 if obj_area > 0 and overlap_area / obj_area > 0.2: # Check significant overlap
                                     is_person_close = True

                             if not is_person_close: # If no significant overlap, check proximity
                                 p_center_x = p_xmin + p_width/2
                                 p_center_y = p_ymin + p_height/2
                                 obj_center_pix_x = o_xmin + o_width/2
                                 obj_center_pix_y = o_ymin + o_height/2
                                 distance_pix = np.sqrt((p_center_x - obj_center_pix_x)**2 + (p_center_y - obj_center_pix_y)**2)
                                 # Proximity threshold based on max dimension of person or object bbox
                                 proximity_thresh_pix = max(max(p_width, p_height), max(o_width, o_height)) * 0.4 # Tune proximity threshold
                                 if distance_pix < proximity_thresh_pix:
                                     is_person_close = True


                     if is_person_close:
                         theft_indicators.append(f"Rapid movement of {obj['name']} near person")
                         # Score contribution based on speed and proximity
                         movement_score_contrib = min(movement_norm / CONFIG["theft_movement_threshold"] * 20, 25) # Cap movement score
                         theft_score_contribution += movement_score_contrib


    # --- 3. Object Disappearance Check (Simple - needs person proximity) ---
    # This requires comparing `last_detected_objects` with `detected_objects`.
    # This logic is better handled in the main loop after both lists are available.
    pass # Placeholder


    # --- 4. Combining Theft Indicators with Pose (Score contribution) ---
    # Check if specific poses coincide with theft indicators
    if current_landmarks and mp_pose and theft_indicators:
         # Don't re-run full detect_unusual_pose, just check specific key poses related to theft
         is_hiding_pose = check_hands_near_body_center(current_landmarks)
         is_bending_crouching = detect_bending(current_landmarks) or detect_crouching(current_landmarks)


         if is_hiding_pose and any("Hand close to" in reason for reason in theft_indicators):
              theft_indicators.append("Suspicious pose (hiding) combined with object proximity")
              theft_score_contribution += 20 # Significant score

         if is_bending_crouching and any("Hand close to" in reason for reason in theft_indicators):
              theft_indicators.append("Suspicious pose (bending/crouching) combined with object proximity")
              theft_score_contribution += 20 # Significant score


    # --- Final Theft Likelihood Score ---
    # The `theft_score_contribution` calculated here is the score contributed by *this frame's* theft indicators.
    # This score will be fed into the overall `suspicion_score` in the main loop.
    is_theft_indicator_present = len(theft_indicators) > 0

    return is_theft_indicator_present, theft_indicators, theft_score_contribution

# --- Main Detection Function (Modified) ---

def detect_suspicious_activities():
    """Main detection function that runs in a separate thread."""
    global camera, output_frame, lock, is_detecting, alert_history
    global last_alert_time, suspicious_activity_counter, suspicion_score
    global object_tracking_history, activity_Q
    global last_detected_objects

    # Clear state on start
    activity_Q.clear()
    object_tracking_history.clear()
    last_detected_objects = []
    last_alert_time = 0
    suspicious_activity_counter = 0
    suspicion_score = 0.0

    if camera is None or not camera.isOpened():
        logger.error("Detection thread started but camera is not open.")
        is_detecting = False
        return # Exit thread if no camera

    # Configure camera buffer size
    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimize buffer
        logger.info(f"Camera buffer size set to 1.")
    except Exception as e:
        logger.warning(f"Could not set camera buffer size: {e}", exc_info=True)

    logger.info("Starting detection loop...")
    is_detecting = True

    frame_count = 0
    last_process_time = time.time()

    while is_detecting and camera is not None and camera.isOpened():
        current_wall_time = time.time() # Use wall time for timeouts/cooldowns

        # Read frame regardless of processing interval to keep buffer fresh
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame or frame is empty. Attempting to re-open camera...")
            # Try to re-open camera
            if not setup_camera(index=CONFIG['camera_index'], source=CONFIG['video_source'], resolution=CONFIG['resolution']):
                 logger.error("Failed to re-open camera. Ending detection loop.")
                 is_detecting = False # Stop detection if camera fails persistently
                 break # Exit the loop
            else:
                 logger.info("Camera re-opened successfully. Continuing detection.")
                 continue # Skip processing this problematic frame

        frame_count += 1

        # Skip frames to reduce processing load
        if frame_count % CONFIG["frame_skip"] != 0:
             # We already read the frame to keep buffer fresh, just skip processing
             continue # Skip processing for this frame

        # --- Frame Processing ---
        process_start_time = time.time()

        try:
            # Resize frame for faster processing
            frame = cv2.resize(frame, CONFIG["resolution"])
            
            # Process the frame with YOLO and Keras model
            output_img, current_frame_objects, activity_label, activity_confidence = process_frame(frame)
            
            # --- Detection and Scoring Logic ---
            frame_suspicion_score = 0.0 # Score contributed by THIS frame
            suspicious_reasons = set() # Reasons detected IN THIS frame
            
            # Add suspicious objects to reasons
            for obj in current_frame_objects:
                if obj['name'] in CONFIG["suspicious_objects"] and obj['confidence'] > CONFIG["confidence_threshold"]:
                    suspicious_reasons.add(f"Suspicious object detected: {obj['name']}")
                    frame_suspicion_score += 20  # High score for suspicious objects

            # Add activity recognition if suspicious
            if activity_label.lower() not in ['normal', 'walking', 'standing'] and activity_confidence > 70:
                suspicious_reasons.add(f"Suspicious activity detected: {activity_label}")
                frame_suspicion_score += 15  # Score for suspicious activity
            
            # --- Update Total Suspicion ---
            # Accumulate score over time, with decay
            decay_rate = 0.95 # Keep 95% of the score from previous frame
            suspicion_score = suspicion_score * decay_rate + frame_suspicion_score

            # Counter increments based on presence of *any* suspicious reason in this frame
            if len(suspicious_reasons) > 0:
                suspicious_activity_counter += 1
            else:
                # Decrease counter slightly if no suspicious reasons in this frame
                suspicious_activity_counter = max(0, suspicious_activity_counter - 0.2) # Slower decay

            # Reset counters after timeout if no new strong indicators
            if current_wall_time - last_alert_time > CONFIG["alert_cooldown"] and frame_suspicion_score < 10:
                 if suspicious_activity_counter > 0 or suspicion_score > 0:
                     logger.debug("Detection timeout reached, resetting counters.") # Use debug for less spam
                 suspicious_activity_counter = 0
                 suspicion_score = 0.0

            # --- Alert Triggering Logic ---
            alert_triggered = False
            alert_message = ""
            alert_type = ""
            confidence_level = 0

            reasons_list = list(suspicious_reasons) # Convert set to list for display

            # Regular Suspicious Activity Alert
            if ((suspicious_activity_counter >= CONFIG["alert_threshold"] and suspicion_score >= CONFIG["alert_threshold"] * 5) or # Require both counter and score
                suspicion_score >= CONFIG["alert_threshold"] * 15 or # High score alone
                len(reasons_list) >= 2) and \
               (current_wall_time - last_alert_time > CONFIG["alert_cooldown"]):

                 alert_triggered = True
                 alert_type = "Suspicious Activity"
                 alert_message = "Suspicious activity detected: " + "; ".join(reasons_list)
                 # Confidence based on total score and counter
                 confidence_level = min(95, int(suspicion_score * 0.5 + suspicious_activity_counter * 5))
                 confidence_level = max(30, confidence_level) # Minimum confidence

                 logger.info(f"--- ALERT TRIGGERED --- Score: {suspicion_score:.1f}, Counter: {suspicious_activity_counter:.1f}, Frame Score: {frame_suspicion_score:.1f}")

            # --- Handle Triggered Alert ---
            if alert_triggered:
                 timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                 alert_data = {
                     "timestamp": timestamp_str,
                     "message": alert_message,
                     "confidence": confidence_level,
                     "suspicion_score": round(suspicion_score, 1),
                     "alert_type": alert_type
                 }

                 with lock:
                     alert_history.append(alert_data) # Add to in-memory history

                 # Save alert to database asynchronously or quickly
                 try:
                     with app.app_context():
                         camera_id = f"Camera_{CONFIG.get('camera_index', 'Unknown')}" # Use .get for safety
                         db_alert = Alert(alert_type=alert_type, message=alert_message[:250], camera_id=camera_id) # Truncate message
                         db.session.add(db_alert)
                         db.session.commit()
                         logger.info(f"Alert saved to database: {alert_type} - {alert_message[:50]}...")
                 except Exception as e:
                     logger.error(f"Failed to save alert to database: {e}", exc_info=True)

                 socketio.emit('new_alert', alert_data)
                 last_alert_time = current_wall_time

                 # Automatically capture screenshot for alerts
                 try:
                     capture_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                     screenshot_dir = os.path.join(app.static_folder, 'screenshots')

                     if not os.path.exists(screenshot_dir):
                         os.makedirs(screenshot_dir)

                     # Use alert type in filename
                     filename_prefix = "suspicious_alert"
                     screenshot_path = os.path.join(screenshot_dir, f'{filename_prefix}_{capture_timestamp_str}.jpg')

                     # Draw alert text on the frame before saving screenshot
                     display_img = output_img.copy() # Draw on a copy for screenshot
                     # Choose text color based on alert type
                     text_color = (0, 165, 255) # Orange

                     cv2.putText(display_img, f"{alert_type} ({confidence_level}%): {alert_message[:40]}...",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, text_color, 2)

                     success = cv2.imwrite(screenshot_path, display_img)

                     if success:
                         logger.info(f"Alert screenshot captured: {screenshot_path}")
                     else:
                         logger.error("Failed to save alert screenshot")

                 except Exception as e:
                     logger.error(f"Error capturing alert screenshot: {e}", exc_info=True)

                 # After any alert, slightly reduce score/counter to allow for recovery
                 suspicion_score *= 0.5 # Keep 50% of score
                 suspicious_activity_counter = max(0, suspicious_activity_counter - 2)

            # --- Store current objects for next frame's reference ---
            last_detected_objects = current_frame_objects.copy()

            # --- Display Information on Frame ---
            # Display current suspicion score and counter
            cv2.putText(output_img, f"Score: {suspicion_score:.1f}", (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(output_img, f"Counter: {int(suspicious_activity_counter)}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display current status / reasons
            if len(reasons_list) > 0:
                 status_text = ", ".join(reasons_list[:2]) # Show up to 2 main reasons
                 if len(reasons_list) > 2: status_text += "..."
                 cv2.putText(output_img, f"Status: {status_text}", (10, 80), # Position below alerts
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Yellow color

            # Display frame processing time and FPS
            process_end_time = time.time()
            processing_time_ms = (process_end_time - process_start_time) * 1000
            # Calculate effective FPS based on time between processing start times
            time_since_last_process = process_end_time - last_process_time
            effective_fps = 1.0 / time_since_last_process if time_since_last_process > 0 else 0

            cv2.putText(output_img, f"FPS: {effective_fps:.1f}", (frame_width - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            last_process_time = process_end_time

            # --- Update Global Output Frame ---
            with lock:
                output_frame = output_img.copy()

        except Exception as e:
            logger.error(f"Major frame processing error: {e}", exc_info=True)
            # If an error occurs during processing, clear the output_frame to indicate an issue
            # Or replace with an error frame
            # with lock:
            #      output_frame = None # Or create a black frame with error text
            continue # Continue loop, hoping it recovers

    # Cleanup after loop ends
    logger.info("Detection loop ended.")
    is_detecting = False
    if camera is not None:
        try:
            camera.release()
            logger.info("Camera released after loop ended.")
        except Exception as e:
             logger.error(f"Error releasing camera after loop: {e}", exc_info=True)
        camera = None # Ensure global camera is None
    with lock:
        output_frame = None # Clear output frame


# --- Flask Routes (Keep as is unless needing changes) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/servilance')
def servilance():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('servilance.html')

@app.route('/video_feed')
def video_feed():
    # The generate_frames function now correctly uses the global output_frame
    # and provides a fallback.
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# The generate_frames function from the previous response is suitable:
# def generate_frames():
#     global output_frame, camera, lock, is_detecting
#     fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#     cv2.putText(fallback_frame, "No Camera Feed Available", (100, 240),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     while True:
#         try:
#             with lock:
#                 if output_frame is not None:
#                     frame_to_send = output_frame.copy()
#                 else:
#                     # If detection is off or frame isn't ready, provide a placeholder
#                     frame_to_send = fallback_frame.copy()
#
#             flag, encoded_image = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 90])
#             if not flag: continue
#
#             frame_data = b'--frame\r\n' \
#                         b'Content-Type: image/jpeg\r\n\r\n' + \
#                         bytearray(encoded_image) + \
#                         b'\r\n'
#             yield frame_data
#
#             time.sleep(0.03) # Small delay to reduce CPU usage on the server side feed

# (Assuming generate_frames is defined elsewhere or copied from previous)
# Pasting generate_frames here for completeness if it was missed:
def generate_frames():
    global output_frame, camera, lock, is_detecting

    # Create a fallback frame (black image with text)
    fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(fallback_frame, "No Camera Feed Available", (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(fallback_frame, "Waiting for detection to start...", (150, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


    while True:
        try:
            # Read frame directly from camera if output_frame is not available and detection is off
            with lock:
                if output_frame is not None:
                    frame_to_send = output_frame.copy()
                else:
                    # If detection is off, but camera is open, show live feed without overlay
                    if not is_detecting and camera is not None and camera.isOpened():
                        ret, frame = camera.read()
                        if ret and frame is not None:
                            frame_to_send = cv2.resize(frame, CONFIG["resolution"]) # Resize raw feed too
                        else:
                            frame_to_send = fallback_frame.copy() # Camera failed
                    else:
                        frame_to_send = fallback_frame.copy() # Detection off or camera not open


            # Encode and send the frame
            flag, encoded_image = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not flag:
                continue # Failed to encode

            frame_data = b'--frame\r\n' \
                        b'Content-Type: image/jpeg\r\n\r\n' + \
                        bytearray(encoded_image) + \
                        b'\r\n'
            yield frame_data

            # Add a short delay to reduce CPU usage for the streaming thread
            # This rate limiting is important for the video feed
            time.sleep(0.04) # Adjust based on desired stream FPS (e.g., 0.04 for ~25fps)

        except Exception as e:
            logger.error(f"Error in generate_frames: {e}", exc_info=True)

            # Send fallback frame on error
            try:
                flag, encoded_image = cv2.imencode('.jpg', fallback_frame)
                if flag:
                    frame_data = b'--frame\r\n' \
                                b'Content-Type: image/jpeg\r\n\r\n' + \
                                bytearray(encoded_image) + \
                                b'\r\n'
                    yield frame_data
            except Exception as e_fallback:
                logger.error(f"Error sending fallback frame: {e_fallback}", exc_info=True)

            # Add delay before retrying or stopping
            time.sleep(1) # Wait longer if there was an error


@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global detection_thread, is_detecting, camera, output_frame, suspicious_activity_counter, suspicion_score, last_alert_time

    if is_detecting:
        return jsonify({"status": "info", "message": "Detection already running"})

    # Update config first if provided
    if request.json:
        updated_config = request.json
        for key, value in updated_config.items():
            if key in CONFIG:
                 try:
                     # Attempt type conversion
                     if isinstance(CONFIG[key], int):
                          CONFIG[key] = int(value)
                     elif isinstance(CONFIG[key], float):
                          CONFIG[key] = float(value)
                     elif isinstance(CONFIG[key], bool):
                          if isinstance(value, str):
                                CONFIG[key] = value.lower() in ['true', '1']
                          else:
                                CONFIG[key] = bool(value)
                     elif isinstance(CONFIG[key], list):
                          if isinstance(value, str):
                                CONFIG[key] = [item.strip() for item in value.split(',') if item.strip()]
                          elif isinstance(value, list):
                                CONFIG[key] = value
                          else:
                              logger.warning(f"Config key {key}: Unexpected list value type {type(value)}")
                              continue
                     elif isinstance(CONFIG[key], tuple): # Handle resolution tuple (e.g., "640,480")
                         if isinstance(value, str):
                             try:
                                 w, h = map(int, value.split(','))
                                 CONFIG[key] = (w, h)
                             except ValueError:
                                 raise ValueError(f"Invalid format for tuple config: {value}. Expected 'width,height'.")
                         elif isinstance(value, (list, tuple)) and len(value) == 2:
                              CONFIG[key] = (int(value[0]), int(value[1]))
                         else:
                             raise ValueError(f"Unexpected value type/format for tuple config: {value}")
                     else: # string or other types - assign directly
                           CONFIG[key] = value
                     logger.info(f"Updated config: {key} = {CONFIG[key]}")
                 except ValueError as e:
                     logger.error(f"Failed to convert config value for key {key}: {value}. Error: {e}")
                     # Optionally return an error or warning to the client
                     pass # Continue processing other keys
                 except Exception as e:
                      logger.error(f"Unexpected error updating config key {key}: {e}", exc_info=True)
                      pass


    # Setup camera *after* config is updated, using the latest CONFIG values
    # This is crucial if camera_index or resolution was changed via config
    if not setup_camera(index=CONFIG['camera_index'], source=CONFIG['video_source'], resolution=CONFIG['resolution']):
        return jsonify({"status": "error", "message": "Failed to setup camera. Check logs."})

    # Reset detection state when starting
    suspicious_activity_counter = 0 # Correct variable name used here
    suspicion_score = 0.0
    last_alert_time = time.time() # Reset cooldown timer
    landmarks_history.clear() # Clear history
    object_tracking_history.clear() # Clear object tracking history
    # last_detected_objects is reset inside the detection thread itself

    # Start the detection thread
    detection_thread = threading.Thread(target=detect_suspicious_activities)
    detection_thread.daemon = True # Daemon threads exit when the main program exits
    detection_thread.start()

    return jsonify({"status": "success", "message": "Detection started", "config": CONFIG})

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    global is_detecting, camera, output_frame, detection_thread

    if not is_detecting:
         return jsonify({"status": "info", "message": "Detection is not running."})

    logger.info("Stopping detection...")
    is_detecting = False # Signal the detection thread to stop

    # The thread should exit its loop because is_detecting is False.
    # We could optionally wait for the thread to join, but for robustness
    # in case the thread gets stuck, it's often better not to block the main thread.
    # The thread is a daemon, so it won't prevent the app from exiting if main stops.

    # Clean up resources regardless
    if camera is not None:
        try:
            camera.release()
            logger.info("Camera released.")
        except Exception as e:
            logger.error(f"Error releasing camera: {e}", exc_info=True)
        camera = None

    # Clear the output frame immediately so the feed shows "No Camera"
    with lock:
        output_frame = None

    # Optional: Clean up detection state variables here too
    # suspicious_activity_counter = 0
    # suspicion_score = 0.0
    # landmarks_history.clear()
    # object_tracking_history.clear()
    # last_detected_objects = [] # Needs lock if modified outside thread

    logger.info("Detection stop requested.")
    return jsonify({"status": "success", "message": "Detection stop requested. System will fully stop processing shortly."}) # Acknowledge request, actual stop happens in thread


@app.route('/get_alerts')
def get_alerts_route():
    try:
        # Fetch recent alerts from the database
        with app.app_context():
            # Order by timestamp descending, limit to 100
            db_alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(100).all()
            alerts_data = []
            for alert in db_alerts:
                # Convert DB object to dictionary
                alerts_data.append({
                    "timestamp": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "message": alert.message,
                    "alert_type": alert.alert_type,
                    "camera_id": alert.camera_id,
                    # Confidence and score are not stored in DB by default Alert model
                    "confidence": 80, # Default confidence for DB alerts
                    "suspicion_score": None # Indicate score is not available from DB
                })

        # Note: We are primarily relying on the DB for historical alerts.
        # The in-memory deque is mostly for quick recent access/display in the thread
        # and SocketIO emission. If you need to guarantee *every* alert shows up
        # even if the DB commit lags, you could merge with the deque, but sorting
        # and de-duplication would be needed, which adds complexity.
        # Relying solely on DB is generally cleaner for historical views.

        return jsonify({"alerts": alerts_data})

    except Exception as e:
        logger.error(f"Error retrieving alerts from database: {e}", exc_info=True)
        # Fall back to in-memory deque if database fails, converting to list
        with lock: # Access deque safely
             in_memory_alerts_list = list(alert_history)
             # Deque appends to the right, so reverse to get newest first if needed, but fetch from DB is already sorted
             # If falling back, the deque already contains the newest, just convert
             return jsonify({"alerts": list(alert_history), "warning": "Database access failed, showing limited in-memory alerts."})


@app.route('/clear_alerts', methods=['POST'])
def clear_alerts_route():
    global alert_history
    with lock: # Clear in-memory history safely
        alert_history.clear()

    try:
        with app.app_context():
            num_deleted = Alert.query.delete()
            db.session.commit()
            logger.info(f"Cleared {num_deleted} alerts from database")
    except Exception as e:
        logger.error(f"Failed to clear alerts from database: {e}", exc_info=True)
        db.session.rollback() # Rollback on error
        return jsonify({"status": "error", "message": f"Failed to clear database alerts: {e}"}), 500

    return jsonify({"status": "success", "message": "Alerts cleared"})

@app.route('/get_config')
def get_config_route():
    # Return a copy to prevent external modification of the original CONFIG dict
    return jsonify({"status": "success", "config": CONFIG.copy()})

@app.route('/update_config', methods=['POST'])
def update_config_route():
    global CONFIG
    if not request.json:
        return jsonify({"status": "error", "message": "No JSON data provided"}), 400

    updated_keys = []
    errors = {}
    temp_config = CONFIG.copy() # Update a temporary copy first

    for key, value in request.json.items():
        if key in temp_config:
            try:
                 # Attempt type conversion based on the type in the existing CONFIG
                 if isinstance(temp_config[key], int):
                      temp_config[key] = int(value)
                 elif isinstance(temp_config[key], float):
                      temp_config[key] = float(value)
                 elif isinstance(temp_config[key], bool):
                      if isinstance(value, str):
                           temp_config[key] = value.lower() in ['true', '1', 'yes', 'on']
                      else:
                           temp_config[key] = bool(value)
                 elif isinstance(temp_config[key], list):
                      if isinstance(value, str):
                           temp_config[key] = [item.strip() for item in value.split(',') if item.strip()]
                      elif isinstance(value, list):
                           temp_config[key] = value
                      else:
                           raise ValueError(f"Unexpected value type for list config: {type(value)}")
                 elif isinstance(temp_config[key], tuple): # Handle tuple config like resolution (e.g., "640,480")
                     if isinstance(value, str):
                         try:
                             vals = [int(v.strip()) for v in value.split(',') if v.strip()]
                             if len(vals) == len(temp_config[key]): # Check if number of elements matches
                                 temp_config[key] = tuple(vals)
                             else:
                                 raise ValueError(f"Incorrect number of values for tuple config. Expected {len(temp_config[key])}, got {len(vals)}")
                         except ValueError:
                             raise ValueError(f"Invalid format for tuple config: {value}. Expected comma-separated numbers.")
                     elif isinstance(value, (list, tuple)) and len(value) == len(temp_config[key]):
                          temp_config[key] = tuple(int(v) for v in value) # Ensure elements are int
                     else:
                         raise ValueError(f"Unexpected value type/format for tuple config: {value}")
                 else: # string or other types - assign directly
                       temp_config[key] = value

                 updated_keys.append(key)
                 # logger.info(f"Pending config update: {key} = {temp_config[key]}") # Log pending changes

            except ValueError as e:
                errors[key] = f"Invalid value or format for '{key}': {value}. Expected type/format mismatch. Error: {e}"
                logger.error(f"Validation error for config key {key}: {errors[key]}")
            except Exception as e:
                 errors[key] = f"An unexpected error occurred validating key '{key}': {e}"
                 logger.error(f"Unexpected error validating config key {key}: {errors[key]}", exc_info=True)
        else:
             errors[key] = f"Config key '{key}' not found."
             logger.warning(f"Attempted to update non-existent config key: {key}")


    if not errors:
        # Apply validated changes to the global CONFIG

        CONFIG = temp_config
        logger.info("CONFIG updated successfully.")
        # Note: Some changes (like resolution, camera_index) require restarting detection to take effect.
        # The front-end should ideally handle this logic or user notification.
        if any(key in updated_keys for key in ['camera_index', 'video_source', 'resolution']):
             logger.warning("Camera/Resolution config updated. Restart detection for changes to take full effect.")
             # You might want to automatically stop/restart detection here, but be cautious
             # as it can interrupt active surveillance unexpectedly. Better to let the user decide.


    if errors:
         status = "warning" if updated_keys else "error"
         message = "Configuration updated with errors for some keys." if updated_keys else "Failed to update configuration due to errors."
         response_status = 200 if updated_keys else 400 # Return 200 if *some* updates succeeded
         return jsonify({"status": status, "message": message, "config": CONFIG.copy(), "errors": errors}), response_status
    else:
         return jsonify({"status": "success", "message": "Configuration updated", "config": CONFIG.copy()})


@app.route('/get_latest_alert')
def get_latest_alert_route():
    with lock:
        if not alert_history:
            return jsonify({"alert_active": False})

        # Get the newest alert from the deque
        latest_alert = alert_history[-1]
        current_time = time.time()
        try:
            # Parse timestamp from string in alert history
            alert_time = datetime.strptime(latest_alert.get("timestamp"), "%Y-%m-%d %H:%M:%S").timestamp()
        except (ValueError, TypeError):
            logger.error(f"Could not parse timestamp for latest alert: {latest_alert.get('timestamp')}")
            return jsonify({"alert_active": False}) # Cannot determine age

        # Only show alert as "active" if it's very recent
        if current_time - alert_time <= CONFIG["alert_cooldown"] + 5: # Show for cooldown + a few seconds
            return jsonify({
                "alert_active": True,
                "alert_message": latest_alert.get("message", "Alert"),
                "alert_type": latest_alert.get("alert_type", "Alert"), # Provide default type
                "confidence": latest_alert.get("confidence", 50), # Provide default confidence
                "timestamp": latest_alert.get("timestamp")
            })

    return jsonify({"alert_active": False})


@app.route('/get_status')
def get_status_route():
    global camera, is_detecting, suspicious_activity_counter, suspicion_score # Correct variable name used here

    camera_status = False
    camera_resolution = None
    if camera is not None:
        try:
             camera_status = camera.isOpened()
             if camera_status:
                 camera_resolution = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        except Exception as e:
             logger.error(f"Error checking camera status properties: {e}", exc_info=True)
             camera_status = False
             camera_resolution = "Error"

    return jsonify({
        "status": "success",
        "system_status": {
            "is_detecting": is_detecting,
            "camera_status": camera_status,
            "camera_resolution": camera_resolution,
            "yolo_loaded": yolo_model is not None,
            "pose_loaded": pose is not None,
            "current_suspicion_score": round(suspicion_score, 1),
            "current_suspicion_counter": int(suspicious_activity_counter) # Correct variable name used here
        }
    })

@app.route('/capture_screenshot', methods=['POST'])
def capture_screenshot():
    global output_frame, camera, is_detecting

    screenshot_frame = None

    # Prioritize the latest processed frame if detection is active
    if is_detecting and output_frame is not None:
        with lock:
            screenshot_frame = output_frame.copy()
            logger.info("Captured screenshot from latest processed output_frame.")
    else:
        # If detection is off or no output_frame, try to grab a fresh frame from the camera
        if camera is not None and camera.isOpened():
            try:
                # Read a fresh frame (might block briefly)
                ret, frame = camera.read()
                if ret and frame is not None:
                     screenshot_frame = cv2.resize(frame, CONFIG["resolution"]) # Resize for consistency
                     logger.info("Captured screenshot from raw camera feed.")
                else:
                    logger.warning("Camera is open but failed to read a fresh frame for screenshot.")
            except Exception as e:
                 logger.error(f"Error reading camera for screenshot: {e}", exc_info=True)
        else:
            logger.warning("Camera not available or detection not active. Cannot capture screenshot.")


    if screenshot_frame is None:
         return jsonify({
             "status": "error",
             "message": "No frame available to capture. Camera not open or detection not started."
         }), 400


    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = os.path.join(app.static_folder, 'screenshots')

        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        screenshot_path = os.path.join(screenshot_dir, f'manual_screenshot_{timestamp}.jpg') # Use 'manual' prefix

        success = cv2.imwrite(screenshot_path, screenshot_frame)

        if not success:
            logger.error(f"Failed to save screenshot file: {screenshot_path}. Check permissions.")
            return jsonify({
                "status": "error",
                "message": "Failed to save screenshot file. Check permissions and disk space."
            }), 500

        # Convert image to base64 for web display in response
        ret, buffer = cv2.imencode('.jpg', screenshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            logger.error("Failed to encode screenshot image for base64.")
            return jsonify({
                "status": "error",
                "message": "Failed to encode screenshot image."
            }), 500

        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        screenshot_filename = os.path.basename(screenshot_path)
        relative_path = f'screenshots/{screenshot_filename}'

        logger.info(f"Screenshot captured successfully: {screenshot_path}")
        return jsonify({
            "status": "success",
            "message": "Screenshot captured successfully",
            "filename": screenshot_filename,
            # path is useful for client to request the image file later
            "path": url_for('serve_screenshot', filename=screenshot_filename), # Use url_for for correct path
            "image_data": jpg_as_text # Send base64 data for immediate preview
        })

    except Exception as e:
        logger.error(f"Screenshot capture process error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Screenshot error: {str(e)}"
        }), 500

@app.route('/screenshots')
def screenshots():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    screenshot_dir = os.path.join(app.static_folder, 'screenshots')
    # Ensure directory exists before listing
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
        screenshots_list = []
        logger.warning(f"Screenshot directory was missing, created: {screenshot_dir}")
    else:
        screenshots_list = []
        try:
            for filename in os.listdir(screenshot_dir):
                # Include all relevant image types
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    file_path = os.path.join(screenshot_dir, filename)
                    try:
                        file_stat = os.stat(file_path)
                        # Get creation date (might vary by OS, ctime is creation on Unix, modified on Windows sometimes)
                        # For reliability, parsing filename is better if consistent
                        creation_time_os = datetime.fromtimestamp(file_stat.st_ctime)

                        # Try to extract timestamp from filename based on known prefixes
                        timestamp = creation_time_os # Default if parsing fails
                        try:
                             if filename.startswith('alert_screenshot_') or filename.startswith('theft_alert_') or filename.startswith('manual_screenshot_'):
                                 parts = filename.rsplit('_', 1) # Split once from right
                                 if len(parts) > 1:
                                     date_time_str = parts[1].split('.')[0] # Get the datetime part before extension
                                     timestamp = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')
                             # Add other filename formats if needed
                        except ValueError:
                             logger.debug(f"Could not parse timestamp from filename: {filename}, using OS ctime.")
                             pass # Fallback to OS ctime

                        screenshots_list.append({
                            'filename': filename,
                            'path': url_for('serve_screenshot', filename=filename), # Use url_for for serving route
                            'timestamp': timestamp,
                            'size': file_stat.st_size
                        })
                    except Exception as e:
                         logger.error(f"Error processing screenshot file {filename}: {e}", exc_info=True)
                         # Skip this file if unable to process
        except Exception as e:
             logger.error(f"Error listing screenshot directory {screenshot_dir}: {e}", exc_info=True)
             screenshots_list = [] # Clear list if listing fails

    # Sort by timestamp (newest first)
    screenshots_list.sort(key=lambda x: x['timestamp'], reverse=True)

    # We don't need to fetch all alerts from DB here unless we plan to match them
    # The current logic just lists screenshots.

    return render_template('screenshots.html', screenshots=screenshots_list) # Pass the list of screenshot dicts

@app.route('/screenshots/<filename>')
def serve_screenshot(filename):
    # Protect this route if screenshots are sensitive
    if 'user_id' not in session:
        # Instead of redirect, return 401 Unauthorized for API-like access to assets
        # Or redirect to login if this route is meant for browser access
        return redirect(url_for('login')) # Assuming this is browser access

    screenshot_dir = os.path.join(app.static_folder, 'screenshots')
    # Use send_from_directory for safe serving
    try:
        # Ensure the filename is safe and doesn't contain directory traversal attempts
        # send_from_directory handles basic safety, but extra validation might be needed
        # if filenames can contain unusual characters.
        return send_from_directory(screenshot_dir, filename)
    except FileNotFoundError:
        logger.warning(f"Screenshot file not found: {filename}")
        return "File not found", 404
    except Exception as e:
        logger.error(f"Error serving screenshot file {filename}: {e}", exc_info=True)
        return "Internal server error", 500


@app.route('/delete_screenshot/<filename>', methods=['POST'])
def delete_screenshot(filename):
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    screenshot_dir = os.path.join(app.static_folder, 'screenshots')
    file_path = os.path.join(screenshot_dir, filename)

    # Basic safety check: ensure filename doesn't try to traverse directories
    # os.path.abspath converts to absolute path and resolves '..'
    # Check if the resulting absolute path is still within the screenshot directory
    safe_filepath = os.path.abspath(file_path)
    safe_screenshot_dir = os.path.abspath(screenshot_dir)

    if not safe_filepath.startswith(safe_screenshot_dir):
         logger.warning(f"Attempted directory traversal in delete_screenshot: {filename}")
         return jsonify({"status": "error", "message": "Invalid filename"}), 400


    try:
        if os.path.exists(safe_filepath):
            os.remove(safe_filepath)
            logger.info(f"Screenshot deleted: {filename}")
            return jsonify({"status": "success", "message": f"Screenshot '{filename}' deleted"})
        else:
            logger.warning(f"Attempted to delete non-existent screenshot: {filename}")
            return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting screenshot '{filename}': {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Failed to delete screenshot: {str(e)}"}), 500

# --- Auth Routes (Keep as is) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, redirect to dashboard
    if 'user_id' in session:
         return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not all([username, password]):
            error = 'Username and password are required'
        else:
            # Allow login by username or email
            user = User.query.filter((User.username == username) | (User.email == username)).first()
            if user and user.check_password(password):
                session['user_id'] = user.username # Store username in session
                logger.info(f"User '{user.username}' logged in.")
                # Redirect to dashboard or a referrer page
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            else:
                error = 'Invalid username or password'

    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Optional: Prevent registration if user is already logged in
    # if 'user_id' in session:
    #      return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Basic Server-side Validation
        if not all([username, email, password, confirm_password]):
            error = 'All fields are required.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters long.'
        elif not any(char.isdigit() for char in password):
            error = 'Password must contain at least one digit.'
        elif not any(char.isalpha() for char in password):
             error = 'Password must contain at least one letter.'
        elif not any(char in '!@#$%^&*()_+' for char in password): # Example special chars
             error = 'Password must contain at least one special character (!@#$%^&*()_+).'
        elif not any(char.isupper() for char in password):
             error = 'Password must contain at least one uppercase letter.'
        # Add email format validation
        # import re
        # if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        #     error = "Invalid email format."
        else:
            existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                error = 'Username or email already exists.'
            else:
                try:
                    new_user = User(username, email, password)
                    db.session.add(new_user)
                    db.session.commit()
                    logger.info(f"New user registered: {username}")
                    # Flash a success message and redirect to login
                    # flash('Registration successful! Please log in.', 'success') # Requires Flask flash messaging setup
                    return redirect(url_for('login'))
                except Exception as e:
                    db.session.rollback() # Roll back the transaction on error
                    logger.error(f"Error during user registration: {e}", exc_info=True)
                    error = 'An error occurred during registration. Please try again.'


    return render_template('register.html', error=error)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        # Redirect to login, but add 'next' parameter so user is sent back here after login
        return redirect(url_for('login', next=request.url))
    # Fetch user details if needed
    # user = User.query.filter_by(username=session['user_id']).first()
    # if not user: # Should not happen if session['user_id'] is valid
    #      session.pop('user_id', None) # Clear invalid session
    #      return redirect(url_for('login'))
    # return render_template('profile.html', user=user)
    return render_template('profile.html')


@app.route('/logout')
def logout():
    user_id = session.pop('user_id', None)
    if user_id:
        logger.info(f"User '{user_id}' logged out.")
    # Redirect to index or login page
    return redirect(url_for('index'))

# Error handlers (Keep as is)
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 Not Found: {request.path}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}", exc_info=True) # Log exception traceback
    return render_template('500.html'), 500


# --- Main execution block ---
if __name__ == '__main__':
    # Ensure upload directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    
    # Attempt to setup the camera on startup
    logger.info("Attempting initial camera setup...")
    initial_camera_success = setup_camera(index=CONFIG['camera_index'], source=CONFIG['video_source'], resolution=CONFIG['resolution'])

    if not initial_camera_success:
        logger.warning("Initial camera setup failed. The video feed might show a fallback image until detection is started and camera becomes available.")
    else:
         logger.info("Initial camera setup appears successful.")

    # Run the Flask SocketIO server
    logger.info("Starting Flask SocketIO server...")
    # Use allow_unsafe_werkzeug=True for Flask 2.2+ if not using a production WSGI server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

# --- Process frame with YOLO and Keras Model ---
def process_frame(frame):
    """Process a single frame with YOLO and Keras model. Returns processed frame and detections."""
    if frame is None:
        return None, []
    
    # Make a copy of the frame for drawing
    output_img = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize results
    current_frame_objects = []
    activity_label = "Unknown"
    activity_confidence = 0
    
    # Run YOLO object detection
    if yolo_model is not None:
        try:
            # Run YOLO on the frame
            yolo_results = yolo_model(frame, conf=CONFIG["confidence_threshold"], verbose=False)

            for result in yolo_results:
                for box in result.boxes:
                    # Check if box and conf exist before accessing
                    if box.xyxy.numel() > 0 and box.conf.numel() > 0 and box.cls.numel() > 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0].item())

                        # Check if class_id is valid in names list
                        if cls_id < len(result.names):
                            label = result.names[cls_id]
                        else:
                            label = f"unknown_cls_{cls_id}"
                            logger.warning(f"Unknown class ID detected: {cls_id}")

                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)

                        current_frame_objects.append({
                            'name': label,
                            'confidence': conf,
                            'bbox': [x, y, w, h],
                            'class_id': cls_id
                        })

                        # Draw object bounding box with color coding
                        color = (0, 255, 0)  # Green by default
                        if label in CONFIG["suspicious_objects"]:
                            color = (0, 0, 255)  # Red for suspicious
                        elif label in CONFIG["theft_related_objects"]:
                            color = (255, 0, 255)  # Purple for theft-related

                        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(output_img, f"{label} {conf:.2f}", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            logger.error(f"YOLO processing error: {e}", exc_info=True)
            current_frame_objects = []
    
    # Run Keras activity recognition
    if keras_model is not None and lb is not None:
        try:
            # Preprocess the frame for activity recognition
            activity_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            activity_frame = cv2.resize(activity_frame, (64, 64)).astype("float32")
            activity_frame -= mean
            
            # Make prediction
            preds = keras_model.predict(np.expand_dims(activity_frame, axis=0))[0]
            activity_Q.append(preds)
            
            # Perform prediction averaging over history
            results = np.array(activity_Q).mean(axis=0)
            max_prob = np.max(results)
            i = np.argmax(results)
            
            if lb is not None:
                activity_label = lb.classes_[i]
                activity_confidence = max_prob * 100
                
                # Decide whether to show alert based on confidence
                rest = 1 - max_prob
                diff = max_prob - rest
                
                if diff > CONFIG["activity_threshold"]:
                    text_color = (0, 255, 0)  # Green for high confidence
                else:
                    text_color = (0, 165, 255)  # Orange for low confidence
                
                # Add activity recognition text to the frame
                text = f"Activity: {activity_label} - {activity_confidence:.2f}%"
                cv2.putText(output_img, text, (35, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
                
        except Exception as e:
            logger.error(f"Keras model processing error: {e}", exc_info=True)
    
    return output_img, current_frame_objects, activity_label, activity_confidence

# --- Add video upload and processing route ---
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"})
    
    file = request.files['video']
    
    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return jsonify({"status": "error", "message": "No video file selected"})
    
    # Check file type is video
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"status": "error", "message": "Invalid video file format"})
    
    try:
        # Create a unique filename
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file
        file.save(filepath)
        logger.info(f"Video uploaded successfully: {filepath}")
        
        # Process the video
        processed_path = process_uploaded_video(filepath)
        
        if processed_path:
            # Return the path to the processed video
            video_url = url_for('static', filename=os.path.join('uploads/processed', os.path.basename(processed_path)))
            return jsonify({
                "status": "success",
                "message": "Video processed successfully",
                "video_path": video_url
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to process video. Check logs for details."
            })
    
    except Exception as e:
        logger.error(f"Error processing uploaded video: {e}", exc_info=True)
        return jsonify({
            "status": "error", 
            "message": f"An error occurred: {str(e)}"
        })

def process_uploaded_video(filepath):
    """Process an uploaded video file with object detection and activity recognition."""
    try:
        # Open the video file
        video = cv2.VideoCapture(filepath)
        if not video.isOpened():
            logger.error(f"Could not open video file: {filepath}")
            return None
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {filepath}, FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        # Create a unique output filename
        output_filename = f"processed_{os.path.basename(filepath)}"
        output_path = os.path.join(processed_folder, output_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            logger.error(f"Could not create output video writer at {output_path}")
            return None
        
        # Clear activity queue for this video
        activity_Q.clear()
        
        # Process each frame
        frame_count = 0
        skip_frames = max(1, int(total_frames / 500))  # Process at most 500 frames for large videos
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % skip_frames != 0 and frame_count > 60:  # Always process the first 60 frames
                writer.write(frame)  # Write original frame
                continue
            
            # Process frame
            processed_frame, objects, activity, confidence = process_frame(frame)
            
            # Write the processed frame
            writer.write(processed_frame)
            
            # Log progress periodically
            if frame_count % 100 == 0:
                logger.info(f"Processing video: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Release resources
        video.release()
        writer.release()
        
        logger.info(f"Video processing complete: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in process_uploaded_video: {e}", exc_info=True)
        return None