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
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import bcrypt
import torch
from collections import deque # For efficient history
import pickle
from tensorflow.keras.models import load_model
import tempfile
import uuid
from werkzeug.utils import secure_filename
import face_recognition
import shutil
from dotenv import load_dotenv
from flask_mail import Mail, Message

load_dotenv()

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
app.config['KNOWN_FACES_DIR'] = os.path.join('static', 'known_faces')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

db = SQLAlchemy(app)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')
mail = Mail(app)

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

# Email alert variables
last_email_alert_time = 0
email_alert_interval = 30  # Send email every 30 seconds

# Object tracking history
object_tracking_history = {}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a subfolder for processed videos
processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
os.makedirs(processed_folder, exist_ok=True)

# Create a folder for known faces if it doesn't exist
os.makedirs(app.config['KNOWN_FACES_DIR'], exist_ok=True)

# For activity recognition model
keras_model = None
lb = None
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
activity_Q = deque(maxlen=10)  # Default size 10 for activity prediction queue

# Face recognition variables
known_face_encodings = []
known_face_names = []
owner_detected = False
owner_detection_timeout = 30  # Seconds to ignore alerts after owner is detected
last_owner_detection_time = 0

# Configuration parameters
CONFIG = {
    "confidence_threshold": 0.4,      # YOLO confidence threshold
    "alert_cooldown": 3,              # Cooldown between alerts (seconds)
    "frame_processing_interval": 0.03, # Target frame processing interval (seconds)
    "frame_skip": 1,                  # Process every N frames. 1 means process every frame.
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
    "activity_threshold": 0.80,       # Threshold for activity detection confidence
    "face_recognition_enabled": True, # Enable face recognition
    "face_recognition_tolerance": 0.6, # Face recognition tolerance (lower is stricter)
    "restricted_room": True,          # Is this a restricted room camera
    "owner_detection_timeout": 60,    # Seconds to ignore alerts after owner detection
    "face_detection_interval": 5,     # Check for face every N frames
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
        # logger.info(f"Activity labels: {lb.classes_}")
    else:
        logger.error(f"Label binarizer file not found at {CONFIG['label_bin_path']}")
except Exception as e:
    logger.error(f"Failed to load Keras model: {e}", exc_info=True)
    keras_model = None
    lb = None

# Load known face encodings
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    # Check if known_faces directory exists
    if not os.path.exists(app.config['KNOWN_FACES_DIR']):
        logger.warning(f"Known faces directory not found: {app.config['KNOWN_FACES_DIR']}")
        return
    
    # Loop through each person's directory
    for person_name in os.listdir(app.config['KNOWN_FACES_DIR']):
        person_dir = os.path.join(app.config['KNOWN_FACES_DIR'], person_name)
        if not os.path.isdir(person_dir):
            continue
        
        # Loop through each image of the person
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Load the image
            image_path = os.path.join(person_dir, image_name)
            face_image = face_recognition.load_image_file(image_path)
            
            # Try to get face encoding
            encodings = face_recognition.face_encodings(face_image)
            if len(encodings) > 0:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)
                logger.info(f"Loaded face encoding for {person_name} from {image_name}")
            else:
                logger.warning(f"No face found in {image_path}")
    
    logger.info(f"Loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} people")

# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    is_owner = db.Column(db.Boolean, default=False)  # Added owner status

    def __repr__(self):
        return f'<User {self.username}>'

    def __init__(self, username, email, password, is_owner=False):
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.is_owner = is_owner

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Alert model for logging
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    camera_id = db.Column(db.String(50), nullable=False)
    owner_verified = db.Column(db.Boolean, default=False)  # Added owner verification status

    def __repr__(self):
        return f'<Alert {self.timestamp} - {self.alert_type}>'

    def __init__(self, alert_type, message, camera_id, owner_verified=False):
        self.alert_type = alert_type
        self.message = message[:255]  # Ensure message fits DB column
        self.camera_id = camera_id
        self.owner_verified = owner_verified

# Create the database and tables
with app.app_context():
    db.create_all()
    # Check if any owners exist, if not create a default one
    if not User.query.filter_by(is_owner=True).first():
        try:
            default_owner = User.query.filter_by(username='admin').first()
            if default_owner:
                default_owner.is_owner = True
                db.session.commit()
                logger.info("Set existing admin user as an owner")
            else:
                admin_owner = User('admin', 'admin@example.com', 'admin_password', is_owner=True)
                db.session.add(admin_owner)
                db.session.commit()
                logger.info("Created default admin owner account. CHANGE THE PASSWORD!")
        except Exception as e:
            logger.error(f"Error setting up default owner: {e}", exc_info=True)
            db.session.rollback()
    
    # Set the email alert time to current time to avoid immediate emails on startup
    last_email_alert_time = time.time()

# Load known faces
try:
    load_known_faces()
except Exception as e:
    logger.error(f"Error loading known faces: {e}", exc_info=True)

# --- Camera Setup Function ---
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
        # Longer delay to ensure camera is fully released
        time.sleep(1.5)

    try:
        if source and os.path.exists(source):
            logger.info(f"Attempting to open video file: {source}")
            camera = cv2.VideoCapture(source)
        else:
            # Try different backends for camera access
            backends = [
                (cv2.CAP_DSHOW + index, f"DirectShow backend for camera {index}"),  # Windows DirectShow
                (cv2.CAP_MSMF + index, f"Microsoft Media Foundation backend for camera {index}"),  # Windows Media Foundation
                (index, f"Default backend for camera {index}")  # Default backend
            ]
            
            # Explicitly set the correct camera index
            CONFIG['camera_index'] = index
            logger.info(f"Setting camera index to {index}")
            
            for backend, description in backends:
                logger.info(f"Attempting to open camera with {description}")
                try:
                    if isinstance(backend, int) and backend < 10:  # Regular camera index
                        camera = cv2.VideoCapture(backend)
                    else:  # Backend-specific syntax
                        camera = cv2.VideoCapture(backend)
                        
                    if camera.isOpened():
                        logger.info(f"Successfully connected to camera using {description}")
                        # Small delay after opening to make sure it's ready
                        time.sleep(0.5)
                        break
                    else:
                        logger.warning(f"Failed to open camera using {description}")
                        # Release before trying next backend
                        camera.release()
                        camera = None
                except Exception as e:
                    logger.error(f"Error trying {description}: {e}")
                    if camera is not None:
                        camera.release()
                        camera = None
            
            # If all backends failed for the configured index, try alternative indices
            if camera is None or not camera.isOpened():
                logger.warning(f"All backends for camera index {index} failed, trying alternative indices (0, 1)...")
                alternative_indices = [0, 1]  # Try both primary and secondary cameras
                for idx in alternative_indices:
                    if idx == index:
                        continue
                    logger.info(f"Trying camera index {idx} with DirectShow backend")
                    try:
                        # Try DirectShow for alternative indices
                        camera = cv2.VideoCapture(cv2.CAP_DSHOW + idx)
                        if camera.isOpened():
                            CONFIG["camera_index"] = idx  # Update config if successful
                            logger.info(f"Successfully connected to camera at index {idx} with DirectShow")
                            break
                        else:
                            logger.info(f"Trying default backend for camera index {idx}")
                            camera = cv2.VideoCapture(idx)
                            if camera.isOpened():
                                CONFIG["camera_index"] = idx
                                logger.info(f"Successfully connected to camera at index {idx} with default backend")
                                break
                            else:
                                camera.release()
                                camera = None
                    except Exception as e:
                        logger.error(f"Error trying camera index {idx}: {e}")
                        if camera is not None:
                            camera.release()
                            camera = None

        # Verify camera opened successfully
        if not camera or not camera.isOpened():
            logger.error("Error: Could not open video source after trying all options.")
            return False

        # Configure camera buffer size to minimize delay
        try:
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info("Camera buffer size set to 1 to minimize delay")
        except Exception as e:
            logger.warning(f"Could not set camera buffer size: {e}")

        # Try to set resolution
        try:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Requested resolution: {resolution}, Actual resolution: ({int(actual_width)}, {int(actual_height)})")
        except Exception as e:
             logger.warning(f"Could not set resolution {resolution}: {e}", exc_info=True)

        # Try reading test frames with multiple attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempting to read test frame (attempt {attempt+1}/{max_attempts})...")
                
                # Flush any stale frames that might be in buffer
                for _ in range(5):
                    camera.grab()
                
                ret, frame = camera.read()
                if not ret or frame is None:
                    logger.warning(f"Could not read frame on attempt {attempt+1}")
                    if attempt == max_attempts - 1:
                        logger.error("All attempts to read initial frame failed.")
                        camera.release()
                        camera = None
                        return False
                    time.sleep(1)  # Wait before next attempt
                    continue
                    
                # Validate the frame as an additional check
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    logger.warning(f"Invalid frame on attempt {attempt+1}: {type(frame)}")
                    if attempt == max_attempts - 1:
                        logger.error("All attempts returned invalid frames.")
                        camera.release()
                        camera = None
                        return False
                    time.sleep(1)
                    continue
                    
                # Check frame dimensions
                if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                    logger.warning(f"Invalid frame dimensions on attempt {attempt+1}: {frame.shape}")
                    if attempt == max_attempts - 1:
                        logger.error("All attempts returned frames with invalid dimensions.")
                        camera.release()
                        camera = None
                        return False
                    time.sleep(1)
                    continue
                
                # If we get here, we have a valid frame
                logger.info(f"Successfully read test frame on attempt {attempt+1} with dimensions {frame.shape}")
                break
                
            except cv2.error as e:
                logger.warning(f"OpenCV error on attempt {attempt+1}: {e}")
                if attempt == max_attempts - 1:
                    logger.error("All attempts to read initial frame resulted in OpenCV errors.")
                    camera.release()
                    camera = None
                    return False
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt+1}: {e}")
                if attempt == max_attempts - 1:
                    logger.error("All attempts to read initial frame failed with unexpected errors.")
                    camera.release()
                    camera = None
                    return False
                time.sleep(1)

        logger.info("Camera setup successful with valid test frame.")
        return True

    except Exception as e:
        logger.error(f"Camera setup error: {e}", exc_info=True)
        if camera is not None:
            try:
                camera.release()
            except:
                pass
        camera = None
        return False

# --- Face Recognition Functions ---
def recognize_faces(frame, tolerance=CONFIG['face_recognition_tolerance']):
    """
    Detect and recognize faces in a frame.
    Returns:
        list of (name, face_location) tuples,
        True if any owner was recognized
    """
    global known_face_encodings, known_face_names, owner_detected, last_owner_detection_time
    
    if len(known_face_encodings) == 0:
        return [], False
        
    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert from BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    is_owner_present = False
    
    for face_encoding in face_encodings:
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # Check if this is an owner (we assume owners' directories start with "owner_")
                if name.lower().startswith("owner_"):
                    is_owner_present = True
                    name = name[6:]  # Remove "owner_" prefix for display
        
        face_names.append(name)
    
    # Convert back to original frame size
    face_locations_original = []
    for (top, right, bottom, left) in face_locations:
        # Scale back to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        face_locations_original.append((top, right, bottom, left))
    
    # Check if owner detection has timed out
    if owner_detected and time.time() - last_owner_detection_time > CONFIG["owner_detection_timeout"]:
        logger.info("Owner detection timed out")
        owner_detected = False
    
    return list(zip(face_names, face_locations_original)), is_owner_present

def draw_faces(frame, face_info):
    """Draw rectangles for detected faces without labels"""
    for name, (top, right, bottom, left) in face_info:
        # Draw rectangle around the face
        if name.lower() == "unknown":
            color = (0, 0, 255)  # Red for unknown
        else:
            color = (0, 255, 0)  # Green for known
            
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # No labels or text
    
    return frame

# --- Process frame with YOLO, Keras Model and Face Recognition ---
def process_frame(frame, check_faces=True):
    """Process a single frame with YOLO, Keras model, and face recognition. Returns processed frame and detections."""
    global owner_detected, last_owner_detection_time
    
    if frame is None:
        logger.warning("Received None frame in process_frame")
        return None, [], "", 0, False, []
    
    # Check if frame is valid
    if not isinstance(frame, np.ndarray):
        logger.error(f"Invalid frame type: {type(frame)}")
        return None, [], "", 0, False, []
    
    # Check frame dimensions
    if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        logger.error(f"Invalid frame dimensions: {frame.shape}")
        return None, [], "", 0, False, []
    
    try:
        # Make a copy of the frame for drawing
        output_img = frame.copy()
        frame_height, frame_width = frame.shape[:2]
    except Exception as e:
        logger.error(f"Error processing frame: {e}", exc_info=True)
        return None, [], "", 0, False, []
    
    # Initialize results
    current_frame_objects = []
    activity_label = "Unknown"
    activity_confidence = 0
    face_info = []
    is_owner_detected = False
    
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

                        # Draw object bounding box with color coding (changed to theme colors)
                        color = (255, 175, 0)  # Orange by default (matches the theme)
                        if label in CONFIG["suspicious_objects"]:
                            color = (0, 0, 255)  # Red for suspicious
                        elif label in CONFIG["theft_related_objects"]:
                            color = (255, 0, 255)  # Purple for theft-related

                        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
                        # No text labels for object detection
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
                
                                # Don't add activity recognition text to the frame - removed as requested
                
        except Exception as e:
            logger.error(f"Keras model processing error: {e}", exc_info=True)
    
    # Run face recognition if enabled and we need to check faces in this frame
    if CONFIG["face_recognition_enabled"] and check_faces:
        try:
            face_info, is_owner_detected = recognize_faces(frame)
            
            # Update owner detection status
            if is_owner_detected:
                owner_detected = True
                last_owner_detection_time = time.time()
                logger.info("👤 Owner detected in frame!")
                # Emit owner detection event
                socketio.emit('owner_detected', {'is_owner_present': True})
                # Also ensure unauthorized alert is hidden
                socketio.emit('unauthorized_status', {'show_unauthorized_alert': False})
            elif time.time() - last_owner_detection_time > CONFIG["owner_detection_timeout"]:
                owner_detected = False
                # Emit owner detection event
                socketio.emit('owner_detected', {'is_owner_present': False})
                # Show unauthorized alert only if persons are detected (handled in detect_suspicious_activities)
            
            # Draw faces on the frame
            output_img = draw_faces(output_img, face_info)
            
                        # Don't add owner detected text - removed as requested
                
        except Exception as e:
            logger.error(f"Face recognition error: {e}", exc_info=True)
    
        # Don't show room status text - removed as requested
    
    return output_img, current_frame_objects, activity_label, activity_confidence, owner_detected, face_info 

# Add a function to ensure mail is properly initialized
def ensure_mail_initialized():
    """Ensure the mail system is properly initialized in a thread-safe way."""
    global mail
    
    try:
        # Check if mail is already initialized and working
        if mail and hasattr(mail, 'state'):
            return True
            
        # Reinitialize mail with current settings
        with app.app_context():
            mail = Mail(app)
            logger.info("Mail system reinitialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize mail system: {e}", exc_info=True)
        return False

# Modify the beginning of send_email_alert to use this function
def send_email_alert(alert_type, message, screenshot_path=None):
    """Send an email alert with the given message and optional screenshot attachment."""
    global mail, owner_detected
    
    # Skip if owner was detected
    if owner_detected:
        logger.info("Owner detected, skipping email alert")
        return
    
    # Ensure mail is initialized
    if not ensure_mail_initialized():
        logger.error("Cannot send email - mail system not properly initialized")
        return False
    
    # Create thread-local storage for screenshot data
    screenshot_data = None
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            # Read the screenshot data in the main thread to avoid file access issues
            with open(screenshot_path, 'rb') as fp:
                screenshot_data = fp.read()
            logger.info(f"Screenshot loaded: {screenshot_path}, size: {len(screenshot_data)} bytes")
        except Exception as e:
            logger.error(f"Error reading screenshot: {e}")
            screenshot_data = None
    
    # Create a thread to send the email
    def send_email_in_thread():
        try:
            # Check if mail credentials are set
            if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
                logger.error("Mail credentials are not set! Please update them in the dashboard settings.")
                return
            
            # All Flask operations must be inside app context
            with app.app_context():
                try:
                    # Find all users marked as owners
                    owner_users = User.query.filter_by(is_owner=True).all()
                    
                    if not owner_users:
                        logger.warning("No owner accounts found to send email alert")
                        return
                        
                    logger.info(f"Found {len(owner_users)} owner accounts to send alerts to")
                    
                    subject = f"SECURITY ALERT: {alert_type}"
                    
                    # Prepare the email body
                    body = f"Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nAlert Type: {alert_type}\n\nDetails: {message}\n\nThis is an automated alert from your Survion security system."
                    
                    # Send email to each owner
                    for owner in owner_users:
                        try:
                            logger.info(f"Preparing email for owner: {owner.username} ({owner.email})")
                            
                            # Create message within app context
                            msg = Message(
                                subject=subject,
                                recipients=[owner.email],
                                body=body
                            )
                            
                            # Attach screenshot if available
                            if screenshot_data:
                                msg.attach(
                                    filename=os.path.basename(screenshot_path),
                                    content_type="image/jpeg",
                                    data=screenshot_data
                                )
                            
                            # Send the email
                            mail.send(msg)
                            logger.info(f"Email alert sent to owner {owner.username} at {owner.email}")
                            
                        except Exception as msg_error:
                            logger.error(f"Error sending email message to {owner.email}: {msg_error}")
                except Exception as db_error:
                    logger.error(f"Database error in email thread: {db_error}")
        
        except Exception as e:
            logger.error(f"Failed to send email alerts: {e}", exc_info=True)
    
    # Create and start the thread
    email_thread = threading.Thread(target=send_email_in_thread)
    email_thread.daemon = True  # Thread will exit when main program exits
    email_thread.start()
    logger.info(f"Started background thread for sending email alert (type: {alert_type})")
    return True

def test_email_connection():
    """Test email connection and settings."""
    logger.info("Testing email connection...")
    
    try:
        # Log current settings
        logger.info(f"MAIL_SERVER: {app.config['MAIL_SERVER']}")
        logger.info(f"MAIL_PORT: {app.config['MAIL_PORT']}")
        logger.info(f"MAIL_USE_TLS: {app.config['MAIL_USE_TLS']}")
        logger.info(f"MAIL_USERNAME: {app.config['MAIL_USERNAME']}")
        logger.info(f"MAIL_PASSWORD: {'*****' if app.config['MAIL_PASSWORD'] else 'Not set'}")
        
        # Check if credentials are set
        if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
            return False, "Mail credentials are not set! Please update them in the dashboard settings."
        
        # All Flask operations must be inside app context
        with app.app_context():
            # Find all owner accounts
            owner_users = User.query.filter_by(is_owner=True).all()
            
            if not owner_users:
                return False, "No owner accounts found to send test email"
            
            logger.info(f"Found {len(owner_users)} owner accounts to send test emails to")
            
            # Create test message template
            subject = "SURVION Email Test"
            body = f"This is a test email from your Survion security system.\nSent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nThis confirms your email alert system is working correctly. You will now receive alerts at this email address when suspicious activities are detected."
            
            success_count = 0
            error_messages = []
            
            # Send test email to each owner
            for owner in owner_users:
                try:
                    logger.info(f"Sending test email to: {owner.username} ({owner.email})")
                    
                    # Create and send message within app context
                    msg = Message(
                        subject=subject,
                        recipients=[owner.email],
                        body=body
                    )
                    
                    # Try to send the email
                    mail.send(msg)
                    logger.info(f"Test email sent successfully to {owner.email}")
                    success_count += 1
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Test email to {owner.email} failed: {error_msg}")
                    error_messages.append(f"Failed to send to {owner.email}: {error_msg[:100]}...")
        
        # Return appropriate message based on results
        if success_count == len(owner_users):
            return True, f"Test emails sent successfully to all {success_count} owner accounts."
        elif success_count > 0:
            return True, f"Test emails sent to {success_count} out of {len(owner_users)} owner accounts. Errors: {'; '.join(error_messages)}"
        else:
            # If no emails were sent successfully, provide more specific guidance
            if any("Authentication" in msg for msg in error_messages):
                return False, "Authentication failed. Please check your email credentials. For Gmail, you may need to use an App Password."
            elif any("SSL" in msg for msg in error_messages):
                return False, "SSL/TLS error. Check your mail server settings."
            elif any("SMTP" in msg for msg in error_messages):
                return False, "SMTP server error. Check your connection and server settings."
            else:
                return False, f"Failed to send test emails: {'; '.join(error_messages)}"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Test email failed: {error_msg}")
        return False, f"Failed to set up email test: {error_msg}"

def detect_suspicious_activities():
    """Main detection loop that processes frames and detects suspicious activities."""
    global camera, output_frame, lock, is_detecting, suspicious_activity_counter, last_alert_time
    global suspicion_score, last_owner_detection_time, owner_detected, last_email_alert_time
    
    # Clear state on start
    activity_Q.clear()
    object_tracking_history.clear()
    last_detected_objects = []
    last_alert_time = 0
    suspicious_activity_counter = 0
    suspicion_score = 0.0
    owner_detected = False
    last_owner_detection_time = 0
    
    # Variables for the unauthorized access periodic alert
    last_unauthorized_alert_time = 0
    unauthorized_alert_interval = 30  # Send an alert every 30 seconds when no owner detected

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

    try:
        logger.info("Starting detection loop...")
        frame_count = 0
        start_time = time.time()
        
        while is_detecting:
            current_wall_time = time.time() # Use wall time for timeouts/cooldowns

            # Read frame regardless of processing interval to keep buffer fresh
            try:
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
                else:
                    frame_to_send = frame.copy()
            except Exception as e:
                logger.error(f"Error reading camera frame: {e}", exc_info=True)
                # Try to re-open camera
                if not setup_camera(index=CONFIG['camera_index'], source=CONFIG['video_source'], resolution=CONFIG['resolution']):
                     logger.error("Failed to re-open camera after error. Ending detection loop.")
                     is_detecting = False
                     break
                continue

            frame_count += 1

            # Skip frames to reduce processing load
            if frame_count % CONFIG["frame_skip"] != 0:
                 # We already read the frame to keep buffer fresh, just skip processing
                 continue # Skip processing for this frame

            # --- Frame Processing ---
            process_start_time = time.time()

            try:
                # Validate frame before resizing
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    logger.warning(f"Invalid frame before resizing: {type(frame) if frame is not None else 'None'}")
                    continue
                    
                # Check frame dimensions to avoid OpenCV assertion errors
                if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                    logger.warning(f"Invalid frame dimensions: {frame.shape}")
                    continue
                    
                # Validate resolution configuration
                if not isinstance(CONFIG["resolution"], tuple) or len(CONFIG["resolution"]) != 2:
                    logger.error(f"Invalid resolution config: {CONFIG['resolution']}")
                    CONFIG["resolution"] = (640, 480)  # Reset to default
                
                if CONFIG["resolution"][0] <= 0 or CONFIG["resolution"][1] <= 0:
                    logger.error(f"Invalid resolution dimensions: {CONFIG['resolution']}")
                    CONFIG["resolution"] = (640, 480)  # Reset to default
                
                try:
                    # Resize frame for faster processing
                    frame = cv2.resize(frame, CONFIG["resolution"])
                except cv2.error as e:
                    logger.error(f"OpenCV resize error: {e}", exc_info=True)
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error during resize: {e}", exc_info=True)
                    continue
                
                # Decide if we should check for faces in this frame
                check_faces = frame_count % CONFIG["face_detection_interval"] == 0
                
                # First, do a quick object detection pass to check if any persons are in the frame
                # Process the frame with YOLO only to detect people
                temp_output_img, current_frame_objects, _, _, _, _ = process_frame(frame, False)
                
                # Count number of persons detected
                person_count = 0
                for obj in current_frame_objects:
                    if obj['name'].lower() == 'person' and obj['confidence'] > CONFIG["confidence_threshold"]:
                        person_count += 1
                
                # If no persons detected, skip the rest of processing
                if person_count == 0:
                    # Just update the output frame with the basic detection (no alerts/analysis)
                    with lock:
                        output_frame = temp_output_img
                    # Ensure unauthorized alert is hidden when no persons are detected
                    socketio.emit('unauthorized_status', {'show_unauthorized_alert': False})
                    continue
                
                # If we get here, persons were detected - do full processing
                # Process the frame with YOLO, Keras model, and face recognition if needed
                output_img, current_frame_objects, activity_label, activity_confidence, current_owner_status, face_info = process_frame(frame, check_faces)
                
                # Emit unauthorized status if persons detected but no owner
                if person_count > 0 and not current_owner_status:
                    socketio.emit('unauthorized_status', {'show_unauthorized_alert': True})
                else:
                    socketio.emit('unauthorized_status', {'show_unauthorized_alert': False})
                
                        # --- Detection and Scoring Logic ---
                frame_suspicion_score = 0.0 # Score contributed by THIS frame
                suspicious_reasons = set() # Reasons detected IN THIS frame
                
                # Count number of persons detected
                person_count = 0
                for obj in current_frame_objects:
                    if obj['name'].lower() == 'person' and obj['confidence'] > CONFIG["confidence_threshold"]:
                        person_count += 1
                
                # If 5 or more persons detected and owner not present, add this as a reason
                if person_count >= 5 and not owner_detected:
                    suspicious_reasons.add(f"{person_count} unauthorized persons detected")
                    frame_suspicion_score += 50  # High score for multiple unauthorized persons
                
                # Add suspicious objects to reasons
                for obj in current_frame_objects:
                    if obj['name'] in CONFIG["suspicious_objects"] and obj['confidence'] > CONFIG["confidence_threshold"]:
                        suspicious_reasons.add(f"Suspicious object detected: {obj['name']}")
                        frame_suspicion_score += 20  # High score for suspicious objects

                # Add activity recognition if suspicious
                if activity_label.lower() not in ['normal', 'walking', 'standing'] and activity_confidence > 70:
                    suspicious_reasons.add(f"Suspicious activity detected: {activity_label}")
                    frame_suspicion_score += 15  # Score for suspicious activity
                    
                # Add restricted room entry detection
                if CONFIG["restricted_room"] and not owner_detected:
                    if activity_label.lower() != 'normal' and activity_confidence > 60:
                        suspicious_reasons.add("Unauthorized person in restricted area")
                        frame_suspicion_score += 30  # Higher score for unauthorized access
                
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
                     
                # --- Send periodic unauthorized access alerts if no owner is detected ---
                if not owner_detected and person_count > 0 and current_wall_time - last_unauthorized_alert_time > unauthorized_alert_interval:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    unauthorized_alert = {
                        "timestamp": timestamp_str,
                        "message": "UNAUTHORIZED ACCESS DETECTED: No authorized person present",
                        "confidence": 95,
                        "suspicion_score": 90.0,
                        "alert_type": "Unauthorized Access",
                        "owner_verified": False
                    }
                    
                    with lock:
                        alert_history.append(unauthorized_alert)
                    
                    # Save alert to database
                    try:
                        with app.app_context():
                            camera_id = f"Camera_{CONFIG.get('camera_index', 'Unknown')}"
                            db_alert = Alert(
                                alert_type="Unauthorized Access", 
                                message="UNAUTHORIZED ACCESS DETECTED: No authorized person present", 
                                camera_id=camera_id,
                                owner_verified=False
                            )
                            db.session.add(db_alert)
                            db.session.commit()
                            logger.info("Unauthorized access alert saved to database")
                    except Exception as e:
                        logger.error(f"Failed to save unauthorized access alert to database: {e}", exc_info=True)
                    
                    # Send the alert to all connected clients
                    socketio.emit('new_alert', unauthorized_alert)
                    last_unauthorized_alert_time = current_wall_time
                    logger.info("--- UNAUTHORIZED ACCESS ALERT SENT ---")

                # --- Alert Triggering Logic ---
                alert_triggered = False
                alert_message = ""
                alert_type = ""
                confidence_level = 0

                reasons_list = list(suspicious_reasons) # Convert set to list for display
                
                # Don't trigger alerts if owner is verified and timeout hasn't expired
                alert_suppressed_by_owner = owner_detected
                
                # Regular Suspicious Activity Alert - if not suppressed by owner verification
                if not alert_suppressed_by_owner and (
                   (suspicious_activity_counter >= CONFIG["alert_threshold"] and suspicion_score >= CONFIG["alert_threshold"] * 5) or # Require both counter and score
                    suspicion_score >= CONFIG["alert_threshold"] * 15 or # High score alone
                    len(reasons_list) >= 2) and \
                   (current_wall_time - last_alert_time > CONFIG["alert_cooldown"]):

                     alert_triggered = True
                     
                                      # Special alert for 5+ persons detected                 if any(reason.endswith("unauthorized persons detected") for reason in reasons_list):                     alert_type = "Multiple Persons Alert"                     for reason in reasons_list:                         if reason.endswith("unauthorized persons detected"):                             alert_message = "UNAUTHORIZED ACCESS DETECTED: " + reason                             break                     confidence_level = min(99, int(suspicion_score + 40))  # Higher confidence for multiple unauthorized persons                 # Special alert for restricted room                 elif CONFIG["restricted_room"] and any("Unauthorized person" in reason for reason in reasons_list):                     alert_type = "Restricted Area Alert"                     alert_message = "UNAUTHORIZED ACCESS DETECTED: " + "; ".join(reasons_list)                     confidence_level = min(98, int(suspicion_score + 30))  # Higher confidence for unauthorized access                 else:                     alert_type = "Suspicious Activity"                     alert_message = "Suspicious activity detected: " + "; ".join(reasons_list)                     # Confidence based on total score and counter                     confidence_level = min(95, int(suspicion_score * 0.5 + suspicious_activity_counter * 5))
                     
                     confidence_level = max(30, confidence_level) # Minimum confidence

                     logger.info(f"--- ALERT TRIGGERED --- Score: {suspicion_score:.1f}, Counter: {suspicious_activity_counter:.1f}, Frame Score: {frame_suspicion_score:.1f}, Owner Verified: {owner_detected}")

                # --- Handle Triggered Alert ---
                if alert_triggered:
                     timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                     alert_data = {
                         "timestamp": timestamp_str,
                         "message": alert_message,
                         "confidence": confidence_level,
                         "suspicion_score": round(suspicion_score, 1),
                         "alert_type": alert_type,
                         "owner_verified": owner_detected
                     }

                     with lock:
                         alert_history.append(alert_data) # Add to in-memory history

                     # Save alert to database asynchronously or quickly
                     try:
                         with app.app_context():
                             camera_id = f"Camera_{CONFIG.get('camera_index', 'Unknown')}" # Use .get for safety
                             db_alert = Alert(
                                 alert_type=alert_type, 
                                 message=alert_message[:250], 
                                 camera_id=camera_id,
                                 owner_verified=owner_detected
                             ) # Truncate message
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
                         if "Multiple Persons" in alert_type:
                             filename_prefix = "multiple_persons_alert"
                         elif "Restricted Area" in alert_type:
                             filename_prefix = "restricted_alert"
                         else:
                             filename_prefix = "suspicious_alert"
                         
                         screenshot_path = os.path.join(screenshot_dir, f'{filename_prefix}_{capture_timestamp_str}.jpg')

                         # Draw alert text on the frame before saving screenshot
                         display_img = output_img.copy() # Draw on a copy for screenshot
                         # Choose text color based on alert type
                         text_color = (0, 0, 255) if "Restricted Area" in alert_type else (0, 165, 255) # Red or Orange

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

                        # --- No display information on frame (removed as requested) ---
                process_end_time = time.time()
                last_process_time = process_end_time

                # --- Update Global Output Frame ---
                with lock:
                    output_frame = output_img.copy()

                # Check if enough time has passed to send an email alert
                now = time.time()
                if now - last_email_alert_time >= email_alert_interval and not owner_detected:
                    try:
                        # Update last email alert time immediately to prevent multiple emails during processing
                        last_email_alert_time = now
                        
                        # Capture a screenshot for email in a thread-safe way
                        with lock:
                            if output_frame is not None:
                                # Use a unique timestamp to prevent file conflicts
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                screenshot_filename = f"alert_email_{timestamp}.jpg"
                                screenshot_path = os.path.join(app.config['UPLOAD_FOLDER'], screenshot_filename)
                                
                                # Save a copy of the frame to prevent modifications during write
                                frame_for_email = output_frame.copy()
                                
                                # Release the lock before disk I/O
                            else:
                                logger.warning("No output frame available for email screenshot")
                                continue
                                
                        # Perform disk I/O outside the lock
                        try:
                            cv2.imwrite(screenshot_path, frame_for_email)
                            logger.info(f"Email screenshot saved to {screenshot_path}")
                        except Exception as e:
                            logger.error(f"Failed to save email screenshot: {e}")
                            screenshot_path = None
                        
                        # Send email with screenshot in background thread
                        alert_message = f"Suspicious activity detected. Screenshot captured at {timestamp}."
                        
                        # This function now handles threading internally
                        send_email_alert("Suspicious Activity", alert_message, screenshot_path)
                        
                        # Log the alert
                        logger.info(f"Email alert triggered at {timestamp}")
                        
                        # Add to alert history and database - do this in a separate thread to avoid blocking
                        def update_alert_history():
                            try:
                                with app.app_context():
                                    try:
                                        camera_id = f"Camera_{CONFIG.get('camera_index', 'Unknown')}"
                                        db_alert = Alert(
                                            alert_type="Email Alert", 
                                            message="Sent email alert with screenshot", 
                                            camera_id=camera_id,
                                            owner_verified=False
                                        )
                                        db.session.add(db_alert)
                                        db.session.commit()
                                        logger.info("Email alert saved to database")
                                        
                                        # Also add to in-memory alert history
                                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        with lock:
                                            alert_history.append({
                                                "timestamp": timestamp_str,
                                                "message": "Sent email alert with screenshot",
                                                "alert_type": "Email Alert",
                                                "confidence": 100,
                                                "suspicion_score": 0,
                                                "owner_verified": False
                                            })
                                    except Exception as e:
                                        logger.error(f"Failed to save email alert to database: {e}", exc_info=True)
                            except Exception as e:
                                logger.error(f"Error in update_alert_history thread: {e}", exc_info=True)
                        
                        # Start the database update in a separate thread
                        history_thread = threading.Thread(target=update_alert_history)
                        history_thread.daemon = True
                        history_thread.start()
                        
                    except Exception as e:
                        logger.error(f"Error in periodic email alert: {e}", exc_info=True)
                        # Don't update the timer on error so we can retry

            except Exception as e:
                logger.error(f"Major frame processing error: {e}", exc_info=True)
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

    except Exception as e:
        logger.error(f"Error in detection loop: {e}", exc_info=True)
        is_detecting = False
    
    logger.info("Detection loop stopped.")

# --- Helper Functions for Face Enrollment ---
def add_owner_face(username, image_data):
    """
    Add a new owner face to the known faces database.
    Image data can be either a base64 encoded string or raw bytes.
    Returns success status and message.
    """
    try:
        # Create owner directory if it doesn't exist
        owner_dir = os.path.join(app.config['KNOWN_FACES_DIR'], f"owner_{username}")
        os.makedirs(owner_dir, exist_ok=True)
        
        # Generate a unique filename for the face image
        filename = f"{username}_{int(time.time())}.jpg"
        file_path = os.path.join(owner_dir, filename)
        
        # Convert image data if needed (base64 to bytes)
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Extract the base64 part
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        elif isinstance(image_data, str):
            # Assume it's already base64 without the data URL prefix
            image_bytes = base64.b64decode(image_data)
        else:
            # Assume it's already bytes
            image_bytes = image_data
        
        # Save the image file
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        # Now try to detect a face in the saved image
        image = face_recognition.load_image_file(file_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            # No face detected, delete the file
            os.remove(file_path)
            return False, "No face detected in the image. Please try again with a clearer image."
            
        if len(face_encodings) > 1:
            # Multiple faces detected
            os.remove(file_path)
            return False, "Multiple faces detected. Please provide an image with only your face."
        
        # Successfully added face - reload known faces
        load_known_faces()
        return True, f"Owner face for {username} added successfully."
        
    except Exception as e:
        logger.error(f"Error adding owner face: {e}", exc_info=True)
        return False, f"Error adding face: {str(e)}"

def remove_owner_face(username):
    """Remove all face images for an owner"""
    try:
        owner_dir = os.path.join(app.config['KNOWN_FACES_DIR'], f"owner_{username}")
        
        if not os.path.exists(owner_dir):
            return False, f"No face data found for {username}"
            
        # Remove the directory and all its contents
        shutil.rmtree(owner_dir)
        
        # Reload known faces
        load_known_faces()
        return True, f"All face data for {username} removed successfully."
        
    except Exception as e:
        logger.error(f"Error removing owner face: {e}", exc_info=True)
        return False, f"Error removing face data: {str(e)}"

# --- Video Stream Generator ---
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
                        try:
                            ret, frame = camera.read()
                            if ret and frame is not None:
                                # Validate frame before resizing
                                if not isinstance(frame, np.ndarray) or frame.size == 0:
                                    logger.warning("Invalid frame from camera in generate_frames")
                                    frame_to_send = fallback_frame.copy()
                                elif frame.shape[0] <= 0 or frame.shape[1] <= 0:
                                    logger.warning(f"Invalid frame dimensions in generate_frames: {frame.shape}")
                                    frame_to_send = fallback_frame.copy()
                                else:
                                    try:
                                        frame_to_send = cv2.resize(frame, CONFIG["resolution"]) # Resize raw feed
                                    except cv2.error as e:
                                        logger.error(f"OpenCV resize error in generate_frames: {e}")
                                        frame_to_send = fallback_frame.copy()
                            else:
                                frame_to_send = fallback_frame.copy() # Camera failed
                        except cv2.error as e:
                            logger.error(f"OpenCV error in generate_frames: {e}")
                            frame_to_send = fallback_frame.copy()
                        except Exception as e:
                            logger.error(f"Exception in generate_frames: {e}")
                            frame_to_send = fallback_frame.copy()
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

# --- Video Processing Function ---
def process_uploaded_video(filepath):
    """Process an uploaded video file with object detection and face recognition."""
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
        
        # Track owner detection throughout the video
        video_owner_detected = False
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % skip_frames != 0 and frame_count > 60:  # Always process the first 60 frames
                writer.write(frame)  # Write original frame
                continue
            
            # Check for faces periodically
            check_faces = frame_count % CONFIG["face_detection_interval"] == 0 or frame_count < 30
            
            # Process frame
            processed_frame, objects, activity, confidence, is_owner, face_info = process_frame(frame, check_faces)
            
            # Update video's owner detection status
            if is_owner:
                video_owner_detected = True
            
                        # Don't add text showing owner verification - removed as requested
            
            # Write the processed frame
            writer.write(processed_frame)
            
            # Log progress periodically
            if frame_count % 100 == 0:
                logger.info(f"Processing video: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Release resources
        video.release()
        writer.release()
        
        logger.info(f"Video processing complete: {output_path}, Owner detected: {video_owner_detected}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in process_uploaded_video: {e}", exc_info=True)
        return None 

# --- Flask Routes ---

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

@app.route('/owner_faces')
def owner_faces():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if user is an owner
    user = User.query.filter_by(username=session['user_id']).first()
    if not user or not user.is_owner:
        return redirect(url_for('dashboard'))
    
    return render_template('owner_faces.html')

@app.route('/video_feed')
def video_feed():
    # The generate_frames function uses the global output_frame
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global detection_thread, is_detecting, camera, output_frame, suspicious_activity_counter, suspicion_score, last_alert_time, owner_detected

    if is_detecting:
        return jsonify({"status": "info", "message": "Detection already running"})

    # Update config first if provided
    camera_index_updated = False
    if request.json:
        updated_config = request.json
        for key, value in updated_config.items():
            if key in CONFIG:
                 try:
                     # Explicitly handle camera_index to track if it was updated
                     if key == 'camera_index':
                         camera_index_updated = True
                         CONFIG[key] = int(value)
                         logger.info(f"Updated camera_index to {CONFIG[key]}")
                     # Attempt type conversion
                     elif isinstance(CONFIG[key], int):
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
    suspicious_activity_counter = 0
    suspicion_score = 0.0
    last_alert_time = time.time() # Reset cooldown timer
    activity_Q.clear() # Clear history
    object_tracking_history.clear() # Clear object tracking history
    
    # Reset owner detection status
    owner_detected = False
    
    # After camera change, we want to prioritize face detection
    if camera_index_updated:
        logger.info(f"Setting face detection interval to 1 for initial owner detection")
        CONFIG['face_detection_interval'] = 1
        # After 10 seconds, revert to normal interval
        threading.Timer(10.0, lambda: CONFIG.update({'face_detection_interval': 5})).start()

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

    logger.info("Detection stop requested.")
    return jsonify({"status": "success", "message": "Detection stop requested. System will fully stop processing shortly."})

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

# --- Face Enrollment Routes ---
@app.route('/enroll_face', methods=['GET', 'POST'])
def enroll_face():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if the user is an owner
    user = User.query.filter_by(username=session['user_id']).first()
    if not user or not user.is_owner:
        return redirect(url_for('dashboard'))
    
    if request.method == 'GET':
        return render_template('enroll_face.html')
    
    # POST request - process the face image
    if 'face_image' not in request.form:
        return jsonify({"status": "error", "message": "No face image provided"})
    
    face_image_data = request.form['face_image']
    success, message = add_owner_face(user.username, face_image_data)
    
    return jsonify({
        "status": "success" if success else "error",
        "message": message
    })

@app.route('/remove_face', methods=['POST'])
def remove_face():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    # Check if the user is an owner
    user = User.query.filter_by(username=session['user_id']).first()
    if not user or not user.is_owner:
        return jsonify({"status": "error", "message": "Only owners can remove faces"}), 403
    
    success, message = remove_owner_face(user.username)
    
    return jsonify({
        "status": "success" if success else "error",
        "message": message
    })

@app.route('/check_owner_status')
def check_owner_status():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    # Check if the user is an owner
    user = User.query.filter_by(username=session['user_id']).first()
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404
    
    # Check if user has face data
    owner_dir = os.path.join(app.config['KNOWN_FACES_DIR'], f"owner_{user.username}")
    has_face_data = os.path.exists(owner_dir) and len(os.listdir(owner_dir)) > 0
    
    return jsonify({
        "status": "success",
        "is_owner": user.is_owner,
        "has_face_data": has_face_data
    })

@app.route('/make_owner', methods=['POST'])
def make_owner():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    # Check if the current user is already an owner
    admin_user = User.query.filter_by(username=session['user_id']).first()
    if not admin_user or not admin_user.is_owner:
        return jsonify({"status": "error", "message": "Only existing owners can make others owners"}), 403
    
    # Get the username to make an owner
    data = request.json
    if not data or 'username' not in data:
        return jsonify({"status": "error", "message": "Username required"}), 400
    
    username = data['username']
    user = User.query.filter_by(username=username).first()
    
    if not user:
        return jsonify({"status": "error", "message": f"User {username} not found"}), 404
    
    # Make the user an owner
    user.is_owner = True
    db.session.commit()
    
    return jsonify({
        "status": "success",
        "message": f"User {username} is now an owner"
    })

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
                    "owner_verified": alert.owner_verified,
                    # Confidence and score are not stored in DB by default Alert model
                    "confidence": 80, # Default confidence for DB alerts
                    "suspicion_score": None # Indicate score is not available from DB
                })

        return jsonify({"alerts": alerts_data})

    except Exception as e:
        logger.error(f"Error retrieving alerts from database: {e}", exc_info=True)
        # Fall back to in-memory deque if database fails, converting to list
        with lock: # Access deque safely
             in_memory_alerts_list = list(alert_history)
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
        
        # Reload known faces if face_recognition settings changed
        if any(key.startswith('face_recognition_') for key in updated_keys):
            try:
                load_known_faces()
            except Exception as e:
                logger.error(f"Error reloading known faces after config update: {e}", exc_info=True)


    if errors:
         status = "warning" if updated_keys else "error"
         message = "Configuration updated with errors for some keys." if updated_keys else "Failed to update configuration due to errors."
         response_status = 200 if updated_keys else 400 # Return 200 if *some* updates succeeded
         return jsonify({"status": status, "message": message, "config": CONFIG.copy(), "errors": errors}), response_status
    else:
         return jsonify({"status": "success", "message": "Configuration updated", "config": CONFIG.copy()})

@app.route('/update_email_settings', methods=['POST'])
def update_email_settings():
    """Update email configuration settings."""
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Authentication required"}), 401
        
    try:
        data = request.json
        
        # Update email interval if provided
        if 'email_interval' in data:
            global email_alert_interval
            email_alert_interval = int(data['email_interval'])
            
        # Update mail username and password if provided
        changes_made = False
        
        if 'mail_username' in data and data['mail_username']:
            app.config['MAIL_USERNAME'] = data['mail_username']
            os.environ['MAIL_USERNAME'] = data['mail_username']
            changes_made = True
            
        if 'mail_password' in data and data['mail_password']:
            app.config['MAIL_PASSWORD'] = data['mail_password']
            os.environ['MAIL_PASSWORD'] = data['mail_password']
            changes_made = True
            
        # Reinitialize mail if credentials changed
        if changes_made:
            if ensure_mail_initialized():
                logger.info("Mail system successfully reinitialized with new settings")
            else:
                logger.warning("Mail system reinitialization failed, but settings were updated")
            
        return jsonify({
            "status": "success", 
            "message": "Email settings updated successfully",
            "current_interval": email_alert_interval
        })
        
    except Exception as e:
        logger.error(f"Error updating email settings: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

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
                "timestamp": latest_alert.get("timestamp"),
                "owner_verified": latest_alert.get("owner_verified", False)
            })

    return jsonify({"alert_active": False})

@app.route('/get_status')
def get_status_route():
    global camera, is_detecting, suspicious_activity_counter, suspicion_score

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
            "keras_loaded": keras_model is not None,
            "face_recognition_enabled": CONFIG["face_recognition_enabled"],
            "restricted_room": CONFIG["restricted_room"],
            "current_suspicion_score": round(suspicion_score, 1),
            "current_suspicion_counter": int(suspicious_activity_counter),
            "owner_detected": owner_detected
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

# --- Auth Routes ---
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
    
    # Fetch user details
    user = User.query.filter_by(username=session['user_id']).first()
    if not user:
         session.pop('user_id', None) # Clear invalid session
         return redirect(url_for('login'))
    
    # Check if user has face data
    owner_dir = os.path.join(app.config['KNOWN_FACES_DIR'], f"owner_{user.username}")
    has_face_data = os.path.exists(owner_dir) and len(os.listdir(owner_dir)) > 0
    
    return render_template('profile.html', user=user, is_owner=user.is_owner, has_face_data=has_face_data)

@app.route('/logout')
def logout():
    user_id = session.pop('user_id', None)
    if user_id:
        logger.info(f"User '{user_id}' logged out.")
    # Redirect to index or login page
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 Not Found: {request.path}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}", exc_info=True) # Log exception traceback
    return render_template('500.html'), 500 

@app.route('/test_email', methods=['POST'])
def test_email_route():
    """Route to test email sending."""
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    
    # Execute the test in a separate thread with proper error handling
    def test_email_async():
        try:
            success, message = test_email_connection()
            # Emit the result via Socket.IO
            socketio.emit('email_test_result', {
                "status": "success" if success else "error",
                "message": message
            })
            logger.info(f"Email test completed and result emitted via Socket.IO: {success}, {message[:50]}...")
        except Exception as e:
            logger.error(f"Error in test_email_async thread: {e}", exc_info=True)
            # Still try to emit a response even if there was an error
            try:
                socketio.emit('email_test_result', {
                    "status": "error",
                    "message": f"Unexpected error during email test: {str(e)}"
                })
            except:
                logger.error("Failed to emit socket.io message after error")
    
    # Start the test in background thread with proper error handling
    try:
        email_test_thread = threading.Thread(target=test_email_async)
        email_test_thread.daemon = True
        email_test_thread.start()
        logger.info("Email test thread started successfully")
    except Exception as e:
        logger.error(f"Failed to start email test thread: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Failed to start email test: {str(e)}"
        })
    
    # Return immediately with pending status
    return jsonify({
        "status": "pending",
        "message": "Email test started. Results will be sent via Socket.IO."
    })

# --- Main execution block ---
if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(app.config['KNOWN_FACES_DIR'], exist_ok=True)
    
    # Make sure screenshots directory exists
    os.makedirs(os.path.join(app.static_folder, 'screenshots'), exist_ok=True)
    
    # Load known faces
    try:
        load_known_faces()
    except Exception as e:
        logger.error(f"Error loading known faces on startup: {e}", exc_info=True)
    
    # Do not setup camera on startup
    logger.info("Camera setup deferred until detection is started.")

    # Run the Flask SocketIO server
    logger.info("Starting Flask SocketIO server...")
    # Use allow_unsafe_werkzeug=True for Flask 2.2+ if not using a production WSGI server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)