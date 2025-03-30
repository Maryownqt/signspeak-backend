import cv2
import numpy as np
import torch
from collections import deque
import time
from ultralytics import YOLO
import mediapipe as mp

# --- Load YOLOv8 Hand Detector ---
yolo_weights = r"E:\SIGNSPEAK\deployment\models\best.pt"  # Update as needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(yolo_weights).to(device)

# --- Load Trained SignBERT-like Model ---
# Ensure your SignBERT class definition is available.
from training_finetune import SignBERT  # Replace with your actual module or definition.
# Load the model with saved weights.
model_signbert = SignBERT(input_dim=63, d_model=256, num_layers=3, num_heads=8, dropout=0.1, num_classes=40)
model_signbert.load_state_dict(torch.load(r"E:\SIGNSPEAK\deployment\models\signbert_finetuned_weighted_v4.pth", map_location=device))
model_signbert.to(device)
model_signbert.eval()

# --- Automatically Extract Hyperparameters from the Model ---
# Here we extract the input dimension from the token_embedding layer.
input_dim = model_signbert.token_embedding.in_features  # Expected to be 63, but now extracted automatically.
d_model = model_signbert.token_embedding.out_features
try:
    num_layers = len(model_signbert.transformer_encoder.layers)
except AttributeError:
    num_layers = "Unknown"
# In our model definition, we set num_heads explicitly; here we assume it is 8.
num_heads = 8  
try:
    dropout = model_signbert.transformer_encoder.layers[0].dropout
except AttributeError:
    dropout = "Unknown"
print("Extracted Model Hyperparameters:")
print(f"  input_dim: {input_dim}")
print(f"  d_model: {d_model}")
print(f"  num_layers: {num_layers}")
print(f"  num_heads: {num_heads}")
print(f"  dropout: {dropout}")

# --- MediaPipe Hands Setup ---
mp_hands = mp.solutions.hands
# We allow detection of up to 2 hands.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2)

def extract_pose_keypoints(image):
    """
    Extract hand keypoints from an RGB image using MediaPipe Hands.
    Returns a flattened array of shape (input_dim,) if successful, else a zeros vector.
    """
    results = hands.process(image)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        keypoints = np.array(keypoints, dtype=np.float32)
        # If the detected keypoints don't match the model's expected dimension, adjust accordingly.
        if keypoints.shape[0] != input_dim:
            if keypoints.shape[0] < input_dim:
                keypoints = np.pad(keypoints, (0, input_dim - keypoints.shape[0]), mode='constant')
            else:
                keypoints = keypoints[:input_dim]
        return keypoints
    else:
        return np.zeros(input_dim, dtype=np.float32)

# --- Sequence Buffer Settings ---
SEQUENCE_LENGTH = 20  # Number of frames per sequence
IMG_SIZE = 224        # Resize cropped hand region to this size

# Create a buffer to hold keypoint sequences.
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

def process_frame(frame):
    """
    Given a BGR frame, run YOLO to detect a hand,
    crop the region (with an expanded bounding box), and extract keypoints.
    Returns:
      - A keypoints vector of shape (input_dim,) if a hand is detected, else None.
      - The frame with the bounding box drawn.
    """
    results = yolo_model(frame, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        return None, frame

    bbox_xywh = []
    confidences = []
    for det in results.boxes:
        x_min, y_min, x_max, y_max = det.xyxy[0].tolist()
        conf = det.conf[0].item()
        w = x_max - x_min
        h = y_max - y_min
        x_c = x_min + w / 2.0
        y_c = y_min + h / 2.0
        bbox_xywh.append([x_c, y_c, w, h])
        confidences.append(conf)
    
    if len(bbox_xywh) == 0:
        return None, frame

    bbox_xywh = np.array(bbox_xywh, dtype=np.float32)
    confidences = np.array(confidences, dtype=np.float32)
    best_idx = np.argmax(confidences)
    x_c, y_c, w, h = bbox_xywh[best_idx]
    left = int(x_c - w / 2)
    top = int(y_c - h / 2)
    right = left + int(w)
    bottom = top + int(h)
    
    # Expand the bounding box by 20%
    margin_x = int(0.2 * w)
    margin_y = int(0.2 * h)
    left_expanded = max(0, left - margin_x)
    top_expanded = max(0, top - margin_y)
    right_expanded = min(frame.shape[1], right + margin_x)
    bottom_expanded = min(frame.shape[0], bottom + margin_y)
    
    cv2.rectangle(frame, (left_expanded, top_expanded), (right_expanded, bottom_expanded), (0, 255, 0), 2)
    
    # Crop, convert to RGB, and resize.
    cropped = frame[top_expanded:bottom_expanded, left_expanded:right_expanded]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_resized = cv2.resize(cropped_rgb, (IMG_SIZE, IMG_SIZE))
    
    keypoints = extract_pose_keypoints(cropped_resized)
    return keypoints, frame

def run_inference(source=0):
    """
    Run real-time gesture recognition from a video source.
    Set source=0 for webcam, or pass a video file path.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_skip = max(1, int(fps / 15))  # Process about 15 frames per second
    
    predicted_class = None
    sequence_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optionally, skip frames to control processing speed.
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        
        keypoints, frame_vis = process_frame(frame)
        if keypoints is not None:
            sequence_buffer.append(keypoints)
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                seq = np.stack(sequence_buffer, axis=0)  # Shape: (20, input_dim)
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 20, input_dim)
                with torch.no_grad():
                    logits = model_signbert(seq_tensor, mode='finetune')
                    probs = torch.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                sequence_buffer.clear()
                sequence_start_time = time.time()
        
        if predicted_class is not None:
            cv2.putText(frame_vis, f"Gesture: {predicted_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Gesture Recognition", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference(source=0)
