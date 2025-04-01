import cv2
import numpy as np
import torch
from collections import deque
import time
from ultralytics import YOLO
import mediapipe as mp
from app.training_finetune import SignBERT

# Initialize models and settings (same as before)
yolo_weights = "app/models/best.pt"  # Use a relative path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(yolo_weights).to(device)

model_signbert = SignBERT(input_dim=63, d_model=256, num_layers=3, num_heads=8, dropout=0.1, num_classes=40)
model_signbert.load_state_dict(torch.load("app/models/signbert_finetuned_weighted_v4.pth", map_location=device))
model_signbert.to(device)
model_signbert.eval()

input_dim = model_signbert.token_embedding.in_features

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2)

def extract_pose_keypoints(image):
    results = hands.process(image)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        keypoints = np.array(keypoints, dtype=np.float32)
        if keypoints.shape[0] != input_dim:
            if keypoints.shape[0] < input_dim:
                keypoints = np.pad(keypoints, (0, input_dim - keypoints.shape[0]), mode='constant')
            else:
                keypoints = keypoints[:input_dim]
        return keypoints
    else:
        return np.zeros(input_dim, dtype=np.float32)

def process_frame(frame):
    # Use YOLOv8 to detect hand and extract keypoints (same as before)
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
    
    margin_x = int(0.2 * w)
    margin_y = int(0.2 * h)
    left_expanded = max(0, left - margin_x)
    top_expanded = max(0, top - margin_y)
    right_expanded = min(frame.shape[1], right + margin_x)
    bottom_expanded = min(frame.shape[0], bottom + margin_y)
    
    cv2.rectangle(frame, (left_expanded, top_expanded), (right_expanded, bottom_expanded), (0, 255, 0), 2)
    
    cropped = frame[top_expanded:bottom_expanded, left_expanded:right_expanded]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_resized = cv2.resize(cropped_rgb, (224, 224))
    
    keypoints = extract_pose_keypoints(cropped_resized)
    return keypoints, frame

def run_inference(file_bytes):
    # Convert the incoming bytes to a NumPy array
    np_arr = np.frombuffer(file_bytes, np.uint8)
    # Decode the image from the numpy array
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "Error: Could not decode image"

    # Process the frame to extract keypoints
    keypoints, frame_vis = process_frame(frame)
    if keypoints is None:
        return "No hand detected"

    # For demonstration, let's assume we are processing just one frame:
    # Prepare the keypoints for the model
    seq = np.expand_dims(keypoints, axis=0)  # Shape: (1, input_dim)
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, input_dim)

    with torch.no_grad():
        logits = model_signbert(seq_tensor, mode='finetune')
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return predicted_class
