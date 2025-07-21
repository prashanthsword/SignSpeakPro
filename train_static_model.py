import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
 
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# === Paths ===
DATA_PATH = "images"
MODEL_PATH = "models/static_gesture_model.pkl"

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# === Helper: Extract landmarks from image ===
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        keypoints = []
        for handLms in result.multi_hand_landmarks:
            for lm in handLms.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        # Pad with zeros if only 1 hand
        while len(keypoints) < 126:
            keypoints.extend([0.0, 0.0, 0.0])
        return keypoints
    return None

# === Load data ===
X, y = [], []
labels = []
print("📁 Loading gesture images...")

for gesture_label in os.listdir(DATA_PATH):
    gesture_path = os.path.join(DATA_PATH, gesture_label)
    if not os.path.isdir(gesture_path): continue
    labels.append(gesture_label)

    for img_file in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_file)
        image = cv2.imread(img_path)
        if image is None: continue

        landmarks = extract_landmarks(image)
        if landmarks:
            X.append(landmarks)
            y.append(gesture_label)

print(f"✅ Loaded {len(X)} samples across {len(set(y))} gesture classes.")

# === Train model ===
print("🧠 Training static gesture model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc:.2f}")

# === Save model ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Static model saved to {MODEL_PATH}")
