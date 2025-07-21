
import os
import cv2
import numpy as np
import mediapipe as mp

# Paths
VIDEOS_PATH = "videos"
SEQUENCES_PATH = "sequences"
os.makedirs(SEQUENCES_PATH, exist_ok=True)

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# Iterate through videos
for gesture in os.listdir(VIDEOS_PATH):
    gesture_dir = os.path.join(VIDEOS_PATH, gesture)
    for video_file in os.listdir(gesture_dir):
        sequence = []
        cap = cv2.VideoCapture(os.path.join(gesture_dir, video_file))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
        cap.release()
        sequence = np.array(sequence)
        
        save_dir = os.path.join(SEQUENCES_PATH, gesture)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{os.path.splitext(video_file)[0]}.npy")
        np.save(save_path, sequence)
        print(f"✅ Saved sequence: {save_path}")
