
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import pyttsx3

# === Load trained LSTM model ===
model = tf.keras.models.load_model("models/lstm_gesture_model.h5")

# === Label map: update as per your classes ===
label_map = {0: 'GoodMorning', 1: 'ThankYouDoctor', 2: 'ILoveYou', 3: 'IAmHungry'}

# === Initialize MediaPipe Holistic ===
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

# === Initialize Text-to-Speech engine ===
engine = pyttsx3.init()
spoken_text = ""

# === Function to extract keypoints ===
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

# === Initialize webcam and sequence buffer ===
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=30)
predicted_text = ""

print("📸 Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB and process with MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    # Extract keypoints and add to sequence
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    # Predict if sequence is full
    if len(sequence) == 30:
        input_seq = np.expand_dims(sequence, axis=0)  # shape: (1, 30, 258)
        prediction = model.predict(input_seq)[0]
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]

        if confidence > 0.8:
            predicted_text = label_map[pred_class]

            # Speak only if changed
            if predicted_text != spoken_text:
                print(f"🗣 Speaking: {predicted_text}")
                engine.say(predicted_text)
                engine.runAndWait()
                spoken_text = predicted_text

    # Display the prediction on frame
    cv2.putText(frame, f"{predicted_text}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Draw pose and hand landmarks
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Show the output
    cv2.imshow("SignSpeakPro - Live Prediction", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Clean up ===
cap.release()
cv2.destroyAllWindows()
print("🛑 Prediction stopped.")
