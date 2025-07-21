import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=only ERROR

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import csv
import os

# MediaPipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Input
label = input("Enter the sign label (A-Z): ").upper()

# Create folders
data_dir = "data"
img_dir = f"images/{label}"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

csv_file = os.path.join(data_dir, f"{label}.csv")

# Webcam
cap = cv2.VideoCapture(0)
print("📸 Press 's' to save sample, 'q' to quit...")

sample_count = 0

with open(csv_file, mode='a', newline='') as f:
    csv_writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Sign Capture", frame)
        key = cv2.waitKey(1)

        if key == ord('s') and results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Save landmarks
                row = []
                for lm in handLms.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                csv_writer.writerow(row)

                # Save image
                img_path = os.path.join(img_dir, f"{label}_{sample_count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"✅ Saved sample {sample_count} for '{label}'")
                sample_count += 1

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Collection complete for '{label}' — {sample_count} samples saved.")
