 
import cv2
import mediapipe as mp
import os
import csv
import time

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Get gesture name
gesture = input("Enter the gesture name (e.g., Hello, ThankYou): ").strip().replace(" ", "_")

# Folder setup
data_dir = "data"
img_dir = f"images/{gesture}"
csv_file = f"{data_dir}/{gesture}.csv"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

print(f"\n📸 Collecting data for gesture: '{gesture}'")
print("✋ One hand: press 's' to save")
print("🤲 Two hands: auto-saving every 2 seconds (up to 8 sec)")
print("➡️ Press 'q' to quit at any time\n")

# Camera setup
cap = cv2.VideoCapture(0)
sample_count = 0
start_time = None
auto_capture_started = False

with open(csv_file, mode='a', newline='') as f:
    csv_writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Show number of hands detected
        cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow(f"Collecting Gesture: {gesture}", frame)
        key = cv2.waitKey(1)

        current_time = time.time()

        if hand_count == 2:
            if not auto_capture_started:
                start_time = current_time
                auto_capture_started = True
                print("🕒 Two hands detected. Auto-saving starts now!")

            # Capture every 2 sec between 2s and 8s
            elapsed = current_time - start_time
            if 2 <= elapsed <= 8 and int((elapsed * 10) % 20) == 0:
                full_row = []
                for handLms in results.multi_hand_landmarks:
                    for lm in handLms.landmark:
                        full_row.extend([lm.x, lm.y, lm.z])
                while len(full_row) < 126:
                    full_row.extend([0.0, 0.0, 0.0])

                csv_writer.writerow(full_row)
                img_path = os.path.join(img_dir, f"{gesture}_{sample_count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"✅ Auto-saved sample {sample_count} for '{gesture}'")
                sample_count += 1

            if elapsed > 8:
                auto_capture_started = False  # Reset to allow next cycle

        elif hand_count == 1 and key == ord('s'):
            row = []
            for handLms in results.multi_hand_landmarks:
                for lm in handLms.landmark:
                    row.extend([lm.x, lm.y, lm.z])
            while len(row) < 126:
                row.extend([0.0, 0.0, 0.0])

            csv_writer.writerow(row)
            img_path = os.path.join(img_dir, f"{gesture}_{sample_count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"✅ Saved sample {sample_count} for '{gesture}'")
            sample_count += 1

        if key == ord('q'):
            print(f"\n✅ Collection complete for '{gesture}' — {sample_count} samples saved.")
            break

cap.release()
cv2.destroyAllWindows()
