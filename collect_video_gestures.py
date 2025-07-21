
import cv2
import os

gesture = input("🎬 Enter gesture label (e.g., GoodMorning, HelpMe): ").strip().replace(" ", "_")
videos_dir = os.path.join("videos", gesture)
os.makedirs(videos_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print(f"\n🎥 Recording gesture: {gesture}")
print("➡️ Press 'r' to start recording for 3 seconds")
print("➡️ Press 'q' to quit\n")

video_count = 0
recording = False
start_time = None
fps = 20
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if recording:
        elapsed = cv2.getTickCount() - start_time
        elapsed_sec = elapsed / cv2.getTickFrequency()

        if elapsed_sec <= 3:
            out.write(frame)
            cv2.putText(frame, f"⏺ Recording... {elapsed_sec:.1f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            recording = False
            out.release()
            print(f"✅ Saved video {video_count} for '{gesture}'")
            video_count += 1

    cv2.imshow("Video Collector", frame)
    key = cv2.waitKey(1)

    if key == ord('r'):
        video_path = os.path.join(videos_dir, f"{gesture}_{video_count}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (frame_width, frame_height))
        recording = True
        start_time = cv2.getTickCount()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
