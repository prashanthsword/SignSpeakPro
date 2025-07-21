
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
import pyttsx3
import time

# Initialize TTS engine once globally
engine = pyttsx3.init()

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        # Sometimes pyttsx3 throws runtime error on multiple calls; ignore
        pass

# Load models (adjust paths if needed)
@st.cache_resource
def load_models():
    static_model = pickle.load(open("models/static_gesture_model.pkl", "rb"))
    sequence_model = tf.keras.models.load_model("models/lstm_gesture_model.h5")
    return static_model, sequence_model

static_model, sequence_model = load_models()

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Labels - replace or extend as per your training
static_labels = ['A', 'B', 'C', 'Hello', 'Bye', 'Thanks']  
sequence_labels = ['Namaste', 'Goodmorning', 'ILoveYou']  

st.title("🧠 SignSpeakPro: AI Sign Language Translator")

selected_mode = st.radio("Choose Mode", ["📷 Static Gesture", "🎥 Dynamic Gesture"])

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_static' not in st.session_state:
    st.session_state.last_static = 0
if 'last_dynamic' not in st.session_state:
    st.session_state.last_dynamic = 0
if 'sequence' not in st.session_state:
    st.session_state.sequence = []

def start_camera():
    st.session_state.running = True

def stop_camera():
    st.session_state.running = False
    st.session_state.sequence = []

if not st.session_state.running:
    st.button("▶️ Start Camera", on_click=start_camera, key="start_cam")
else:
    st.button("🛑 Stop Camera", on_click=stop_camera, key="stop_cam")

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

def extract_landmarks(results):
    hand_landmarks = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])
    # Pad if only one hand
    while len(hand_landmarks) < 126:
        hand_landmarks.extend([0.0, 0.0, 0.0])
    return np.array(hand_landmarks).reshape(1, -1)

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Unable to access the camera.")
        st.session_state.running = False
        cap.release()
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(results)

            current_time = time.time()

            if selected_mode == "📷 Static Gesture":
                if current_time - st.session_state.last_static > 5:  # 5 seconds delay
                    prediction_raw = static_model.predict(landmarks)[0]
                    prediction = static_labels[np.argmax(prediction_raw)]
                    prediction_placeholder.subheader(f"🧠 Static Prediction: **{prediction}**")
                    speak(prediction)
                    st.session_state.last_static = current_time

            elif selected_mode == "🎥 Dynamic Gesture":
                st.session_state.sequence.append(landmarks[0])
                # Keep last 30 frames
                st.session_state.sequence = st.session_state.sequence[-30:]

                if len(st.session_state.sequence) == 30 and current_time - st.session_state.last_dynamic > 10:  # 10 seconds delay
                    input_seq = np.expand_dims(st.session_state.sequence, axis=0).astype(np.float32)
                    res = sequence_model.predict(input_seq)[0]
                    prediction = sequence_labels[np.argmax(res)]
                    prediction_placeholder.subheader(f"🎬 Dynamic Prediction: **{prediction}**")
                    speak(prediction)
                    st.session_state.last_dynamic = current_time
                    st.session_state.sequence = []

        else:
            prediction_placeholder.info("✋ Waiting for hand...")

        frame_placeholder.image(frame, channels="BGR")
        cap.release()
else:
    st.info("🛑 Camera stopped. Click Start Camera to begin.")
