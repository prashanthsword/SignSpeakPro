 
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths
SEQUENCES_PATH = "sequences"
MODEL_PATH = "models/lstm_gesture_model.h5"
os.makedirs("models", exist_ok=True)

# Params
MAX_SEQ_LEN = 30  # Fixed sequence length (in frames)

# Load sequences
X = []
y = []
label_map = {}

print("📥 Loading and padding sequences...")

for i, gesture in enumerate(os.listdir(SEQUENCES_PATH)):
    label_map[gesture] = i
    gesture_folder = os.path.join(SEQUENCES_PATH, gesture)
    for file in os.listdir(gesture_folder):
        seq = np.load(os.path.join(gesture_folder, file))
        
        # Pad or trim to fixed length
        if len(seq) < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - len(seq), seq.shape[1]))
            seq = np.concatenate((seq, pad))
        elif len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        
        X.append(seq)
        y.append(i)

X = np.array(X)
y = to_categorical(y)

print(f"✅ Loaded {len(X)} samples with shape: {X.shape}")
print(f"🎯 Classes: {label_map}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Model definition
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_SEQ_LEN, X.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, callbacks=[checkpoint])

print(f"\n✅ Model saved to: {MODEL_PATH}")
print(f"🧠 Label map: {label_map}")
