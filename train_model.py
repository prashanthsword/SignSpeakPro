
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Folders
data_dir = "data"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

X = []
y = []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        path = os.path.join(data_dir, file)

        if os.path.getsize(path) == 0:
            print(f"⚠️ Skipping empty file: {file}")
            continue

        try:
            data = pd.read_csv(path, header=None)
            if data.empty:
                print(f"⚠️ No data in {file}, skipping.")
                continue

            for row in data.values:
                row = row.tolist()
                # Pad to 126 (for both hands); if already 126, leave it; if less, pad with 0s
                if len(row) < 126:
                    row.extend([0.0] * (126 - len(row)))
                elif len(row) > 126:
                    row = row[:126]  # trim if over
                X.append(row)
                y.append(label)

            print(f"✅ Loaded: {label} ({len(data)} padded samples)")

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

# Final check
if not X:
    print("🚫 No valid data found.")
    exit()

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
print("\n🧠 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")

# Save
model_path = os.path.join(model_dir, "static_gesture_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"💾 Model saved to {model_path}")
