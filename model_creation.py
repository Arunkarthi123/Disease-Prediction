import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Convert to 1D array

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "disease_prediction_model.pkl")

print("✅ Model trained and saved successfully.")

