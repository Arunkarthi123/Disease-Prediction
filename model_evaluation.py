import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load model and test data
model = joblib.load("disease_prediction_model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

