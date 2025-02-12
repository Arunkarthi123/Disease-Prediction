import numpy as np
import joblib
import pandas as pd

# Load trained model
model = joblib.load("disease_prediction_model.pkl")

# Load training data to get symptom feature names
train_data = pd.read_csv("/content/training_data.csv")
train_data = train_data.drop(columns=['Unnamed: 133'], errors='ignore')
symptom_columns = list(train_data.columns[:-1])

# Function to convert user input into feature vector
def get_symptom_vector(user_symptoms):
    input_vector = [0] * len(symptom_columns)
    for symptom in user_symptoms:
        if symptom in symptom_columns:
            input_vector[symptom_columns.index(symptom)] = 1
    return np.array(input_vector).reshape(1, -1)

# Function to predict top related diseases
def predict_disease(user_symptoms, top_n=3):
    input_data = get_symptom_vector(user_symptoms)
    probabilities = model.predict_proba(input_data)[0]
    disease_classes = model.classes_
    sorted_indices = np.argsort(probabilities)[::-1][:top_n]
    return [(disease_classes[i], round(probabilities[i] * 100, 2)) for i in sorted_indices]

# User input
user_input = input("Enter symptoms separated by commas: ").lower().split(",")
user_symptoms = [symptom.strip() for symptom in user_input]

# Predict diseases
predicted_diseases = predict_disease(user_symptoms)

# Display results
print("\nPredicted Diseases and Probabilities:")
for disease, prob in predicted_diseases:
    print(f"🔹 {disease}: {prob}%")

