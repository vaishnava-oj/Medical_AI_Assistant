import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/disease_prediction_model.pkl")

# Load dataset
df = pd.read_csv("dataset/archive (1)/Training.csv")

# Remove unwanted column
if "Unnamed: 133" in df.columns:
    df = df.drop("Unnamed: 133", axis=1)

# Get symptom names
symptoms = df.drop("prognosis", axis=1).columns

# Create input vector
input_data = [0] * len(symptoms)

# Example symptoms
selected_symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions"
]

for symptom in selected_symptoms:
    if symptom in symptoms:
        index = list(symptoms).index(symptom)
        input_data[index] = 1

# Predict
prediction = model.predict([input_data])

print("Predicted Disease:", prediction[0])