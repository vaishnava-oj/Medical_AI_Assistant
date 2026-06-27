import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/disease_prediction_model.pkl")

# Load dataset to get symptom names
df = pd.read_csv("dataset/archive (1)/Training.csv")

# Remove unwanted column
if "Unnamed: 133" in df.columns:
    df = df.drop("Unnamed: 133", axis=1)

# Get symptom columns
symptoms = df.drop("prognosis", axis=1).columns

# Create empty input
input_data = [0] * len(symptoms)

# Example symptoms
selected_symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions"
]

# Mark selected symptoms
for symptom in selected_symptoms:
    if symptom in symptoms:
        index = list(symptoms).index(symptom)
        input_data[index] = 1

# Predict disease
prediction = model.predict([input_data])

print("Predicted Disease:", prediction[0])