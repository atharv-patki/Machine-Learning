import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("student_score_model.pkl")

# Function to predict student score based on study hours

def predict_score(hours):
    hours_df = pd.DataFrame({'Hours': [hours]})  # Convert to DataFrame with column name
    prediction = model.predict(hours_df)
    return round(prediction[0], 2)

# Take user input for study hours
try:
    hours = float(input("📚 Enter study hours: "))
    predicted_score = predict_score(hours)
    print(f"Predicted Score: {predicted_score}")
except ValueError:
    print("Invalid input. Please enter a numeric value for hours.")
