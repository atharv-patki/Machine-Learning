import numpy as np
import pandas as pd
import joblib

model = joblib.load("Salary_Data.pkl")

years = float(input("Enter years of experience: "))

experience = pd.DataFrame([[years]], columns=["YearsExperience"])

predicted_salary = model.predict(experience)

print(f"Predicted Salary: ${round(predicted_salary[0], 2)}")
