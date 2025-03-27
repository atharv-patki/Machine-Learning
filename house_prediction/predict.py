import joblib
import pandas as pd

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Function to predict house price
def predict_price(area, bedrooms):
    # Convert input into a DataFrame with proper column names
    input_data = pd.DataFrame([[area, bedrooms]], columns=["Area", "Bedrooms"])

    # Make prediction
    prediction = model.predict(input_data)

    # Extract single value safely
    return round(prediction.item(), 2)

# Take user input for prediction
area = float(input("Enter house area (in sqft): "))
bedrooms = int(input("Enter number of bedrooms: "))

# Get predicted price
predicted_price = predict_price(area, bedrooms)

print(f"Predicted House Price: ${predicted_price}")
