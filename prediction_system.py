import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Load the feature names
feature_names = joblib.load("feature_names.pkl")

print("Model loaded successfully!")
print("Features:", feature_names)


# Function to determine risk level
def get_risk_level(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.70:
        return "Medium Risk"

    else:
        return "High Risk"


# Function to provide maintenance recommendation
def get_maintenance_recommendation(risk_level):

    if risk_level == "Low Risk":
        return "Continue normal operation and monitor machine parameters."

    elif risk_level == "Medium Risk":
        return "Schedule inspection and monitor temperature, torque and tool wear."

    else:
        return "Perform maintenance inspection immediately and consider stopping the machine."


# Example machine input
machine_data = {
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 50.0,
    "Tool wear [min]": 100,
    "Type_H": 0,
    "Type_L": 0,
    "Type_M": 1
}

# Convert input into DataFrame
input_data = pd.DataFrame([machine_data])

# Make sure feature order is exactly the same as during training
input_data = input_data[feature_names]

# Predict failure probability
failure_probability = model.predict_proba(input_data)[0][1]

# Determine risk
risk_level = get_risk_level(failure_probability)

# Get recommendation
recommendation = get_maintenance_recommendation(risk_level)

print("\n--- Machine Prediction ---")
print("Failure Probability:", round(failure_probability * 100, 2), "%")
print("Risk Level:", risk_level)
print("Maintenance Recommendation:", recommendation)