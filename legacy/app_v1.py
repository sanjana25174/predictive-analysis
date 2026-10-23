import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("random_forest_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("⚙️ AI-Based Smart Manufacturing & Predictive Maintenance")
st.write(
    "Enter the current machine sensor values to predict machine-failure risk "
    "and receive a maintenance recommendation."
)


# -----------------------------
# Input section
# -----------------------------
st.header("Machine Sensor Information")

col1, col2 = st.columns(2)

with col1:
    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=295.0,
        max_value=305.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=305.0,
        max_value=314.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10
    )

with col2:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=80.0,
        value=50.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )


# -----------------------------
# Prediction button
# -----------------------------
if st.button("🔍 Predict Machine Failure", use_container_width=True):

    # Encode machine type
    type_h = 1 if machine_type == "H" else 0
    type_l = 1 if machine_type == "L" else 0
    type_m = 1 if machine_type == "M" else 0

    # Create input data
    machine_data = {
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": type_h,
        "Type_L": type_l,
        "Type_M": type_m
    }

    input_data = pd.DataFrame([machine_data])

    # Make sure feature order matches training
    input_data = input_data[feature_names]

    # Predict probability
    failure_probability = model.predict_proba(input_data)[0][1]

    probability_percentage = failure_probability * 100

    # -----------------------------
    # Determine risk level
    # -----------------------------
    if failure_probability < 0.30:
        risk_level = "Low Risk"
        recommendation = (
            "Continue normal operation and monitor machine parameters."
        )

    elif failure_probability < 0.70:
        risk_level = "Medium Risk"
        recommendation = (
            "Schedule inspection and closely monitor temperature, "
            "torque and tool wear."
        )

    else:
        risk_level = "High Risk"
        recommendation = (
            "Perform maintenance inspection immediately and consider "
            "stopping the machine."
        )

    # -----------------------------
    # Display results
    # -----------------------------
    st.header("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Failure Probability",
            f"{probability_percentage:.2f}%"
        )

    with result_col2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with result_col3:
        if failure_probability >= 0.50:
            prediction = "Machine Failure Risk Detected"
        else:
            prediction = "No Immediate Failure Risk"

        st.metric(
            "Prediction",
            prediction
        )

    st.subheader("Maintenance Recommendation")
    st.info(recommendation)

    # Progress bar
    st.subheader("Failure Probability")

    st.progress(
        min(failure_probability, 1.0)
    )