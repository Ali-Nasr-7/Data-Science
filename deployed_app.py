import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

# Streamlit Page Settings
st.set_page_config(page_title="Machine Failure Predictor", page_icon="🛠️", layout="centered")

st.title("🛠️ Machine Failure Prediction App")
st.write("Enter the machine sensor readings to predict the **failure risk**:")

# Sidebar example inputs
st.sidebar.header("📈 Example Inputs")

air_temp = st.sidebar.number_input('Air Temperature [K]', value=300.0, step=1.0)
process_temp = st.sidebar.number_input('Process Temperature [K]', value=305.0, step=1.0)
rot_speed = st.sidebar.number_input('Rotational Speed [rpm]', value=1500.0, step=10.0)
torque = st.sidebar.number_input('Torque [Nm]', value=50.0, step=1.0)
tool_wear = st.sidebar.number_input('Tool Wear [min]', value=10.0, step=1.0)

type_option = st.sidebar.selectbox('Machine Type', ['L', 'M', 'H'])

# Encode Machine Type
Type_L = 1 if type_option == 'L' else 0
Type_M = 1 if type_option == 'M' else 0
# Type_H is implicit (both L and M are 0)

# Auto-calculate additional feature: Power (Torque x Rotational Speed)
power = (torque * rot_speed) / 9550  # Simplified mechanical power formula

st.sidebar.markdown(f"⚡ **Auto-calculated Power [W]**: `{power:.2f}`")

st.write("---")

# Main Button
if st.button('🚀 Predict Machine Failure'):
    input_features = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, Type_L, Type_M, power,0,0,0,0]])

    # Predict probability
    failure_probability = model.predict_proba(input_features)[0][1]  # Probability of Class 1 (Failure)
    failure_percent = failure_probability * 100

    st.subheader("🔎 Failure Risk Assessment:")

    # Progress Bar
    st.progress(failure_probability)

    # Risk Message
    if failure_percent >= 70:
        st.error(f"🚨 **High Risk!** Machine Failure Probability: `{failure_percent:.2f}%`")
    elif 40 <= failure_percent < 70:
        st.warning(f"⚠️ **Moderate Risk.** Machine Failure Probability: `{failure_percent:.2f}%`")
    else:
        st.success(f"✅ **Low Risk.** Machine Failure Probability: `{failure_percent:.2f}%`")

    # Add note for interpretation
    st.caption("ℹ️ *Note: High risk machines should be scheduled for immediate maintenance to avoid breakdowns.*")

