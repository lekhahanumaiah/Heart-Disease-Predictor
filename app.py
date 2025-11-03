# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load trained model and scaler
# with open("model.pkl", "rb") as f:
#     model = joblib.load(f)

# scaler = joblib.load("scaler.pkl")

# # App title
# st.title("🫀 Heart Disease Prediction App")

# # User input fields
# age = st.number_input("Age", min_value=1, max_value=120, value=50)
# sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
# cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
# trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
# chol = st.number_input("Serum Cholestoral in mg/dl (chol)", min_value=100, max_value=600, value=200)
# fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
# restecg = st.selectbox("Resting ECG results (restecg)", options=[0, 1, 2])
# thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
# exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
# oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
# slope = st.selectbox("Slope of the peak exercise ST segment (slope)", options=[0, 1, 2])
# ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3])
# thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# # Add Predict button
# if st.button("🔍 Predict"):

#     # Input dictionary
#     input_dict = {
#         "age": age,
#         "sex": sex,
#         "cp": cp,
#         "trestbps": trestbps,
#         "chol": chol,
#         "fbs": fbs,
#         "restecg": restecg,
#         "thalach": thalach,
#         "exang": exang,
#         "oldpeak": oldpeak,
#         "slope": slope,
#         "ca": ca,
#         "thal": thal
#     }

#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_dict])

#     # Scale the input features
#     input_scaled = scaler.transform(input_df)

#     # Predict using the model
#     prediction = model.predict(input_scaled)[0]

#     # For probability
#     try:
#         probability = model.predict_proba(input_scaled)[0][1]
#     except AttributeError:
#         decision = model.decision_function(input_scaled)[0]
#         probability = 1 / (1 + np.exp(-decision))  # Sigmoid approximation

#     # Show result
#     if prediction == 1:
#         st.error(f"⚠️ High risk: Patient likely has heart disease. (Confidence: {probability:.2f})")
#     else:
#         st.success(f"✅ Low risk: Patient unlikely to have heart disease. (Confidence: {probability:.2f})")
import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the saved model & scaler
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

st.title("🫀 Heart Disease Prediction App")

st.markdown("""
Enter patient medical details below to predict the **risk of heart disease**.
""")

# -----------------------------
# Collect input features
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 52)
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1 = Normal, 2 = Fixed, 3 = Reversible)", [1, 2, 3])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    # Arrange features as in training
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])
    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High risk: Patient likely has heart disease.")
    else:
        st.success("✅ Low risk: Patient unlikely to have heart disease.")