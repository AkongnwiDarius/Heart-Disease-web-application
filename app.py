import pickle

import numpy as np
import streamlit as st
from tensorflow import keras

# page configuration
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

model_path = "Heart_Disease_ann.keras"
kit_path = "Heart_Disease_inference_kit.pkl"

#Loading our model
@st.cache_resource
def load_model_and_kit():
    model = keras.models.load_model(model_path)
    with open(kit_path, "rb") as f:
        kit = pickle.load(f)
    return model, kit

model, kit = load_model_and_kit()

#loading our scaler mean and scaler scale for new patient data 
scaler_mean = np.array(kit["scaler_mean"], dtype=np.float32)
scaler_scale = np.array(kit["scaler_scale"], dtype=np.float32)
feature_cols = kit["feature_cols"]

level_smoking = {'Never':2, 'Current':1, 'Former':0}
level_physical = {'Sedentary':2,'Active':1,'Moderate':0}
level_stress = {'Medium':2,'High':1,'Low':0}
level_alcohol = {'Low':0,'Moderate':1,'High':2}

# Level_smoke = {v: k for k, v in level_smoking.items()}
# Level_activity = {v: k for k, v in level_physical.items()}
# Level_st = {v: k for k, v in level_stress.items()}
# Level_alcohol_intake = {v: k for k, v in level_alcohol.items()}


#Building the user interface
st.title("Heart Disease Prediction System")
st.caption(
    "Enter the information below and get a prediction."
)


#User Inputs
with st.form('Patient_form'):
    st.subheader('User Information')
    age =st.number_input('Age', min_value=1, max_value=120, value=int(1))
    gender = st.selectbox('Gender', options=['Male', 'Female'], index=0)
    bmi = st.number_input('BMI', min_value=5.0, max_value=100.0, value=float(5))

    st.subheader('Lifestyle')
    col3, col4 = st.columns(2)
    with col3:
        smoking = st.selectbox('Smoking Level', options=['Never','Current','Former'], index=0)
        alcohol = st.selectbox("Alcohol Intake", options=["Low", "Moderate", "High"], index=0)
    with col4:
        activity = st.selectbox("Physical Activity", options=['Sedentary','Active','Moderate'], index=0)
        stress = st.selectbox('Stress Level', options=['Low','Medium','High'], index=0)
    st.subheader('Medical Information')
    hypertension = st.selectbox('Hypertension', options=['Yes','No'])
    diabetes = st.selectbox('Diabetes', options=['Yes','No'])
    hyperlipidemia = st.selectbox('Hyperlipidemia', options=['Yes','No'])
    family = st.checkbox('Family History of Heart Attack')
    previous = st.checkbox('Previous Heart Attack')
    col5,col6 = st.columns(2)
    with col5:
        systolic = st.number_input('Sytolic BP (mmHg)', min_value=60, max_value=260, value=int(60))
        diastolic = st.number_input('Diastolic BP (mmHg)', min_value=30, max_value=160, value=int(30))
    with col6:
        heart = st.number_input("Heart rate (bpm)", min_value=30, max_value=220, value=int(30))
        blood= st.number_input("Fasting blood sugar (mg/dL)", min_value=40, max_value=500, value=int(40))

    cholesterol = st.number_input("Total cholesterol (mg/dL)", min_value=80, max_value=500, value=int(80))

    submitted = st.form_submit_button("Predict", use_container_width=True)


#Prediction
if submitted:
    raw = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "BMI": bmi,
        "Smoking": level_smoking[smoking],
        "Alcohol_Intake": level_alcohol[alcohol],
        "Physical_Activity": level_physical[activity],
        "Stress_Level": level_stress[stress],
        "Hypertension": 1 if hypertension == "Yes" else 0,
        "Diabetes": 1 if diabetes == "Yes" else 0,
        "Hyperlipidemia": 1 if hyperlipidemia == "Yes" else 0,
        "Family_History": int(family),
        "Previous_Heart_Attack": int(previous),
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Heart_Rate": heart,
        "Blood_Sugar_Fasting": blood,
        "Cholesterol_Total": cholesterol,
    }
     # Build the feature vector in the exact order the model was trained on
    x = np.array([[raw[col] for col in feature_cols]], dtype=np.float32)
    
        # Apply the SAME scaling used during training
    x_scaled = (x - scaler_mean) / scaler_scale
    
    prob = float(model.predict(x_scaled, verbose=0).flatten()[0])
    pred_label = "Heart Disease" if prob >= 0.5 else "No Heart Disease"
    
    st.divider()
    st.subheader("Result")
    
    if prob >= 0.5:
        st.error(f"⚠️ Prediction: **{pred_label}**")
    else:
        st.success(f"✅ Prediction: **{pred_label}**")
    
    st.metric("Predicted probability of heart disease", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))
    st.warning('⚠️ This prediction is just for educational and demonstration purposes' \
' and should not substitute for medical diagnosis or treatment')
    
    with st.expander("See input values sent to the model"):
        st.json(raw)
    