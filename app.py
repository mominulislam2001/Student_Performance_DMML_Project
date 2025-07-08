import streamlit as st
import pickle
import numpy as np
import json
from joblib import load

# Load mappings from JSON
with open('mappings.json', 'r') as f:
    mappings = json.load(f)

gender_map = mappings['gender_map']
income_map = mappings['income_map']
hometown_map = mappings['hometown_map']
preparation_map = mappings['preparation_map']
gaming_map = mappings['gaming_map']
attendance_map = mappings['attendance_map']
job_map = mappings['job_map']
extra_map = mappings['extra_map']

# Load model
model = load('saved_models/RandomForestClassifier_model.pkl')

# Streamlit UI
st.title("🎓 Student CGPA Class Prediction")

st.header("Input Student Information:")

gender = st.selectbox("Gender", list(gender_map.keys()))
hsc = st.number_input("HSC GPA (e.g., 4.50)", min_value=0.0, max_value=5.0, step=0.01)
ssc = st.number_input("SSC GPA (e.g., 5.00)", min_value=0.0, max_value=5.0, step=0.01)
income = st.selectbox("Family Income", list(income_map.keys()))
hometown = st.selectbox("Hometown", list(hometown_map.keys()))
computer = st.slider("Computer Skill Level (1–5)", min_value=1, max_value=5, step=1)
preparation = st.selectbox("Daily Study Preparation", list(preparation_map.keys()))
gaming = st.selectbox("Daily Gaming Time", list(gaming_map.keys()))
attendance = st.selectbox("Class Attendance", list(attendance_map.keys()))
job = st.selectbox("Part-time Job?", list(job_map.keys()))
english = st.slider("English Proficiency (1–5)", min_value=1, max_value=5, step=1)
extra = st.selectbox("Participates in Extra Curricular?", list(extra_map.keys()))



if st.button("Predict CGPA Class"):
    features = [
        gender_map[gender],
        hsc,
        ssc,
        income_map[income],
        hometown_map[hometown],
        computer,
        preparation_map[preparation],
        gaming_map[gaming],
        attendance_map[attendance],
        job_map[job],
        english,
        extra_map[extra]
    ]
    prediction = model.predict([features])[0]
    class_map = {
    0: ("Low", "≤ 2.75"),
    1: ("Medium", "2.76 – 3.25"),
    2: ("High", "> 3.25")
    }

    label, cgpa_range = class_map[prediction]
    st.success(f"**Predicted CGPA Class:** {label} with **Approx CGPA Range:** {cgpa_range}")