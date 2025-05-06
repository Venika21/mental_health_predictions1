# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    df = pd.read_csv("survey.csv")
    df = df[['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere', 'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'treatment']]

    df = df.dropna()
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    X = df.drop("treatment", axis=1)
    y = df["treatment"]
    return X, y, label_encoders
# train_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data

X, y, encoders = load_and_preprocess_data()
model = RandomForestClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
# app.py
import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("Mental Health Prediction App")

# Collect user input
def user_input():
    age = st.slider('Age', 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
    family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Work Interference", ["Never", "Rarely", "Sometimes", "Often"])
    no_employees = st.selectbox("Company Size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Remote Work?", ["Yes", "No"])
    tech_company = st.selectbox("Tech Company?", ["Yes", "No"])
    benefits = st.selectbox("Mental Health Benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Care Options?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness Program?", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Seek Help?", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Anonymity Protected?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of Taking Leave", ["Very difficult", "Somewhat difficult", "Somewhat easy", "Very easy", "Don't know"])
    mental_health_consequence = st.selectbox("Mental Health Consequence?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Physical Health Consequence?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Talk to Coworkers?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Talk to Supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Interview - Mental Health?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Interview - Physical Health?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Mental vs Physical Health", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Observed Consequences?", ["Yes", "No"])

    user_data = {
        'Age': age,
        'Gender': gender,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'no_employees': no_employees,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': benefits,
        'care_options': care_options,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'mental_health_consequence': mental_health_consequence,
        'phys_health_consequence': phys_health_consequence,
        'coworkers': coworkers,
        'supervisor': supervisor,
        'mental_health_interview': mental_health_interview,
        'phys_health_interview': phys_health_interview,
        'mental_vs_physical': mental_vs_physical,
        'obs_consequence': obs_consequence
    }

    return pd.DataFrame([user_data])

df_input = user_input()

# Encode inputs
for col in df_input.columns:
    if col in encoders:
        df_input[col] = encoders[col].transform(df_input[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(df_input)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts a likelihood of needing mental health treatment.")
    else:
        st.success("✅ The model predicts no immediate indication of mental health treatment needed.")

