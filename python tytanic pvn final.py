import streamlit as st
import joblib
import numpy as np


model = joblib.load("rf_titanic.pkl")


st.title("Titanic Survival Prediction App")

st.write("Enter passenger details:")


pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100)
sibsp = st.number_input("Siblings/Spouses", min_value=0)
parch = st.number_input("Parents/Children", min_value=0)
fare = st.number_input("Fare", min_value=0.0)

sex = st.selectbox("Sex", ["Male", "Female"])
embarked = st.selectbox("Embarked", ["Q", "S"])


sex_male = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

if st.button("Predict"):

    input_data = np.array([[
        pclass,
        age,
        sibsp,
        parch,
        fare,
        sex_male,
        embarked_Q,
        embarked_S
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
        