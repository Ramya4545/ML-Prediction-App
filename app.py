


import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# 1. Load Model and Encoders
# --------------------------------------------------

model = joblib.load("titanic_model.pkl")

sex_encoder = joblib.load("sex_encoder.pkl")

embarked_encoder = joblib.load("embarked_encoder.pkl")


# --------------------------------------------------
# 2. Streamlit App
# --------------------------------------------------

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival.")


# --------------------------------------------------
# 3. User Input
# --------------------------------------------------

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    sex_encoder.classes_
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.selectbox(
    "Embarked",
    embarked_encoder.classes_
)


# --------------------------------------------------
# 4. Prediction
# --------------------------------------------------

if st.button("Predict Survival"):

    # Convert Sex into number
    sex_encoded = sex_encoder.transform([sex])[0]

    # Convert Embarked into number
    embarked_encoded = embarked_encoder.transform([embarked])[0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_encoded],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked_encoded]
    })

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("🎉 Passenger is predicted to SURVIVE.")

    else:
        st.error("❌ Passenger is predicted NOT to survive.")
