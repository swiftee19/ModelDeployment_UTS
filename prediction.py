import streamlit as st
import pandas as pd
import pickle

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

# Load the OneHotEncoder
with open("oneHotEncoder.pkl", 'rb') as file:
    encoder = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)


def preprocess_input(data):
    # Preprocess categorical variables
    categorical_columns = ['Geography']

    data.replace({"Gender": {"Male": 1, "Female": 0}}, inplace=True)
    data.drop()

    encoded_columns = pd.DataFrame(encoder.transform(data[categorical_columns]))
    encoded_columns.columns = encoder.get_feature_names_out(categorical_columns)
    data.drop(categorical_columns, axis=1, inplace=True)
    data = pd.concat([data, encoded_columns], axis=1)

    # Preprocess numerical variables
    data = scaler.transform(data)

    return data


def predict(data):
    data = preprocess_input(data)

    prediction = model.predict(data)

    return prediction


def main():
    st.title("Churn Prediction")

    credit_score = st.number_input("Enter Credit Score:")
    age = st.number_input("Enter Age:")
    tenure = st.number_input("Enter Tenure:")
    balance = st.number_input("Enter Balance:")
    num_of_products = st.number_input("Enter Number of Products:")
    has_credit_card = st.selectbox("Has Credit Card:", [0, 1])
    is_active_member = st.selectbox("Is Active Member:", [0, 1])
    estimated_salary = st.number_input("Enter Estimated Salary:")
    geography = st.selectbox("Geography:", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Gender:", ['Male', 'Female'])

    user_input = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography],
        'Gender': [gender]
    })

    if st.button("Predict Churn"):
        prediction = predict(user_input)
        st.write("Churn Prediction:", prediction[0])


if __name__ == "__main__":
    main()
