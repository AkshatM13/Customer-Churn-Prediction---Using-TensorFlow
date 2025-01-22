import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# Load and preprocess data
@st.cache_data
def load_data():
    data_path = "Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data = pd.read_csv(data_path)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(subset=['TotalCharges'], inplace=True)
    data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
    data = data.drop(['customerID'], axis=1)

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    return data

data = load_data()

# Split data
X = data.drop('Churn_Yes', axis=1)
y = data['Churn_Yes']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the model
def build_model():
    model = models.Sequential([
        layers.Dense(19, input_shape=(X_train.shape[1],), activation='relu'),
        layers.Dense(15, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# Streamlit app
st.title("Customer Churn Prediction")

# User input form
st.sidebar.header("Input Customer Details")
def user_input_features():
    features = {}
    for col in X.columns:
        if data[col].max() <= 1:  # Binary or normalized column
            features[col] = st.sidebar.slider(col, 0.0, 1.0, 0.5)
        else:
            features[col] = st.sidebar.number_input(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Preprocess user input
input_scaled = scaler.transform(input_df)

# Make prediction
prediction_proba = model.predict(input_scaled)[0][0]
prediction = "Yes" if prediction_proba > 0.5 else "No"

# Display prediction
st.subheader("Prediction")
st.write(f"The customer will churn: **{prediction}**")

st.subheader("Prediction Probability")
st.write(f"Probability of churn: **{prediction_proba:.2f}**")

# Developer details
st.sidebar.header("Developer Details")
st.sidebar.write("**Name:** Akshat Maurya")
st.sidebar.write("[GitHub](https://github.com/akshatm13)")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/makshat13)")

# Display confusion matrix and classification report
st.subheader("Model Performance on Test Set")
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_classes)
cr = classification_report(y_test, y_pred_classes)

st.write("Confusion Matrix")
st.write(pd.DataFrame(cm, columns=['No Churn', 'Churn'], index=['No Churn', 'Churn']))

st.write("Classification Report")
st.text(cr)
