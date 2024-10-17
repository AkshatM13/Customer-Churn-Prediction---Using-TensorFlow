import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Streamlit app title
st.title("Customer Churn Prediction App")

# Load data
@st.cache_data
def load_data():
    data_path = "D:/MSIS/Customer churn prediction/Customer-Churn-Prediction---Using-TensorFlow/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data = pd.read_csv(data_path)
    return data

data = load_data()
st.write("### Dataset Preview")
st.write(data.head())

# Data Preprocessing
df = data.drop(['customerID'], axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Encode categorical variables
def encode_data(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

df = encode_data(df)

# Store feature names for later use
features = df.drop('Churn', axis=1).columns

# Split the data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
X_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
st.write("### Model Accuracy")
accuracy = model.score(X_test, y_test)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# User Input Form for Prediction
st.write("## Predict Churn for a New Customer")
def user_input():
    data = {
        'tenure': st.number_input("Tenure", min_value=0, max_value=100, value=10),
        'MonthlyCharges': st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0),
        'TotalCharges': st.number_input("Total Charges", min_value=0.0, value=1000.0),
        'SeniorCitizen': 1 if st.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes" else 0,
        'Partner': 1 if st.selectbox("Partner", ["No", "Yes"]) == "Yes" else 0,
        'Dependents': 1 if st.selectbox("Dependents", ["No", "Yes"]) == "Yes" else 0,
        'PhoneService': 1 if st.selectbox("Phone Service", ["No", "Yes"]) == "Yes" else 0,
        'MultipleLines': 1 if st.selectbox("Multiple Lines", ["No", "Yes"]) == "Yes" else 0,
        'InternetService': st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        'OnlineSecurity': 1 if st.selectbox("Online Security", ["No", "Yes"]) == "Yes" else 0,
        'OnlineBackup': 1 if st.selectbox("Online Backup", ["No", "Yes"]) == "Yes" else 0,
        'DeviceProtection': 1 if st.selectbox("Device Protection", ["No", "Yes"]) == "Yes" else 0,
        'TechSupport': 1 if st.selectbox("Tech Support", ["No", "Yes"]) == "Yes" else 0,
        'StreamingTV': 1 if st.selectbox("Streaming TV", ["No", "Yes"]) == "Yes" else 0,
        'StreamingMovies': 1 if st.selectbox("Streaming Movies", ["No", "Yes"]) == "Yes" else 0,
        'Contract': st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        'PaperlessBilling': 1 if st.selectbox("Paperless Billing", ["No", "Yes"]) == "Yes" else 0,
        'PaymentMethod': st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    }

    # Convert categorical fields to numerical
    data['InternetService'] = {"DSL": 0, "Fiber optic": 1, "No": 2}[data['InternetService']]
    data['Contract'] = {"Month-to-month": 0, "One year": 1, "Two year": 2}[data['Contract']]
    data['PaymentMethod'] = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[data['PaymentMethod']]

    # Convert to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    return input_df

input_df = user_input()

# Ensure input DataFrame has the same feature order and fill missing columns with 0
input_df = input_df.reindex(columns=features, fill_value=0)

# Standardize numerical input data
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.write(f"### Prediction: {result}")
