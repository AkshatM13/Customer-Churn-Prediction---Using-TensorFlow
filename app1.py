import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess data
@st.cache_data
def load_data():
    data_path = "D:/MSIS/Customer churn prediction/Customer-Churn-Prediction---Using-TensorFlow/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data = pd.read_csv(data_path)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(subset=['TotalCharges'], inplace=True)
    data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
    data = data.drop(['customerID'], axis=1)
    
    # Encode categorical variables
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = LabelEncoder().fit_transform(data[column])

    return data

data = load_data()

# Split data
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize numerical columns
scaler = StandardScaler()
X_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
X_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get feature names for reindexing user input
features = X.columns

# Streamlit app layout
st.title("Customer Churn Prediction")

# User input function
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

    # Map categorical inputs to numbers
    data['InternetService'] = {"DSL": 0, "Fiber optic": 1, "No": 2}[data['InternetService']]
    data['Contract'] = {"Month-to-month": 0, "One year": 1, "Two year": 2}[data['Contract']]
    data['PaymentMethod'] = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[data['PaymentMethod']]

    # Convert input to DataFrame
    return pd.DataFrame(data, index=[0])

# Collect user input
input_df = user_input()

# Ensure input DataFrame matches training feature structure
input_df = input_df.reindex(columns=features, fill_value=0)

# Standardize numerical input features
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.write(f"### Prediction: {result}")
