import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

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

# App Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.write(
    "This application predicts whether a customer will **churn** based on various attributes. "
    "Please provide the necessary details in the sidebar and click **Predict Churn**."
)

# Sidebar for User Input
st.sidebar.header("📋 Customer Details")
def user_input():
    data = {
        'tenure': st.sidebar.slider("Tenure (Months)", min_value=0, max_value=100, value=10),
        'MonthlyCharges': st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0),
        'TotalCharges': st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0),
        'SeniorCitizen': 1 if st.sidebar.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes" else 0,
        'Partner': 1 if st.sidebar.selectbox("Partner", ["No", "Yes"]) == "Yes" else 0,
        'Dependents': 1 if st.sidebar.selectbox("Dependents", ["No", "Yes"]) == "Yes" else 0,
        'PhoneService': 1 if st.sidebar.selectbox("Phone Service", ["No", "Yes"]) == "Yes" else 0,
        'MultipleLines': 1 if st.sidebar.selectbox("Multiple Lines", ["No", "Yes"]) == "Yes" else 0,
        'InternetService': st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        'OnlineSecurity': 1 if st.sidebar.selectbox("Online Security", ["No", "Yes"]) == "Yes" else 0,
        'OnlineBackup': 1 if st.sidebar.selectbox("Online Backup", ["No", "Yes"]) == "Yes" else 0,
        'DeviceProtection': 1 if st.sidebar.selectbox("Device Protection", ["No", "Yes"]) == "Yes" else 0,
        'TechSupport': 1 if st.sidebar.selectbox("Tech Support", ["No", "Yes"]) == "Yes" else 0,
        'StreamingTV': 1 if st.sidebar.selectbox("Streaming TV", ["No", "Yes"]) == "Yes" else 0,
        'StreamingMovies': 1 if st.sidebar.selectbox("Streaming Movies", ["No", "Yes"]) == "Yes" else 0,
        'Contract': st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        'PaperlessBilling': 1 if st.sidebar.selectbox("Paperless Billing", ["No", "Yes"]) == "Yes" else 0,
        'PaymentMethod': st.sidebar.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
    }

    # Map categorical inputs to numbers
    data['InternetService'] = {"DSL": 0, "Fiber optic": 1, "No": 2}[data['InternetService']]
    data['Contract'] = {"Month-to-month": 0, "One year": 1, "Two year": 2}[data['Contract']]
    data['PaymentMethod'] = {
        "Electronic check": 0, 
        "Mailed check": 1, 
        "Bank transfer (automatic)": 2, 
        "Credit card (automatic)": 3
    }[data['PaymentMethod']]

    # Convert input to DataFrame
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Ensure input DataFrame matches the training feature structure
input_df = input_df.reindex(columns=features, fill_value=0)
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

# Prediction Button and Output
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_df)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    
    # Display result with color formatting
    if result == "Churn":
        st.markdown(
            f"<h2 style='text-align: center; color: red;'>⚠️ Prediction: {result}</h2>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h2 style='text-align: center; color: green;'>✅ Prediction: {result}</h2>", 
            unsafe_allow_html=True
        )
