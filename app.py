import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="💰",
    layout="centered"
)

# ---------------- LOAD CSS ----------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- LOAD OR TRAIN MODELS ----------------
@st.cache_resource
def load_or_train_models():
    if (os.path.exists("svm_linear.pkl") and os.path.exists("svm_poly.pkl") and 
        os.path.exists("svm_rbf.pkl") and os.path.exists("scaler.pkl")):
        # Load existing models
        svm_linear = joblib.load("svm_linear.pkl")
        svm_poly = joblib.load("svm_poly.pkl")
        svm_rbf = joblib.load("svm_rbf.pkl")
        scaler = joblib.load("scaler.pkl")
    else:
        # Train models with sample data if files don't exist
        st.info("Training models from sample data... Please wait.")
        
        # Create sample loan data
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'ApplicantIncome': np.random.randint(1500, 15000, n_samples),
            'LoanAmount': np.random.randint(50, 500, n_samples),
            'Credit_History': np.random.choice(['Yes', 'No'], n_samples),
            'Employment_Status': np.random.choice(['Salaried', 'Self-Employed'], n_samples),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
            'Loan_Status': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Prepare features
        x = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']
        
        # Encode categorical columns
        label_encoders = {}
        categorical_cols = x.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            x[col] = le.fit_transform(x[col])
            label_encoders[col] = le
        
        # Split and scale data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        # Train SVM models
        svm_linear = SVC(kernel='linear', C=1)
        svm_linear.fit(x_train, y_train)
        
        svm_poly = SVC(kernel='poly', degree=3)
        svm_poly.fit(x_train, y_train)
        
        svm_rbf = SVC(kernel='rbf')
        svm_rbf.fit(x_train, y_train)
        
        # Save models
        joblib.dump(svm_linear, "svm_linear.pkl")
        joblib.dump(svm_poly, "svm_poly.pkl")
        joblib.dump(svm_rbf, "svm_rbf.pkl")
        joblib.dump(scaler, "scaler.pkl")
    
    return svm_linear, svm_poly, svm_rbf, scaler

svm_linear, svm_poly, svm_rbf, scaler = load_or_train_models()

# ---------------- TITLE ----------------
st.markdown("<h1 class='title'>Smart Loan Approval System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='description'>This system uses Support Vector Machines to predict loan approval.</p>",
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ----------------
st.sidebar.header("📋 Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------------- ENCODING ----------------
credit = 1 if credit_history == "Yes" else 0
employment = 1 if employment == "Self-Employed" else 0
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_map[property_area]

# Feature order must match training
features = np.array([[income, loan_amount, credit, employment, property_area]])
features_scaled = scaler.transform(features)

# ---------------- MODEL SELECTION ----------------
st.subheader("🔍 Choose SVM Kernel")
kernel_choice = st.radio(
    "Select Kernel:",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

model = {
    "Linear SVM": svm_linear,
    "Polynomial SVM": svm_poly,
    "RBF SVM": svm_rbf
}[kernel_choice]

# ---------------- PREDICTION ----------------
if st.button("✅ Check Loan Eligibility"):
    prediction = model.predict(features_scaled)[0]
    confidence = abs(model.decision_function(features_scaled)[0]) * 100

    if prediction == 1:
        st.markdown("<div class='approved'>✅ Loan Approved</div>", unsafe_allow_html=True)
        explanation = (
            "Based on the applicant's credit history and income pattern, "
            "the applicant is likely to repay the loan."
        )
    else:
        st.markdown("<div class='rejected'>❌ Loan Rejected</div>", unsafe_allow_html=True)
        explanation = (
            "Based on credit risk and income pattern, "
            "the applicant is unlikely to repay the loan."
        )

    # ---------------- OUTPUT ----------------
    st.markdown("### 📊 Business Explanation")
    st.write(explanation)

    st.markdown("### ⚙️ Model Information")
    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")
