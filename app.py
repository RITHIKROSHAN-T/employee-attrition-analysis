import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# --- ESSENTIAL CONSTANTS ---
MODEL_PATH = "final_attrition_pipeline.pkl"  # Correct model file name
FINAL_THRESHOLD = 0.35  # The optimal threshold found during analysis

# The exact order and names of the 19 features the model was trained on
FINAL_FEATURE_ORDER = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction', 'OverTime',
    'WorkLifeBalance', 'JobRole', 'Department', 'Gender', 'YearsAtCompany',
    'MaritalStatus', 'TotalWorkingYears', 'NumCompaniesWorked',
    'EnvironmentSatisfaction', 'RelationshipSatisfaction',
    'YearsPerJob', 'DistancePerIncome', 'Satisfaction_Index', 'Experience_Gap'
]

# Load model
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"FATAL ERROR: Model file not found at {MODEL_PATH}. Check the filename and location.")
    model = None
    
# Load data (CRITICAL FIX: Using your specific path again)
DATA_FILE = "../data/Employee-Attrition - Employee-Attrition.csv"
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    # If the specific path fails, try the current directory (most common fix)
    DATA_FILE = "Employee-Attrition - Employee-Attrition.csv"
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.warning("Warning: Data file not found. Ensure 'Employee-Attrition - Employee-Attrition.csv' is in the root or '../data/' directory.")
        df = pd.DataFrame()


# Sidebar navigation
page = st.sidebar.radio("Employee Attrition Analysis", ["Home", "Predict Employee Attrition"])

# ----------------------- HOME DASHBOARD --------------------------
if page == "Home":
    st.title("📊 Employee Insights Dashboard")

    if not df.empty and "Attrition" in df.columns:
        st.subheader("🚨 High-Risk Employees")
        df['attrition_risk'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🔍 High Attrition Risk")
            st.dataframe(df[['EmployeeNumber', 'attrition_risk', 'PerformanceRating']].sort_values(by='attrition_risk', ascending=False).head(10))

        with col2:
            st.markdown("### 🙂 High Job Satisfaction")
            high_sat = df[df['JobSatisfaction'] == 4]
            st.dataframe(high_sat[['EmployeeNumber', 'JobSatisfaction', 'attrition_risk']].head(10))

        with col3:
            st.markdown("### 🏆 High Performance Score")
            top_perf = df.sort_values(by='PerformanceRating', ascending=False)
            st.dataframe(top_perf[['EmployeeNumber', 'PerformanceRating', 'JobSatisfaction']].head(10))
    else:
        st.info("Dashboard data not fully available. Please place the data file correctly.")


# ----------------------- PREDICTION PAGE --------------------------
elif page == "Predict Employee Attrition":
    st.title("🧠 Predict Employee Attrition")

    if model is None:
        st.stop()
        
    st.markdown("### Enter Employee Details")

    # --- User Inputs (Vertical Design Preserved. All 15 required inputs included) ---
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", df['JobRole'].unique() if not df.empty else ["Sales Executive"])
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
    distance = st.slider("Distance From Home (miles)", 1, 30, 5)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    
    # Original Inputs
    job_satisfaction = st.selectbox("Job Satisfaction (1 - 4)", [1, 2, 3, 4])
    work_life_balance = st.selectbox("Work-Life Balance (1 - 4)", [1, 2, 3, 4])
    years_at_company = st.slider("Years at Company", 0, 40, 5)

    # Added Missing Base Features (Integrated into the vertical flow)
    total_working_years = st.slider("Total Working Years (Total career experience)", 0, 40, 10)
    num_companies_worked = st.slider("Num Companies Worked (Excluding current one)", 0, 9, 2)
    env_satisfaction = st.selectbox("Environment Satisfaction (1 - 4)", [1, 2, 3, 4])
    rel_satisfaction = st.selectbox("Relationship Satisfaction (1 - 4)", [1, 2, 3, 4])


    # Prepare a dictionary with ALL 15 base inputs
    base_inputs = {
        'Gender': gender, 'MaritalStatus': marital_status, 'Department': department,
        'JobRole': job_role, 'Age': age, 'MonthlyIncome': monthly_income,
        'DistanceFromHome': distance, 'OverTime': overtime, 'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
        'TotalWorkingYears': total_working_years, 'NumCompaniesWorked': num_companies_worked,
        'EnvironmentSatisfaction': env_satisfaction, 'RelationshipSatisfaction': rel_satisfaction
    }
    input_df = pd.DataFrame([base_inputs])


    if st.button("🔍 Predict Attrition"):
        
        # 1. IMPLEMENT FEATURE ENGINEERING (CRITICAL FIX)
        num_companies_worked_adj = input_df['NumCompaniesWorked'].replace(0, 1).iloc[0]
        
        input_df['YearsPerJob'] = input_df['TotalWorkingYears'] / num_companies_worked_adj
        input_df['DistancePerIncome'] = input_df['DistanceFromHome'] / input_df['MonthlyIncome']
        input_df['Satisfaction_Index'] = (input_df['JobSatisfaction'] + input_df['EnvironmentSatisfaction'] + input_df['RelationshipSatisfaction']) / 3
        input_df['Experience_Gap'] = input_df['TotalWorkingYears'] - input_df['YearsAtCompany']
        
        # 2. Reorder features to match model's expected input (19 total features)
        final_input_df = input_df[FINAL_FEATURE_ORDER]
        
        # 3. Calculate probability
        probability = model.predict_proba(final_input_df)[0][1]

        # 4. APPLY CUSTOM THRESHOLD (CRITICAL FIX)
        prediction = 1 if probability >= FINAL_THRESHOLD else 0

        st.subheader("📢 Prediction Result")
        st.metric(label="Probability of Leaving", value=f"{probability * 100:.2f}%")
        
        if prediction == 1:
            st.error(f"🚨 HIGH ATTRITION RISK! The employee is predicted to leave (Threshold: {FINAL_THRESHOLD}).")
        else:
            st.success(f"✅ Low Attrition Risk. The employee is likely to stay.")

        st.markdown("### 🔍 Scaled Input Preview")
        st.write(final_input_df)