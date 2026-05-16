
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Page Configuration
st.set_page_config(
    page_title="FraudGuard - AI Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}
    .stButton>button {width: 100%; background-color: #1E40AF; color: white; height: 50px; font-size: 16px;}
    .success-box {background-color: #DCFCE7; padding: 20px; border-radius: 10px;}
    .danger-box {background-color: #FEE2E2; padding: 20px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🔐 FraudGuard")
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>Intelligent Credit Card Fraud Detection System</h2>", unsafe_allow_html=True)
st.caption("**Husnian** | Big Data Analysis Course Project")

# Sidebar
with st.sidebar:
    st.header("📌 About Project")
    st.info("""
    This project detects fraudulent credit card transactions using Machine Learning.
    - Handles highly imbalanced Big Data
    - Real-time & Batch Prediction
    - High Precision Model
    """)
    st.markdown("---")
    st.write("**Status:** Model Ready")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Single Prediction", "📁 Batch Prediction"])

# ====================== TAB 1: OVERVIEW ======================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Training Dataset (CSV with 'Class' column)", type="csv", key="train")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset Loaded: **{len(df):,}** transactions | **{df['Class'].sum()}** Fraud Cases")
            st.bar_chart(df['Class'].value_counts())

# ====================== TAB 2: TRAIN MODEL ======================
with tab2:
    st.subheader("Train Fraud Detection Model")
    if st.button("🚀 Train Improved Model", type="primary"):
        if uploaded_file is None:
            st.error("Please upload dataset first!")
        else:
            with st.spinner("Training model..."):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ Model Trained Successfully! **AUC Score: {auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                st.session_state.model = model
                st.session_state.columns = X.columns.tolist()

# ====================== TAB 3: SINGLE PREDICTION ======================
with tab3:
    if st.session_state.get('model') is None:
        st.warning("⚠️ Please train the model first in the Train Model tab")
    else:
        st.subheader("Single Transaction Check")
        col1, col2 = st.columns(2)
        
        with col1:
            v1 = st.number_input("V1", value=-2.5, step=0.1)
            v2 = st.number_input("V2", value=1.8, step=0.1)
            v3 = st.number_input("V3", value=-3.2, step=0.1)
            amount = st.number_input("Transaction Amount ($)", value=999.99, step=10.0)
        
        with col2:
            v4 = st.number_input("V4", value=2.1, step=0.1)
            v5 = st.number_input("V5", value=-1.5, step=0.1)
        
        if st.button("🔍 Check for Fraud", type="primary"):
            input_data = pd.DataFrame([[v1, v2, v3, v4, v5, amount]], columns=st.session_state.columns[:6])
            for col in st.session_state.columns[6:]:
                input_data[col] = 0.0
                
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
            else:
                st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")

# ====================== TAB 4: BATCH PREDICTION ======================
with tab4:
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload Transactions CSV for Bulk Prediction", type="csv", key="batch")
    
    if batch_file is not None and st.session_state.get('model') is not None:
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing batch..."):
                test_df = pd.read_csv(batch_file)
                X_test = test_df.drop(['Class'], axis=1) if 'Class' in test_df.columns else test_df
                
                predictions = st.session_state.model.predict(X_test)
                probabilities = st.session_state.model.predict_proba(X_test)[:, 1]
                
                test_df['Predicted_Fraud'] = predictions
                test_df['Fraud_Probability'] = probabilities
                
                st.success(f"**Batch Prediction Complete!** Detected **{predictions.sum()}** Fraud Cases")
                st.dataframe(test_df.head(20))
                
                csv = test_df.to_csv(index=False)
                st.download_button("📥 Download Full Results", csv, "fraud_predictions.csv", "text/csv")

st.info("**Note:** This is a demonstration project. In production, real-time streaming, better explainability, and continuous model retraining would be implemented.")
