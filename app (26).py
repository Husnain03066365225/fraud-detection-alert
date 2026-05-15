%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(page_title="FraudGuard", layout="wide")
st.title("🔐 FraudGuard - Intelligent Fraud Detection")
st.subheader("Big Data Analysis + Machine Learning")
st.caption("**Husnian** | Big Data Analysis Course")

st.sidebar.header("About Project")
st.sidebar.info("""
This project demonstrates fraud detection using Machine Learning on large transaction data.
- Handles highly imbalanced data
- Single & Batch Prediction
- Big Data Analysis Course Project
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.columns = None

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Single Prediction", "📁 Batch Prediction"])

# ====================== TAB 1: OVERVIEW ======================
with tab1:
    uploaded_file = st.file_uploader("Upload Training Dataset", type="csv", key="train")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df):,}** transactions | Fraud Cases: **{df['Class'].sum()}**")
        st.bar_chart(df['Class'].value_counts())

# ====================== TAB 2: TRAIN MODEL ======================
with tab2:
    if st.button("🚀 Train Fraud Detection Model"):
        if uploaded_file is None:
            st.error("Please upload dataset first!")
        else:
            with st.spinner("Training..."):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ Model Trained! AUC Score: **{auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                st.session_state.model = model
                st.session_state.columns = X.columns.tolist()

# ====================== TAB 3: SINGLE PREDICTION ======================
with tab3:
    if st.session_state.model is None:
        st.warning("Train the model first!")
    else:
        st.subheader("Single Transaction Check")
        col1, col2 = st.columns(2)
        with col1:
            v1 = st.number_input("V1", value=-2.5)
            v2 = st.number_input("V2", value=1.8)
            v3 = st.number_input("V3", value=-3.2)
            amount = st.number_input("Amount ($)", value=999.99)
        with col2:
            v4 = st.number_input("V4", value=2.1)
            v5 = st.number_input("V5", value=-1.5)
        
        if st.button("🔍 Check for Fraud"):
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
    st.subheader("Batch Prediction (Upload CSV)")
    batch_file = st.file_uploader("Upload New Transactions CSV", type="csv", key="batch")
    
    if batch_file is not None and st.session_state.model is not None:
        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Predicting on all transactions..."):
                test_df = pd.read_csv(batch_file)
                X_test = test_df.drop(['Class'], axis=1) if 'Class' in test_df.columns else test_df
                
                predictions = st.session_state.model.predict(X_test)
                probabilities = st.session_state.model.predict_proba(X_test)[:, 1]
                
                test_df['Predicted_Fraud'] = predictions
                test_df['Fraud_Probability'] = probabilities
                
                fraud_count = predictions.sum()
                st.success(f"**Batch Prediction Complete!** Detected **{fraud_count}** Fraud Cases")
                
                st.dataframe(test_df.head(20))
                
                csv = test_df.to_csv(index=False)
                st.download_button("📥 Download Full Results", csv, "fraud_predictions.csv", "text/csv")

st.info("""
**Note:** This is a demonstration project. 
In production, more features would be added such as:
- Real-time data streaming (Kafka)
- Integration with bank APIs
- Better explainability (why a transaction is flagged)
- Mobile app version
- Continuous model retraining
""")
