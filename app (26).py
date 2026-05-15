
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

# Sidebar
st.sidebar.header("About Project")
st.sidebar.info("""
This project detects fraudulent credit card transactions using Machine Learning.
- Handles highly imbalanced data
- Real-time prediction
- Batch prediction support
""")

# Main Content
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🛠️ Train Model", "🔮 Predict Fraud"])

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.columns = None

# ====================== TAB 1: OVERVIEW ======================
with tab1:
    uploaded_file = st.file_uploader("Upload Training Dataset (CSV with 'Class' column)", type="csv", key="train")
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

# ====================== TAB 3: PREDICT ======================
with tab3:
    if st.session_state.model is None:
        st.warning("Please train the model first in Tab 2")
    else:
        st.subheader("Single Transaction Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            v1 = st.number_input("V1", value=-1.0)
            v2 = st.number_input("V2", value=1.0)
            amount = st.number_input("Amount ($)", value=999.99)
        
        with col2:
            v3 = st.number_input("V3", value=-2.0)
            v4 = st.number_input("V4", value=2.0)
        
        if st.button("🔍 Check for Fraud"):
            input_data = pd.DataFrame([[v1, v2, v3, v4, amount]], columns=st.session_state.columns[:5])
            # Fill remaining columns with 0 for demo
            for col in st.session_state.columns[5:]:
                input_data[col] = 0.0
            
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
            else:
                st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")

st.info("**Note:** This is a demonstration project. In production, more features and real-time data would be used.")
