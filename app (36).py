
import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(page_title="FraudGuard", layout="wide")
st.title("🔐 FraudGuard - Intelligent Fraud Detection")
st.subheader("Big Data Analysis + Machine Learning")
st.caption("**Husnian** | Big Data Analysis Course")

st.sidebar.header("Project Features")
st.sidebar.info("""
• Random Forest Classifier
• SHAP Explainability
• Real-time Streaming Simulation
• Continuous Retraining
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Single Prediction", "📁 Batch Prediction", "⚡ Real-time Streaming"])

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.columns = None

# TAB 1: Overview
with tab1:
    uploaded_file = st.file_uploader("Upload Training Dataset", type="csv", key="train")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")
        st.bar_chart(df['Class'].value_counts())

# TAB 2: Train Model
with tab2:
    if st.button("🚀 Train Model"):
        if uploaded_file is None:
            st.error("Upload dataset first!")
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

# TAB 3: Single Prediction (Only V1-V5 + Amount)
with tab3:
    if st.session_state.model is None:
        st.warning("Train the model first!")
    else:
        st.subheader("Single Transaction Check")
        
        col1, col2 = st.columns(2)
        with col1:
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)
            amount = st.number_input("Transaction Amount ($)", value=999.99)
        
        if st.button("🔍 Predict + Explain with SHAP", type="primary"):
            # Using only available columns
            input_data = pd.DataFrame([[v1, v2, v3, v4, v5, amount]], columns=st.session_state.columns[:6])
            
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
            else:
                st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")
            
            # SHAP Explainability
            st.subheader("🔍 SHAP Explainability - Why this decision?")
            with st.spinner("Calculating SHAP values..."):
                import shap
                explainer = shap.TreeExplainer(st.session_state.model)
                shap_values = explainer.shap_values(input_data)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False)
                st.pyplot(fig)
                
                st.info("**Longer bars = stronger influence.** Positive values pushed toward fraud.")

# TAB 4 & 5 (Batch + Real-time) can stay as previous if you want, or I can add them.

st.info("""
**Future Enhancements:** Real Apache Kafka streaming, continuous retraining, advanced SHAP explanations, and mobile app version.
""")
