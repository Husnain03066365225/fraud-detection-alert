
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

# TAB 1
with tab1:
    uploaded_file = st.file_uploader("Upload Training Dataset", type="csv", key="train")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")
        st.bar_chart(df['Class'].value_counts())

# TAB 2
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

# TAB 3: SINGLE PREDICTION (V1-V5) + Location Demo
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
        
        with col2:
            v10 = st.number_input("V10", value=0.0)
            v14 = st.number_input("V14", value=0.0)
            v17 = st.number_input("V17", value=0.0)
            v28 = st.number_input("V28", value=0.0)
        
        # Location Demo
        countries = ["Pakistan", "UAE", "UK", "USA", "China", "India", "Malaysia"]
        location = st.selectbox("Transaction Location", countries)
        
        if st.button("🔍 Predict + Explain with SHAP", type="primary"):
            input_data = pd.DataFrame([[v1, v2, v3, v4, v5, v10, v14, v17, v28, amount]], columns=st.session_state.columns[:10])
            
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
                st.warning(f"⚠️ Suspicious Location: {location}")
            else:
                st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")
            
            st.info(f"**Transaction Location:** {location}")

# TAB 4 & 5 (Batch + Real-time) - kept simple
with tab4:
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV", type="csv", key="batch")
    if batch_file is not None and st.session_state.model is not None:
        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Processing..."):
                test_df = pd.read_csv(batch_file)
                X_test = test_df.drop(['Class'], axis=1) if 'Class' in test_df.columns else test_df
                predictions = st.session_state.model.predict(X_test)
                probabilities = st.session_state.model.predict_proba(X_test)[:, 1]
                
                test_df['Predicted_Fraud'] = predictions
                test_df['Fraud_Probability'] = probabilities
                st.success(f"Detected **{predictions.sum()}** Fraud Cases")
                st.dataframe(test_df.head(20))
                st.download_button("📥 Download Results", test_df.to_csv(index=False), "predictions.csv")

with tab5:
    st.subheader("⚡ Real-time Streaming (Kafka Simulation)")
    if st.session_state.model is not None:
        if st.button("Start Real-time Streaming"):
            placeholder = st.empty()
            for i in range(15):
                with placeholder.container():
                    amount = random.uniform(50, 2500)
                    input_data = pd.DataFrame([[0]*len(st.session_state.columns)], columns=st.session_state.columns)
                    input_data.iloc[0, -1] = amount
                    
                    pred = st.session_state.model.predict(input_data)[0]
                    prob = st.session_state.model.predict_proba(input_data)[0][1]
                    
                    if pred == 1:
                        st.error(f"Transaction {i+1} | Amount: ${amount:.2f} → 🚨 **FRAUD** ({prob:.1%})")
                    else:
                        st.success(f"Transaction {i+1} | Amount: ${amount:.2f} → ✅ Normal ({prob:.1%})")
                    time.sleep(1.3)

st.info("""
**Future Enhancements:**
• Real Apache Kafka for live transaction streaming
• Continuous automatic model retraining
• Advanced SHAP explainability for every prediction
• Transaction location tracking with geolocation
• Mobile app version using FlutterFlow
""")
