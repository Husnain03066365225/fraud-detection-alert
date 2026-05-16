
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import time
import random

st.set_page_config(page_title="FraudGuard", layout="wide")
st.title("🔐 FraudGuard - Intelligent Fraud Detection")
st.subheader("Big Data Analysis + Machine Learning + Real-time Streaming")
st.caption("**Husnian** | Big Data Analysis Course")

st.sidebar.header("Project Highlights")
st.sidebar.info("""
• Fraud Detection using Random Forest
• SHAP Explainability
• Single & Batch Prediction
• Real-time Streaming Simulation (Kafka Style)
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Single Prediction", "📁 Batch Prediction", "⚡ Real-time Streaming"])

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.columns = None

# ====================== TAB 1 ======================
with tab1:
    uploaded_file = st.file_uploader("Upload Training Dataset", type="csv", key="train")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")
        st.bar_chart(df['Class'].value_counts())

# ====================== TAB 2 ======================
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

# ====================== TAB 3: SINGLE PREDICTION ======================
with tab3:
    if st.session_state.model is None:
        st.warning("Train the model first!")
    else:
        st.subheader("Single Transaction Check")
        cols = st.columns(4)
        values = []
        for i in range(28):
            with cols[i % 4]:
                val = st.number_input(f"V{i+1}", value=0.0, step=0.01, key=f"v{i}")
                values.append(val)
        amount = st.number_input("Transaction Amount ($)", value=999.99, step=10.0)
        
        if st.button("🔍 Check for Fraud", type="primary"):
            input_data = pd.DataFrame([values + [amount]], columns=st.session_state.columns[:29])
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
            else:
                st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")

# ====================== TAB 4: BATCH ======================
with tab4:
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV", type="csv", key="batch")
    if batch_file is not None and st.session_state.model is not None:
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing..."):
                test_df = pd.read_csv(batch_file)
                X_test = test_df.drop(['Class'], axis=1) if 'Class' in test_df.columns else test_df
                predictions = st.session_state.model.predict(X_test)
                probabilities = st.session_state.model.predict_proba(X_test)[:, 1]
                
                test_df['Predicted_Fraud'] = predictions
                test_df['Fraud_Probability'] = probabilities
                st.success(f"Detected **{predictions.sum()}** Fraud Cases")
                st.dataframe(test_df.head(20))
                st.download_button("Download Results", test_df.to_csv(index=False), "predictions.csv")

# ====================== TAB 5: REAL-TIME STREAMING ======================
with tab5:
    st.subheader("⚡ Real-time Transaction Streaming (Kafka Simulation)")
    st.info("This simulates real-time data streaming like Apache Kafka")

    if st.session_state.model is None:
        st.warning("Train the model first!")
    else:
        if st.button("Start Real-time Streaming"):
            st.write("**Live Transactions Streaming...**")
            placeholder = st.empty()
            
            for i in range(20):   # Simulate 20 transactions
                with placeholder.container():
                    # Generate random transaction
                    amount = random.uniform(10, 2000)
                    v_values = np.random.normal(0, 1, 28)
                    input_data = pd.DataFrame([list(v_values) + [amount]], columns=st.session_state.columns[:29])
                    
                    pred = st.session_state.model.predict(input_data)[0]
                    prob = st.session_state.model.predict_proba(input_data)[0][1]
                    
                    if pred == 1:
                        st.error(f"Transaction {i+1} | Amount: ${amount:.2f} → 🚨 **FRAUD** ({prob:.2%})")
                    else:
                        st.success(f"Transaction {i+1} | Amount: ${amount:.2f} → ✅ Normal ({prob:.2%})")
                    
                    time.sleep(1.2)  # Simulate streaming delay

st.info("""
**Note:** This is a demonstration project. 
In production, we would use **Apache Kafka** for real-time data streaming, integrate with bank systems, and add continuous model retraining.
""")
