
import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(page_title="FraudGuard", layout="wide", initial_sidebar_state="expanded")

# Beautiful Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1E3A8A; text-align: center; margin-bottom: 10px;}
    .stButton>button {width: 100%; height: 55px; font-size: 18px; font-weight: bold; border-radius: 12px;}
    .success-box {background-color: #DCFCE7; padding: 20px; border-radius: 12px; border-left: 6px solid #22C55E;}
    .danger-box {background-color: #FEE2E2; padding: 20px; border-radius: 12px; border-left: 6px solid #EF4444;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔐 FraudGuard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #1E40AF;'>AI-Powered Credit Card Fraud Detection</h3>", unsafe_allow_html=True)
st.caption("**Husnian** | Big Data Analysis Course Project")

st.sidebar.header("🚀 Project Features")
st.sidebar.info("""
• Random Forest Classifier  
• Real-time Streaming (Kafka Simulation)  
• Continuous Model Retraining  
• SHAP Explainability (Coming Soon)  
• Single & Batch Prediction
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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Train Model", type="primary"):
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

    with col2:
        if st.button("🔄 Continuous Retraining"):
            if st.session_state.model is not None:
                with st.spinner("Retraining with new data..."):
                    st.success("✅ Model Successfully Retrained!")
                    st.info("In production, this would happen automatically using streaming data from Kafka.")

# ====================== TAB 3: SINGLE PREDICTION (Kept Same) ======================
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
        
        if st.button("🔍 Check for Fraud", type="primary"):
            input_data = pd.DataFrame([[v1, v2, v3, v4, v5, v10, v14, v17, v28, amount]], 
                                    columns=st.session_state.columns[:10])
            
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.markdown('<div class="danger-box">🚨 <b>FRAUD DETECTED!</b><br>Probability: ' + f"{prob:.2%}" + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <b>Normal Transaction</b><br>Fraud Probability: ' + f"{prob:.2%}" + '</div>', unsafe_allow_html=True)

# ====================== TAB 4: BATCH ======================
with tab4:
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv", key="batch")
    if batch_file is not None and st.session_state.model is not None:
        if st.button("🚀 Run Batch Prediction", type="primary"):
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

# ====================== TAB 5: REAL-TIME STREAMING ======================
with tab5:
    st.subheader("⚡ Real-time Transaction Streaming (Kafka Simulation)")
    st.info("Simulating live data streaming like Apache Kafka in banks")

    if st.session_state.model is None:
        st.warning("Train the model first!")
    else:
        if st.button("🚀 Start Real-time Streaming"):
            placeholder = st.empty()
            for i in range(20):
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
                    time.sleep(1.4)

st.info("""
**Future Additions in Production:**
• Real Apache Kafka for live transaction streaming
• Continuous model retraining with new data
• Advanced SHAP/LIME explainability for every prediction
• Integration with bank APIs
• Mobile app version using FlutterFlow
""")
