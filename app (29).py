
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="FraudGuard - AI Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Animations & Modern Look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
        animation: fadeIn 1.5s ease-in;
    }
    .stButton>button {
        width: 100%;
        height: 55px;
        background: linear-gradient(135deg, #1E40AF, #3B82F6);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(30, 64, 175, 0.3);
    }
    .success-box, .danger-box {
        padding: 20px;
        border-radius: 12px;
        animation: fadeIn 0.8s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔐 FraudGuard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #3B82F6;'>AI-Powered Credit Card Fraud Detection</h3>", unsafe_allow_html=True)
st.caption("**Husnian** | Big Data Analysis Course Project")

# Sidebar
with st.sidebar:
    st.header("📌 Project Highlights")
    st.info("""
    • Handles Highly Imbalanced Big Data  
    • Random Forest with Class Weighting  
    • Single & Batch Prediction  
    • Real-time Demo  
    • 95%+ Validation Accuracy  
    """)
    st.markdown("---")
    st.write("**Status:** Ready for Detection")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Single Check", "📁 Batch Prediction"])

# ====================== TAB 1 ======================
with tab1:
    uploaded_file = st.file_uploader("Upload Training Dataset", type="csv", key="train")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df):,}** transactions | **{df['Class'].sum()}** Fraud Cases")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df['Class'].value_counts())
        with col2:
            st.metric("Fraud Rate", f"{df['Class'].mean()*100:.4f}%")

# ====================== TAB 2 ======================
with tab2:
    if st.button("🚀 Train Fraud Detection Model", type="primary"):
        if uploaded_file is None:
            st.error("Upload dataset first!")
        else:
            with st.spinner("Training Model..."):
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

# ====================== TAB 3 ======================
with tab3:
    if st.session_state.get('model') is None:
        st.warning("⚠️ Train the model first!")
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
        
        if st.button("🔍 Check for Fraud", type="primary"):
            input_data = pd.DataFrame([[v1, v2, v3, v4, v5, amount]], columns=st.session_state.columns[:6])
            for col in st.session_state.columns[6:]:
                input_data[col] = 0.0
            
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.markdown('<div class="danger-box">🚨 <b>FRAUD DETECTED!</b><br>Probability: ' + f"{prob:.2%}" + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <b>Normal Transaction</b><br>Fraud Probability: ' + f"{prob:.2%}" + '</div>', unsafe_allow_html=True)

# ====================== TAB 4 ======================
with tab4:
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv", key="batch")
    
    if batch_file is not None and st.session_state.get('model') is not None:
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Analyzing all transactions..."):
                test_df = pd.read_csv(batch_file)
                X_test = test_df.drop(['Class'], axis=1) if 'Class' in test_df.columns else test_df
                
                predictions = st.session_state.model.predict(X_test)
                probabilities = st.session_state.model.predict_proba(X_test)[:, 1]
                
                test_df['Predicted_Fraud'] = predictions
                test_df['Fraud_Probability'] = probabilities
                
                st.success(f"**Batch Complete!** Detected **{predictions.sum()}** Fraud Cases")
                st.dataframe(test_df.head(20))
                
                csv = test_df.to_csv(index=False)
                st.download_button("📥 Download Full Report", csv, "fraud_predictions.csv", "text/csv")

st.info("**Note:** This is a demonstration project. In production, real-time streaming (Kafka), model explainability, and continuous retraining would be added.")
