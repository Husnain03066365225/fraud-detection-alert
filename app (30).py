
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudGuard", layout="wide")
st.title("🔐 FraudGuard - Intelligent Fraud Detection")
st.subheader("Big Data Analysis + Machine Learning + Explainability")
st.caption("**Husnian** | Big Data Analysis Course")

st.sidebar.header("Project Info")
st.sidebar.info("Fraud Detection with SHAP Explainability")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Single Prediction", "📁 Batch Prediction"])

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
                
                st.success(f"✅ Model Trained! AUC: **{auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                st.session_state.model = model
                st.session_state.columns = X.columns.tolist()

# ====================== TAB 3: SINGLE PREDICTION + SHAP ======================
with tab3:
    if st.session_state.model is None:
        st.warning("Train the model first!")
    else:
        st.subheader("Single Transaction Prediction")
        
        # Full V1 to V28 inputs in 4 columns
        cols = st.columns(4)
        values = []
        for i in range(28):
            with cols[i % 4]:
                val = st.number_input(f"V{i+1}", value=0.0, step=0.01, key=f"v{i}")
                values.append(val)
        
        amount = st.number_input("Transaction Amount ($)", value=999.99, step=10.0)
        
        if st.button("🔍 Predict + Explain with SHAP", type="primary"):
            input_data = pd.DataFrame([values + [amount]], columns=st.session_state.columns[:29])
            
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
            else:
                st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")
            
            # ====================== SHAP EXPLAINABILITY ======================
            st.subheader("🔍 SHAP Explainability (Why this prediction?)")
            with st.spinner("Calculating SHAP values..."):
                import shap
                explainer = shap.TreeExplainer(st.session_state.model)
                shap_values = explainer.shap_values(input_data)
                
                # Summary Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False)
                st.pyplot(fig)
                
                st.info("**Interpretation:** Longer bars = more impact on the prediction. Red = pushed toward fraud, Blue = pushed toward normal.")

# ====================== TAB 4: BATCH ======================
with tab4:
    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV", type="csv", key="batch")
    if batch_file is not None and st.session_state.model is not None:
        if st.button("Run Batch Prediction"):
            with st.spinner("Processing..."):
                test_df = pd.read_csv(batch_file)
                X_test = test_df.drop(['Class'], axis=1) if 'Class' in test_df.columns else test_df
                predictions = st.session_state.model.predict(X_test)
                probabilities = st.session_state.model.predict_proba(X_test)[:, 1]
                
                test_df['Predicted_Fraud'] = predictions
                test_df['Fraud_Probability'] = probabilities
                
                st.success(f"Detected **{predictions.sum()}** Fraud Cases")
                st.dataframe(test_df.head(15))
                st.download_button("Download Results", test_df.to_csv(index=False), "predictions.csv")

st.info("**Note:** This is a demonstration project. In production, real-time streaming, better explainability, and continuous retraining would be implemented.")
