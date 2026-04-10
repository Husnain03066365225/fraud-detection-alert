
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection System")
st.subheader("Big Data + Machine Learning")
st.caption("**Husnian** | Big Data Analysis Course")

uploaded_file = st.file_uploader("Upload Training Dataset (CSV with 'Class' column)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} records | Fraud Cases: {df['Class'].sum()}")

    if st.button("🚀 Train Model"):
        with st.spinner("Training Model..."):
            X = df.drop(['Class'], axis=1)
            y = df['Class']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)

            st.success(f"Model Trained Successfully! AUC = {auc:.4f}")
            st.text(classification_report(y_test, y_pred))

            st.session_state.model = model
            st.success("Model is ready for prediction!")

    # Prediction Section
    if 'model' in st.session_state:
        st.divider()
        st.subheader("Predict Fraud")

        # Simple input for testing
        amount = st.number_input("Transaction Amount", value=100.0)

        if st.button("Predict"):
            # Dummy input (28 V features + Amount)
            input_data = np.zeros((1, 29))
            input_data[0, -1] = amount
            pred = st.session_state.model.predict(input_data)[0]
            prob = st.session_state.model.predict_proba(input_data)[0][1]

            if pred == 1:
                st.error(f"🚨 FRAUD DETECTED! ({prob:.2%} probability)")
            else:
                st.success(f"✅ Normal Transaction ({prob:.2%} fraud probability)")

else:
    st.info("Please upload your dataset to begin")
