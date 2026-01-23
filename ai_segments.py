import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Invoice Fraud Detection",
    page_icon="📊",
    layout="wide"
)

st.title("🤖 AI Invoice Fraud Detection & Payment Scheduler")
st.markdown("Automated invoice validation, anomaly detection & payment prioritization")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Invoice CSV", type=["csv"])

if uploaded_file:

    # -------------------------------
    # DATA AGENT
    # -------------------------------
    df = pd.read_csv(uploaded_file)

    df['issuedDate'] = pd.to_datetime(df['issuedDate'], format='%d-%m-%Y')
    df['dueDate'] = pd.to_datetime(df['dueDate'], format='%d-%m-%Y')

    st.success("✅ Data Loaded Successfully")

    # -------------------------------
    # INVOICE VALIDATION AGENT
    # -------------------------------
    df['is_valid'] = True
    df.loc[df['total'] <= 0, 'is_valid'] = False

    expected_gst = df['total'] * 0.18
    df.loc[abs(df['tax'] - expected_gst) > 100, 'is_valid'] = False

    # -------------------------------
    # DUPLICATE DETECTION AGENT
    # -------------------------------
    le = LabelEncoder()
    df['vendor_enc'] = le.fit_transform(df['client'])

    df['duplicate_flag'] = 0
    duplicates = df.duplicated(
        subset=['client', 'id_invoice', 'total'],
        keep=False
    )
    df.loc[duplicates, 'duplicate_flag'] = 1

    # -------------------------------
    # ANOMALY DETECTION AGENT (AI)
    # -------------------------------
    model = IsolationForest(contamination=0.1, random_state=42)
    df['amount_scaled'] = df['total']
    df['anomaly'] = model.fit_predict(df[['amount_scaled']])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    # -------------------------------
    # PAYMENT SCHEDULING AGENT
    # -------------------------------
    today = datetime.today()
    df['days_to_due'] = (df['dueDate'] - today).dt.days

    df['payment_priority'] = np.where(
        (df['days_to_due'] <= 7) &
        (df['anomaly'] == 0) &
        (df['duplicate_flag'] == 0) &
        (df['is_valid'] == True),
        'HIGH',
        'HOLD'
    )

    # -------------------------------
    # DASHBOARD METRICS
    # -------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("📄 Total Invoices", len(df))
    col2.metric("✅ Valid", df['is_valid'].sum())
    col3.metric("🔁 Duplicates", df['duplicate_flag'].sum())
    col4.metric("🚨 Anomalies", df['anomaly'].sum())
    col5.metric("💰 Ready for Payment", (df['payment_priority'] == 'HIGH').sum())

    st.divider()

    # -------------------------------
    # VISUAL CHARTS
    # -------------------------------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("🚨 Anomaly Distribution")
        st.bar_chart(df['anomaly'].value_counts())

    with colB:
        st.subheader("💰 Payment Priority")
        st.bar_chart(df['payment_priority'].value_counts())

    # -------------------------------
    # DATA TABLE
    # -------------------------------
    st.subheader("📋 Invoice Details")
    st.dataframe(
        df[['id_invoice', 'client', 'total', 'is_valid',
            'duplicate_flag', 'anomaly', 'payment_priority']],
        use_container_width=True
    )

    # -------------------------------
    # DOWNLOAD OUTPUT
    # -------------------------------
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇ Download Processed Invoices",
        data=csv,
        file_name="processed_invoices_output.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a CSV file to start analysis")
