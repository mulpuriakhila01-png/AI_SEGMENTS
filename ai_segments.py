import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Smart AI Invoice System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Smart AI Invoice Processing System")
st.caption("Works with ANY invoice CSV format")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader("📂 Upload Invoice CSV", type=["csv"])

# -----------------------------------
# COLUMN MATCHING DICTIONARY
# -----------------------------------
COLUMN_MAP = {
    "id_invoice": ["invoice_id", "invoice_no", "inv_id", "bill_no"],
    "client": ["vendor", "supplier", "client_name", "customer"],
    "total": ["amount", "total_amount", "invoice_value", "bill_amount"],
    "tax": ["tax", "gst", "vat", "tax_amount"],
    "issuedDate": ["invoice_date", "bill_date", "issued_date"],
    "dueDate": ["due_date", "payment_due", "pay_date"]
}

def auto_map_columns(df):
    mapped = {}
    for key, aliases in COLUMN_MAP.items():
        for col in df.columns:
            if col.lower() in aliases:
                mapped[key] = col
                break
    return mapped

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded Successfully")

    detected_map = auto_map_columns(df_raw)

    st.subheader("🔍 Column Mapping")
    st.info("Auto-detected columns. You can change if needed.")

    user_map = {}
    for field in COLUMN_MAP.keys():
        options = df_raw.columns.tolist()
        default = detected_map.get(field)
        user_map[field] = st.selectbox(
            f"Select column for **{field}**",
            options,
            index=options.index(default) if default in options else 0
        )

    if st.button("🚀 Run AI Analysis"):

        df = df_raw.rename(columns={
            user_map["id_invoice"]: "id_invoice",
            user_map["client"]: "client",
            user_map["total"]: "total",
            user_map["tax"]: "tax",
            user_map["issuedDate"]: "issuedDate",
            user_map["dueDate"]: "dueDate"
        })

        # -----------------------------------
        # DATE PARSING (ROBUST)
        # -----------------------------------
        df['issuedDate'] = pd.to_datetime(df['issuedDate'], errors='coerce')
        df['dueDate'] = pd.to_datetime(df['dueDate'], errors='coerce')

        # -----------------------------------
        # VALIDATION AGENT
        # -----------------------------------
        df['is_valid'] = True
        df.loc[df['total'] <= 0, 'is_valid'] = False
        expected_gst = df['total'] * 0.18
        df.loc[abs(df['tax'] - expected_gst) > 100, 'is_valid'] = False

        # -----------------------------------
        # DUPLICATE AGENT
        # -----------------------------------
        le = LabelEncoder()
        df['vendor_enc'] = le.fit_transform(df['client'])
        df['duplicate_flag'] = df.duplicated(
            subset=['client', 'id_invoice', 'total'],
            keep=False
        ).astype(int)

        # -----------------------------------
        # AI ANOMALY AGENT
        # -----------------------------------
        model = IsolationForest(contamination=0.1, random_state=42)
        df['anomaly'] = model.fit_predict(df[['total']])
        df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

        # -----------------------------------
        # PAYMENT AGENT
        # -----------------------------------
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

        # -----------------------------------
        # DASHBOARD
        # -----------------------------------
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📄 Total", len(df))
        col2.metric("✅ Valid", df['is_valid'].sum())
        col3.metric("🔁 Duplicates", df['duplicate_flag'].sum())
        col4.metric("🚨 Anomalies", df['anomaly'].sum())
        col5.metric("💰 Pay Now", (df['payment_priority'] == 'HIGH').sum())

        st.divider()

        st.subheader("📊 Payment Priority")
        st.bar_chart(df['payment_priority'].value_counts())

        st.subheader("📋 Invoice Data")
        st.dataframe(
            df[['id_invoice', 'client', 'total',
                'is_valid', 'duplicate_flag',
                'anomaly', 'payment_priority']],
            use_container_width=True
        )

        st.download_button(
            "⬇ Download Processed CSV",
            df.to_csv(index=False).encode(),
            "processed_invoices_output.csv",
            "text/csv"
        )

else:
    st.info("👆 Upload any invoice CSV file to begin")
