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
st.caption("Schema-agnostic AI system for any invoice CSV")

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

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------
    df_raw = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully")

    detected_map = auto_map_columns(df_raw)

    st.subheader("🔍 Column Mapping")
    st.info("Auto-detected columns. Please confirm or change if needed.")

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

        # -----------------------------------
        # RENAME COLUMNS (SAFE)
        # -----------------------------------
        df = df_raw.rename(columns={
            user_map["id_invoice"]: "id_invoice",
            user_map["client"]: "client",
            user_map["total"]: "total",
            user_map["tax"]: "tax",
            user_map["issuedDate"]: "issuedDate",
            user_map["dueDate"]: "dueDate"
        })

        # -----------------------------------
        # CRITICAL FIX: REMOVE DUPLICATE COLUMNS
        # -----------------------------------
        df.columns = df.columns.astype(str)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(axis=1, how='all')

        # -----------------------------------
        # SAFE DATE PARSING
        # -----------------------------------
        if "issuedDate" in df.columns:
            df["issuedDate"] = pd.to_datetime(df["issuedDate"], errors="coerce")

        if "dueDate" in df.columns:
            df["dueDate"] = pd.to_datetime(df["dueDate"], errors="coerce")

        # -----------------------------------
        # INVOICE VALIDATION AGENT
        # -----------------------------------
        df["is_valid"] = True
        df.loc[df["total"] <= 0, "is_valid"] = False

        expected_gst = df["total"] * 0.18
        df.loc[abs(df["tax"] - expected_gst) > 100, "is_valid"] = False

        # -----------------------------------
        # DUPLICATE DETECTION AGENT
        # -----------------------------------
        le = LabelEncoder()
        df["vendor_enc"] = le.fit_transform(df["client"].astype(str))

        df["duplicate_flag"] = df.duplicated(
            subset=["client", "id_invoice", "total"],
            keep=False
        ).astype(int)

        # -----------------------------------
        # AI ANOMALY DETECTION AGENT
        # -----------------------------------
        model = IsolationForest(contamination=0.1, random_state=42)
        df["anomaly"] = model.fit_predict(df[["total"]])
        df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

        # -----------------------------------
        # PAYMENT SCHEDULING AGENT
        # -----------------------------------
        today = datetime.today()
        df["days_to_due"] = (df["dueDate"] - today).dt.days

        df["payment_priority"] = np.where(
            (df["days_to_due"] <= 7) &
            (df["anomaly"] == 0) &
            (df["duplicate_flag"] == 0) &
            (df["is_valid"] == True),
            "HIGH",
            "HOLD"
        )

        # -----------------------------------
        # DASHBOARD METRICS
        # -----------------------------------
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("📄 Total Invoices", len(df))
        col2.metric("✅ Valid", int(df["is_valid"].sum()))
        col3.metric("🔁 Duplicates", int(df["duplicate_flag"].sum()))
        col4.metric("🚨 Anomalies", int(df["anomaly"].sum()))
        col5.metric("💰 Ready for Payment", int((df["payment_priority"] == "HIGH").sum()))

        st.divider()

        # -----------------------------------
        # CHARTS
        # -----------------------------------
        colA, colB = st.columns(2)

        with colA:
            st.subheader("🚨 Anomaly Distribution")
            st.bar_chart(df["anomaly"].value_counts())

        with colB:
            st.subheader("💰 Payment Priority")
            st.bar_chart(df["payment_priority"].value_counts())

        # -----------------------------------
        # DATA TABLE
        # -----------------------------------
        st.subheader("📋 Processed Invoice Data")
        st.dataframe(
            df[[
                "id_invoice",
                "client",
                "total",
                "is_valid",
                "duplicate_flag",
                "anomaly",
                "payment_priority"
            ]],
            use_container_width=True
        )

        # -----------------------------------
        # DOWNLOAD
        # -----------------------------------
        st.download_button(
            "⬇ Download Processed CSV",
            df.to_csv(index=False).encode("utf-8"),
            "processed_invoices_output.csv",
            "text/csv"
        )

else:
    st.info("👆 Upload any invoice CSV file to begin analysis")
