import requests
import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide"
)

# FastAPI backend endpoint
FASTAPI_URL = "http://127.0.0.1:8000"

st.title("💳 Real-Time Credit Card Fraud Detection System")
st.write("Adjust transaction details below to calculate live fraud risk.")

# Input Layout - Split into 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0
    )
    transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
    merchant_category = st.selectbox(
        "Merchant Category",
        ["Electronics", "Travel", "Grocery", "Entertainment", "Online Shopping"],
    )

with col2:
    foreign_transaction = st.selectbox(
        "Is Foreign Transaction?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    location_mismatch = st.selectbox(
        "Location Mismatch Flag?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    device_trust_score = st.slider("Device Trust Score (0-100)", 0, 100, 75)

with col3:
    velocity_last_24h = st.number_input(
        "Transactions in Last 24 Hours", min_value=0, value=2, step=1
    )
    cardholder_age = st.slider("Cardholder Age", 18, 100, 35)

st.markdown("---")

# Predict button
if st.button("Evaluate Fraud Risk", type="primary"):
    payload = {
        "amount": amount,
        "transaction_hour": transaction_hour,
        "merchant_category": merchant_category,
        "foreign_transaction": foreign_transaction,
        "location_mismatch": location_mismatch,
        "device_trust_score": float(device_trust_score),
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
    }

    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict", json=payload, timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            proba = result["fraud_probability"] * 100
            risk_level = result["risk_level"]

            st.subheader("Assessment Results")

            m1, m2 = st.columns(2)
            m1.metric("Fraud Probability Score", f"{proba:.2f}%")
            m2.metric("Risk Classification", risk_level)

            if result["risk_level"] == "HIGH RISK":
                st.error("🚨 Flagged as High Fraud Risk!")
            elif result["risk_level"] == "MEDIUM RISK":
                st.warning("⚠️ Elevated Risk Level. Additional verification recommended.")
            else:
                st.success("✅ Low Risk. Transaction appears legitimate.")
        else:
            # Displays the exact error detail from FastAPI response
            error_msg = response.json().get("detail", "Unknown server error")
            st.error(f"❌ API Error ({response.status_code}): {error_msg}")

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Connection Refused: FastAPI server is not running on http://127.0.0.1:8000. "
            "Please start FastAPI in a terminal using: `uvicorn main:app --reload`"
        )
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")