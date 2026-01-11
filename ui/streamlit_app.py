import streamlit as st
import pandas as pd
import numpy as np # Required for np.mean

# Import functions and variables from created modules
from ai.model import load_model_components, predict_recovery_probability
from ai.explainability import decide_action
from backend.sla_engine import check_sla_risk
from backend.audit import initialize_audit_log, append_audit_log_entry
# Import the global audit_log_df and model components from decision_engine
from backend.decision_engine import audit_log_df, model, feature_names, simulate_decision_workflow

# --- Initialization (using Streamlit's caching for efficiency) ---

@st.cache_resource
def load_data():
    """
    Loads the synthetic dataset from CSV.
    """
    try:
        df = pd.read_csv('ai/data/synthetic_cases.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'ai/data/synthetic_cases.csv' not found. Please ensure data is generated and saved.")
        st.stop()

df = load_data()

# --- KPI Calculation ---
total_cases = len(df)
high_risk_cases = df[df['customer_risk_score'] > 70].shape[0]

# Calculate predicted recovery probability for all cases for overall KPI
# Preprocess the entire DataFrame for prediction
X_pred = df.drop(['case_id', 'recovery_outcome'], axis=1, errors='ignore')

# One-hot encode categorical features ('industry', 'dca_id')
categorical_cols = ['industry', 'dca_id']
X_pred_encoded = pd.get_dummies(X_pred, columns=categorical_cols)

# Align columns with feature_names (from the loaded model)
missing_cols = set(feature_names) - set(X_pred_encoded.columns)
for c in missing_cols:
    X_pred_encoded[c] = False # Use False for boolean dtypes from get_dummies

extra_cols = set(X_pred_encoded.columns) - set(feature_names)
X_pred_encoded = X_pred_encoded.drop(columns=list(extra_cols))

# Ensure the order of columns is the same as in feature_names
X_pred_final = X_pred_encoded[feature_names]

# Predict recovery probabilities
all_recovery_probabilities = model.predict_proba(X_pred_final)[:, 1]

overall_predicted_recovery_rate = np.mean(all_recovery_probabilities)

# --- UI Functions ---

def display_kpis():
    st.header("Debt Collection Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Cases", value=f"{total_cases}")
    with col2:
        st.metric(label="High-Risk Cases (CS > 70)", value=f"{high_risk_cases}")
    with col3:
        st.metric(label="Overall Predicted Recovery Rate", value=f"{overall_predicted_recovery_rate:.2f}")
    st.markdown("--- ")

def display_case_details(case_id):
    st.subheader(f"Case Details for ID: {case_id}")

    selected_case = df[df['case_id'] == case_id].iloc[0].copy()

    # Run the simulation workflow for the selected case
    # This call will also update the global audit_log_df in backend.decision_engine
    ai_recommendation = simulate_decision_workflow(selected_case)

    st.write("#### Original Case Information")
    st.dataframe(selected_case.to_frame().T)

    st.write("#### AI Recommendation and Explanation")
    st.info(f"**Action:** {ai_recommendation['action']}\n\n" +
            f"**Confidence:** {ai_recommendation['confidence']:.2f}\n\n" +
            f"**Reasoning:** {ai_recommendation['reasoning']}")
    st.markdown("--- ")

def display_audit_log():
    st.subheader("Audit Log")
    if not audit_log_df.empty:
        # Convert AI_recommendation dictionary to a readable string for display
        audit_log_display = audit_log_df.copy()
        audit_log_display['AI_recommendation'] = audit_log_display['AI_recommendation'].apply(lambda x: f"Action: {x['action']}, Confidence: {x['confidence']:.2f}, Reasoning: {x['reasoning']}")
        st.dataframe(audit_log_display)
    else:
        st.write("No audit log entries yet.")
    st.markdown("--- ")

# --- Main Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="DECIDE Prototype")
st.title("Debt Collection AI Decision Engine (DECIDE) Prototype")

display_kpis()

# Case selection
case_ids = df['case_id'].tolist()
selected_case_id = st.selectbox("Select a Case ID to view details and AI Recommendation:", options=case_ids)

if selected_case_id:
    display_case_details(selected_case_id)

display_audit_log()

print("ui/streamlit_app.py created successfully.")
