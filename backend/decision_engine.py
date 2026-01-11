import pandas as pd

# Import components
from ai.model import load_model_components, predict_recovery_probability
from ai.explainability import decide_action
from backend.sla_engine import check_sla_risk
from backend.audit import initialize_audit_log, append_audit_log_entry

# Global variables for the audit log and model components
audit_log_df = initialize_audit_log()
model, feature_names = load_model_components()

def simulate_decision_workflow(case: pd.Series):
    """
    Simulates the AI decision engine workflow for a single debt collection case,
    integrating prediction, explainability, SLA checks, and audit logging.

    Args:
        case (pd.Series): A single row from the DataFrame representing a case.

    Returns:
        dict: The action, confidence, and reasoning from the decide_action function.
    """
    global audit_log_df, model, feature_names

    # 1. Predict recovery probability
    recovery_probability = predict_recovery_probability(model, feature_names, case)

    # 2. Determine SLA risk
    sla_risk = check_sla_risk(case['sla_days_remaining'])

    # 3. Call decide_action for AI recommendation
    decision = decide_action(
        recovery_probability=recovery_probability,
        sla_risk=sla_risk,
        dca_performance_score=case['dca_performance_score']
    )

    # 4. Log the decision to audit_log_df
    audit_log_df = append_audit_log_entry(
        audit_df=audit_log_df,
        case_id=case['case_id'],
        ai_recommendation=decision, # Store the full decision dict
        decision_type='AI',
        action_taken=decision['action']
    )

    return decision

print("backend/decision_engine.py created successfully with simulate_decision_workflow function.")
