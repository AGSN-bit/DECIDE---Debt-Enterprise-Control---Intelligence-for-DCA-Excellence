import pandas as pd
from datetime import datetime

def initialize_audit_log():
    """
    Initializes an empty pandas DataFrame for audit logging.

    Returns:
        pd.DataFrame: An empty DataFrame with predefined columns for the audit log.
    """
    columns = ['timestamp', 'case_id', 'AI_recommendation', 'decision_type', 'action_taken']
    return pd.DataFrame(columns=columns)

def append_audit_log_entry(audit_df, case_id, ai_recommendation, decision_type, action_taken):
    """
    Appends a new entry to the audit log DataFrame.

    Args:
        audit_df (pd.DataFrame): The current audit log DataFrame.
        case_id (int): The ID of the case being audited.
        ai_recommendation (dict): The full AI recommendation dictionary.
        decision_type (str): The type of decision (e.g., 'AI', 'Human').
        action_taken (str): The final action taken.

    Returns:
        pd.DataFrame: The updated audit log DataFrame with the new entry.
    """
    new_log_entry = {
        'timestamp': datetime.now(),
        'case_id': case_id,
        'AI_recommendation': ai_recommendation,
        'decision_type': decision_type,
        'action_taken': action_taken
    }

    # Use pd.concat for appending as it's the recommended way for DataFrames
    # Handle initial append to an empty DataFrame to avoid FutureWarning
    if audit_df.empty and not new_log_entry:
        return audit_df # Return empty if no entry to append to an empty DF
    elif audit_df.empty:
        return pd.DataFrame([new_log_entry])
    else:
        return pd.concat([audit_df, pd.DataFrame([new_log_entry])], ignore_index=True)

print("backend/audit.py created successfully with initialize_audit_log and append_audit_log_entry functions.")
