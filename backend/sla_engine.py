def check_sla_risk(sla_days_remaining):
    """
    Checks if an SLA breach is imminent or has occurred.

    Args:
        sla_days_remaining (int): Number of days remaining until SLA breach.

    Returns:
        bool: True if SLA breach is imminent (<=7 days) or occurred (<=0 days), False otherwise.
    """
    return sla_days_remaining <= 7

print("backend/sla_engine.py created successfully with check_sla_risk function.")
