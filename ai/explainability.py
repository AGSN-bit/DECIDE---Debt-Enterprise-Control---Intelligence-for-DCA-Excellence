def decide_action(recovery_probability, sla_risk, dca_performance_score):
    """
    Determines the appropriate action for a debt collection case based on recovery probability,
    SLA risk, and DCA performance, with explainable reasoning.

    Args:
        recovery_probability (float): Probability of successful debt recovery (0 to 1).
        sla_risk (bool): True if SLA breach is imminent or occurred, False otherwise.
        dca_performance_score (float): Performance score of the assigned Debt Collection Agency.

    Returns:
        dict: A dictionary containing the 'action', 'confidence', and 'reasoning'.
    """
    if sla_risk:
        action = 'Escalate immediately'
        confidence = 0.95
        reasoning = f'SLA breach imminent or occurred (SLA Risk: {sla_risk}), requiring urgent intervention.'
    elif recovery_probability < 0.3 and dca_performance_score < 0.7:
        action = 'Reassign to better DCA'
        confidence = 0.85
        reasoning = (f'Low recovery probability ({recovery_probability:.2f}) combined with ')
        reasoning += (f'an underperforming DCA ({dca_performance_score:.2f}) suggests reassigning the case.')
    elif recovery_probability < 0.5:
        action = 'Continue monitoring'
        confidence = 0.7
        reasoning = (f'Recovery probability is moderate ({recovery_probability:.2f}), requiring ')
        reasoning += 'continued observation to assess potential changes.'
    else:
        action = 'Continue monitoring'
        confidence = 0.9
        reasoning = (f'Recovery probability is good ({recovery_probability:.2f}), ')
        reasoning += 'continue with current collection efforts.'

    return {
        'action': action,
        'confidence': confidence,
        'reasoning': reasoning
    }

print("ai/explainability.py created successfully with decide_action function.")
