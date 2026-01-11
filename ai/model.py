import joblib
import pandas as pd

def load_model_components():
    """
    Loads the trained machine learning model and its feature names.

    Returns:
        tuple: A tuple containing the trained model and the list of feature names.
    """
    model = joblib.load('ai/recovery_prediction_model.joblib')
    feature_names = joblib.load('ai/model_feature_names.joblib')
    return model, feature_names

def predict_recovery_probability(model, feature_names, input_data):
    """
    Predicts the recovery probability for a given input case.

    Args:
        model: The trained machine learning model.
        feature_names (list): List of feature names the model was trained on.
        input_data (pd.Series or pd.DataFrame): The preprocessed input features for a single case.

    Returns:
        float: The predicted probability of recovery.
    """
    # Ensure input_data is a DataFrame for consistent processing
    if isinstance(input_data, pd.Series):
        input_df = input_data.to_frame().T
    else:
        input_df = input_data.copy()

    # Drop 'case_id' and 'recovery_outcome' if present, as they are not features for prediction
    input_df = input_df.drop(columns=['case_id', 'recovery_outcome'], errors='ignore')

    # One-hot encode categorical features
    categorical_cols = ['industry', 'dca_id']
    # Check which categorical columns are actually in the input_df before applying get_dummies
    cols_to_encode = [col for col in categorical_cols if col in input_df.columns]
    input_encoded = pd.get_dummies(input_df, columns=cols_to_encode)

    # Align columns with the model's expected feature names
    missing_cols = set(feature_names) - set(input_encoded.columns)
    for c in missing_cols:
        input_encoded[c] = False # Use False for boolean dtypes from get_dummies

    extra_cols = set(input_encoded.columns) - set(feature_names)
    input_encoded = input_encoded.drop(columns=list(extra_cols))

    # Ensure the order of columns is the same as the model's training features
    final_features = input_encoded[feature_names]

    # Predict recovery probability (probability of the positive class, which is 1)
    recovery_probability = model.predict_proba(final_features)[:, 1][0]

    return recovery_probability

print("ai/model.py created successfully with load_model_components and predict_recovery_probability functions.")
