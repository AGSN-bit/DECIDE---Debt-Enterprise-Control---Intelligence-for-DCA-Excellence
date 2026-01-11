import pandas as pd
import numpy as np # Although not explicitly mentioned for train.py, it's good practice for general data handling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model():
    # 1. Load the synthetic dataset
    try:
        df = pd.read_csv('ai/data/synthetic_cases.csv')
        print("Dataset 'synthetic_cases.csv' loaded successfully.")
    except FileNotFoundError:
        print("Error: 'ai/data/synthetic_cases.csv' not found. Please ensure data is generated and saved.")
        return

    # 2. Separate features (X) from target (y)
    X = df.drop(['recovery_outcome', 'case_id'], axis=1)
    y = df['recovery_outcome']
    print("Features (X) and target (y) separated.")

    # 3. Apply one-hot encoding to categorical features
    categorical_cols = ['industry', 'dca_id']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print("Categorical features one-hot encoded.")

    # Ensure consistent columns after one-hot encoding, especially for deployment
    # For this script, we'll capture the final columns of X_train directly.

    # 4. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Store feature names for later use in prediction workflow (important for consistency)
    feature_names = X_train.columns.tolist()

    # 5. Instantiate a Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    print("Logistic Regression model instantiated.")

    # 6. Train the model
    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")

    # 7. Save the trained model and feature names
    model_path = 'ai/recovery_prediction_model.joblib'
    joblib.dump(model, model_path)
    print(f"Trained model saved to '{model_path}'")

    # Save feature names to ensure consistency when loading the model for inference
    feature_names_path = 'ai/model_feature_names.joblib'
    joblib.dump(feature_names, feature_names_path)
    print(f"Model feature names saved to '{feature_names_path}'")

if __name__ == '__main__':
    train_model()
