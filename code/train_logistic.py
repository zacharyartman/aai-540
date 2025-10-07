import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json

if __name__ == '__main__':
    train_df = pd.read_csv('/opt/ml/input/data/train/train.csv', header=None)
    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join('/opt/ml/model', 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
