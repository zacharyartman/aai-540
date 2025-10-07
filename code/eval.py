import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tarfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    args, _ = parser.parse_known_args()
    
    model_path = '/opt/ml/processing/model/model.tar.gz'
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall('/opt/ml/processing/model/')
    
    model = joblib.load('/opt/ml/processing/model/model.joblib')
    
    test_df = pd.read_csv('/opt/ml/processing/test/test.csv', header=None)
    X_test = test_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'model_name': args.model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    }
    
    output_dir = '/opt/ml/processing/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    report = classification_report(y_test, y_pred)
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(report)
