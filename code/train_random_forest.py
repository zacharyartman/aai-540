import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == '__main__':
    train_df = pd.read_csv('/opt/ml/input/data/train/train.csv', header=None)
    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    
    print("Training Random Forest")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    model_path = os.path.join('/opt/ml/model', 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Important Features:")
    print(feature_importance.head(10))
