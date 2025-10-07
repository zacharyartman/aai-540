import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    
    input_data_path = '/opt/ml/processing/input'
    df = pd.read_csv(f'{input_data_path}/raw_marketing_data.csv')
    
    target_col = 'Conversion'
    categorical_features = ['Gender', 'CampaignChannel', 'CampaignType', 
                           'AdvertisingPlatform', 'AdvertisingTool']
    numeric_features = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
                       'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
                       'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints']
    
    df = df.drop('CustomerID', axis=1)
    
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data: 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Save processed data (target as first column)
    train_data = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    val_data = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1)
    test_data = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
    
    train_data.to_csv('/opt/ml/processing/train/train.csv', header=False, index=False)
    val_data.to_csv('/opt/ml/processing/validation/validation.csv', header=False, index=False)
    test_data.to_csv('/opt/ml/processing/test/test.csv', header=False, index=False)
    
    # Save preprocessing objects
    joblib.dump(scaler, '/opt/ml/processing/model/scaler.joblib')
    joblib.dump(le_dict, '/opt/ml/processing/model/label_encoders.joblib')
    
    print(f'Preprocessing complete')
    print(f'Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}')
