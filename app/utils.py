import joblib
import pandas as pd

# Load model
model = joblib.load('../model/scam_detector.pkl')

def preprocess(df):
    df = df.fillna('')
    df['text'] = df['title'] + ' ' + df['description']
    return df

def predict(df):
    df = preprocess(df)
    predictions = model.predict(df['text'])
    probabilities = model.predict_proba(df['text'])[:, 1]
    df['Prediction'] = predictions
    df['Fraud Probability'] = probabilities
    return df
