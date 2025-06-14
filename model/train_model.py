import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Load data
df = pd.read_csv('../data/train.csv')
df = df.dropna(subset=['title', 'description'])  # drop rows with missing text

# Combine text fields
df['text'] = df['title'] + ' ' + df['description']
X = df['text']
y = df['fraudulent']

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Balance classes using SMOTE
X_vec = tfidf.fit_transform(X)
X_res, y_res = SMOTE().fit_resample(X_vec, y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# Save pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', model)
])

joblib.dump(pipeline, '../model/scam_detector.pkl')
print("Model saved!")
