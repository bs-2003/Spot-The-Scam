import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import predict

st.set_page_config(layout="wide")
st.title("🕵️ Spot the Scam - Job Fraud Detection Tool")

uploaded_file = st.file_uploader("Upload Job Postings CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    with st.spinner("Analyzing..."):
        result_df = predict(df)
    
    st.subheader("🔍 Prediction Results")
    st.dataframe(result_df[['title', 'location', 'Prediction', 'Fraud Probability']])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Fraud Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(result_df['Fraud Probability'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("🧮 Fraud vs Genuine Pie Chart")
        labels = ['Genuine', 'Fraud']
        sizes = result_df['Prediction'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=["#66bb6a", "#ef5350"])
        st.pyplot(fig)
    
    st.subheader("⚠️ Top 10 Most Suspicious Jobs")
    st.table(result_df.sort_values('Fraud Probability', ascending=False).head(10)[['title', 'location', 'Fraud Probability']])
