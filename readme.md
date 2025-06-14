# 🕵️ Spot the Scam

Detect fraudulent job postings using machine learning and visualize insights in a Streamlit dashboard.

## Features
- Binary classifier for fraud detection
- Upload your own job posting CSV
- Visual analytics (probabilities, pie chart, top suspicious jobs)

## Setup

```bash
pip install -r requirements.txt
cd model && python train_model.py
streamlit run app/dashboard.py
