# 📊 Customer Churn Prediction Dashboard

A complete Machine Learning project that predicts whether a customer is likely to churn (leave a service) using historical data. The project includes data analysis, model building, and an interactive dashboard built with Streamlit.

---

## Project Overview

Customer churn is one of the most critical problems for subscription-based businesses. This project uses machine learning to identify customers who are likely to leave, helping businesses take proactive actions to retain them.

---

## Features

- 📈 Data Cleaning & Preprocessing
- 📊 Exploratory Data Analysis (EDA)
- ⚖️ Handling Imbalanced Data using SMOTE
- 🤖 Machine Learning Model (Random Forest)
- 🎨 Interactive Dashboard using Streamlit
- 📉 Real-time Churn Prediction
- 📊 Insights Dashboard with Visualizations

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib & Seaborn
- Streamlit

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                      # Streamlit dashboard
├── main.py                     # Model training script
├── eda.py                      # EDA and graph generation
│
├── data/
│   └── churn.csv               # Dataset
│
├── model.pkl                   # Trained model
├── columns.pkl                 # Feature columns
├── sample_input.pkl            # Sample input template
│
├── tenure_distribution.png     # EDA graph
├── tenure_vs_churn.png         # EDA graph
├── churn_rate.png              # EDA graph
│
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignore unnecessary files
```
## How It Works

1. Data is cleaned and preprocessed
2. Categorical variables are encoded
3. SMOTE is applied to handle class imbalance
4. A Random Forest model is trained
5. The model predicts churn probability
6. Results are displayed in an interactive Streamlit dashboard

---

## Model Performance

- Accuracy: ~77%
- Improved prediction using Random Forest over Logistic Regression
- Better handling of feature interactions and complex patterns

## Key Insights

- Customers with low tenure are more likely to churn
- Month-to-month contracts have higher churn rates
- Higher monthly charges increase churn probability
- Long-term customers are more loyal

---

## 💻 Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/TwinkleGhosh/customer-churn.git
cd customer-churn

pip install -r requirements.txt

python eda.py

streamlit run app.py
Open in browser:
http://localhost:8501

🎯 Future Improvements
Add more input features in UI
Deploy app on cloud (Streamlit Cloud / Render)
Add feature importance visualization
Improve model with hyperparameter tuning

💼 Why This Project Matters

This project demonstrates:

End-to-end ML pipeline
Real-world problem solving
Model deployment
Data storytelling using dashboards

👤 Author

Twinkle Ghosh

⭐ If you found this project useful, consider giving it a star!
