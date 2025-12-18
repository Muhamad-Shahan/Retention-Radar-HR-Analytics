# 👔 Retention Radar: Employee Flight Risk Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Logistic_Regression-green.svg)](https://scikit-learn.org/)

## 📋 Executive Summary
Employee turnover is costly, affecting productivity and morale. Traditional HR relies on exit interviews (reactive), but **Retention Radar** enables a **proactive** approach.

This project is a **People Analytics Dashboard** that predicts the probability of an employee leaving. By analyzing 15,000 employee records, the system identifies key turnover drivers (Satisfaction, Tenure, Overwork) and provides an interactive "What-If" simulator for HR managers to test retention strategies.

> **[🔴 Launch the Simulator](https://retention-radar-hr-analytics-krlanbjseppxgxfmsvpvuy.streamlit.app/)**

## 🛠️ Technical Architecture
The solution solves the "Imbalanced Class" problem common in HR data (where most people stay, making it hard to catch those who leave).

### 1. Methodology
* **Algorithm:** Logistic Regression (optimized for interpretability).
* **Class Balancing:** Utilized `class_weight='balanced'` to penalize false negatives, achieving an **80% Recall Rate** for detecting flight risks.
* **Preprocessing:**
    * **One-Hot Encoding:** For categorical departments (Sales, IT, HR).
    * **Standard Scaling:** Normalized numerical features (Hours vs. Satisfaction) to ensure fair model weighting.

### 2. Key Insights
The model identified a non-linear relationship between workload and turnover:
* **The Burnout Zone:** Employees working >250 hours/month are high risk.
* **The Boreout Zone:** Employees working <130 hours/month are also high risk.
* **The Sweet Spot:** 150-200 hours/month correlates with highest retention.

## 📊 Model Performance
* **Recall (Flight Risk):** 0.80 (Captures 80% of employees likely to leave).
* **Accuracy:** 77% (Balanced against the need to minimize false negatives).

## 💻 Installation & Usage

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone [https://github.com/Muhammad-Shahan/Retention-Radar-HR-Analytics.git](https://github.com/Muhammad-Shahan/Retention-Radar-HR-Analytics.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate Models (First Run Only)
python setup_model.py

# 4. Run the Dashboard
streamlit run app.py
