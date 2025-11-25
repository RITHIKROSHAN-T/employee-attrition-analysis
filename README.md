# 🚀 Employee Attrition Analysis and Prediction Dashboard

---

## 📘 Project Overview

This project delivers a robust **Machine Learning** solution to address a critical HR challenge: **predicting and preventing employee turnover.**

The solution is a comprehensive analysis and a predictive model deployed through an interactive **Streamlit Dashboard**. This allows HR managers to forecast an employee's risk of leaving in real-time.

### Business Value:
* **Proactive Retention:** The model identifies high-risk employees early, enabling management to implement timely and targeted retention strategies.
* **Cost Savings:** Directly minimizes the significant financial and operational costs associated with recruitment, onboarding, and training new staff.

---

## 🎯 Key Results and Model Quality

The final model is a **Random Forest Classifier** optimized for maximizing the identification of employees who will leave (Recall).

### Key Performance Summary:
* **Catch Rate (Recall):** The model achieves **68% Recall**, meaning it successfully flags **68 out of every 100 employees** who are truly going to leave.
* **Overall Accuracy:** The model is correct in its predictions $\mathbf{75.85\%}$ of the time overall.
* **Prediction Rule:** The model uses a low custom threshold of **0.35** to ensure maximum coverage of high-risk cases.

**Quality Rationale:**
The model's configuration was chosen for its **reliability** ($\text{Precision} = 36\%$) and high catch rate. This provides the best balance for a business solution, ensuring management can act on a majority of the critical attrition cases.

---

## 🧠 Tools & Technologies Used

| **Category** | **Tools** |
|---|---|
| Programming | Python (Pandas, NumPy, Scikit-learn, Pickle) |
| Web Application | Streamlit |
| Machine Learning | Random Forest Classifier, Feature Engineering |
| Domain | HR Analytics |

---

## 🔍 Key Analysis & ML Steps

### 🧹 Data Preprocessing & Feature Engineering
- Extensive data cleaning and handling of missing values.
- Creation of highly predictive features such as the **Experience Gap** ($\text{TotalWorkingYears} - \text{YearsAtCompany}$) and **Satisfaction Index** from multiple satisfaction scores.

### 📈 Dashboard Functionality
- **Interactive Prediction Tool:** Real-time risk scoring based on user inputs.
- **HR Insights Dashboard:** Visual summaries of high-risk employees and key attrition drivers.

---
