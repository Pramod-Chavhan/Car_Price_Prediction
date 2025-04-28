# 🚗✨ Car Price Prediction using Linear & Lasso Regression

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 📌 Overview

Welcome to the **Car Price Prediction** project!  
This machine learning project uses **Linear Regression** and **Lasso Regression** models to predict a car's selling price based on various features.

🛠️ Features:
- Data preprocessing & feature engineering
- Model building (Linear & Lasso Regression)
- Evaluation with R² Score, MAE, MSE, RMSE
- Model saving for production use

---

## 💡 Demo Highlights

- 🌟 Accurate price prediction models
- 📊 Beautiful, clear visualizations
- 📃 Production-ready model export

---

## 📈 Tech Stack

| Tool | Purpose |
| :--- | :------ |
| **Python** | Programming Language |
| **Pandas, NumPy** | Data Processing |
| **Matplotlib, Seaborn** | Data Visualization |
| **Scikit-Learn** | Machine Learning |
| **Joblib** | Model Serialization |

---

## 🔍 Project Structure

```bash
├── car_data.csv
├── car_price_prediction.ipynb
├── linear_regression_model.pkl
├── lasso_regression_model.pkl
├── scaler.pkl
├── assets/
│   ├── selling_price_distribution.png
│   ├── actual_vs_predicted_train.png
│   ├── actual_vs_predicted_test.png
├── README.md
├── requirements.txt
```

---

## 📅 Workflow

1. **Load Dataset**
2. **Data Preprocessing**
   - Encoding categorical features
   - Feature Scaling with StandardScaler
3. **Train-Test Split**
4. **Model Training**
5. **Model Evaluation**
6. **Model Saving**

---

## 📊 Visualizations

- **Selling Price Distribution**
- **Actual vs Predicted Prices (Train/Test Set)**
- **Model Performance Comparison**

*(Stored inside the `assets/` folder.)*

---

## 💡 Model Performance

| Metric | Linear Regression | Lasso Regression |
| :----- | :---------------- | :--------------- |
| Train R² Score | ~0.94 | ~0.93 |
| Test R² Score | ~0.92 | ~0.91 |
| RMSE | Low | Low |

---

## 🚀 How to Run

Clone this repository:

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
jupyter notebook car_price_prediction.ipynb
```

---

## 📁 Model Loading Example

```python
import joblib

# Load models
lin_reg_model = joblib.load('linear_regression_model.pkl')
lasso_reg_model = joblib.load('lasso_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Predict
scaled_features = scaler.transform(your_features)
predicted_price = lin_reg_model.predict(scaled_features)
```

---

## 🔬 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Deploy web app using Flask/FastAPI
- Try Ensemble models (Random Forest, XGBoost)

---

## 👤 About Developer

**Author:** Pramod  
📧 Email: pramodchavhanm@gmail.com  
🔗 LinkedIn: [Your LinkedIn Profile](www.linkedin.com/in/pramodchavhan)

---

## 🌟 Contributing

Contributions are always welcome!  
Fork the project and submit a Pull Request. 🚀

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

# 📢 If you liked this project, please give it a star ⭐!

