# 🌦 Weather Prediction ML System

A full-stack Machine Learning project that predicts whether it will rain tomorrow using historical weather data and real-time user inputs.

---

## 🚀 Project Overview

This project uses classification algorithms to predict **RainTomorrow (Yes/No)** based on weather features.

It includes:
- Data preprocessing pipeline
- Machine Learning model training
- Model evaluation
- Flask-based web application
- Real-time prediction UI

---

## 🧠 ML Models Used

### 1. Logistic Regression
- Training Accuracy: **84.90%**
- Test Accuracy: **84.96%**
- Iterations: **1000**

🔻 Increasing iterations:

- Training Accuracy: **79.29%**
- Test Accuracy: **78.79%**
- Iterations: **10000**

👉 Observation:
- Increasing iterations **decreased performance**
- Likely due to **overfitting / optimization instability**

---

### 2. Random Forest

- Test Accuracy: ~**84%**
- High precision for "No Rain"
- Lower recall for "Rain"

👉 Observation:
- Handles non-linearity well
- But struggles with **class imbalance**

---

### 3. XGBoost

- Validation Accuracy: ~**82–83%**
- Good recall for "Rain"
- Test instability observed

👉 Observation:
- Strong model but requires **careful tuning**
- Sensitive to data imbalance

---

## 📊 Model Comparison

| Model               | Accuracy  | Precision      | Recall (Rain) | Stability | Notes             |
|---------------------|-----------|----------------|---------------|-----------|-------------------|
| Logistic Regression | ⭐ 84.9% | Good            | Moderate      | High      | Best overall      |
| Random Forest       | 84%       | High (No Rain) | Low           | Medium    | Misses rain cases  |
| XGBoost             | 82–83%    | Balanced       | Better Recall | Medium    | Needs tuning       |

---

## 🏆 Final Model Choice

👉 **Logistic Regression**

Reason:
- Best accuracy
- Stable performance
- Simpler and faster
- Works well with current features

---

## ⚙️ Tech Stack

- Python
- Flask
- Pandas, NumPy
- Scikit-learn
- XGBoost
- HTML, CSS (Glassmorphism UI)

---

## ⚡ Features

- Real-time prediction via UI
- Auto-fill weather data using API
- Clean modern frontend (glass UI)
- End-to-end ML pipeline

---

## 📁 Project Structure

weathercast-ml-system-full/
│
├── 📁 app/
│   ├── app.py
│   └──📁 templates/
│       └── index.html
│
├── 📁 data/
│   ├── raw/    ---- Dataset sourced from: Kaggle (Australia Rain Prediction Dataset) 
│   │    └── weatherAUS.csv          
│   ├── processed/
│   │   ├── train.csv 
│   │   ├── validation.csv
│   │   └── test.csv
│   └── README.md     
│
├── 📁 notebooks/
│   └── weatherSystem.ipynb      
│
├── 📁 screenshots/
│   ├── homepage.png
│   ├── prediction_result_not_rain.png
│   └── prediction_result_rain.png    
│
├── 📁 src/
│    ├── __init__.py
│    ├── data_loader.py
│    ├── evaluate.py
│    ├── main.py
│    ├── predict.py
│    ├── preprocess.py
│    ├── split.py
│    ├── train.py
│    ├── evaluate.py
│    └── utlis.py
│
│   
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
└── 📄 LICENSE

---

## 🌐 How to Run Locally

```bash
pip install -r requirements.txt
python app/app.py

Then Open
http://127.0.0.1:5000

$Future Improvements
---- Use weather forecast API instead of current data
---- Improve feature engineering
---- Handle class imbalance better (SMOTE / tuning)
---- Add location-based prediction
---- Deploy with CI/CD pipeline

$Key Learnings
---- Difference between models (LR vs RF vs XGBoost)
---- Importance of preprocessing consistency
---- Handling real-world deployment issues
---- Building full ML system (not just model)

## 🌐 Live Demo
https://weathercast-ml-systems.onrender.com



👨‍💻Author
Rupak Kumar
B.Tech ECE | ML Enthusiast | BS DATA SCIENCE