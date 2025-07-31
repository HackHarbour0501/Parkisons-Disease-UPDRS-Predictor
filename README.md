# 🧠 Parkinson's Disease UPDRS Score Predictor

A Streamlit web application that predicts the **Total UPDRS (Unified Parkinson’s Disease Rating Scale)** score based on patient vocal and demographic features using a **Random Forest Regressor**.

---

## 🚀 Features

- Predict the total UPDRS score using clinical voice and demographic data.
- Classifies patient status into:
  - ✅ Normal
  - ⚠️ At Risk
  - ❗ High Risk
- Visualize:
  - Feature importance from Random Forest model
  - Correlation heatmap of all features
- Interactive sliders for user input
- Real-time model evaluation metrics

---

## 📁 Dataset

- The dataset used is `Parkinsons.csv`, which includes:
  - Demographics: age, sex
  - Acoustic features: jitter, shimmer, HNR, RPDE, DFA, PPE
  - Target: total UPDRS

---

## 🧪 Model Details

- **Algorithm**: Random Forest Regressor
- **Train-Test Split**: 80/20
- **Performance Metrics**:
  - R² Score
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

---

## 📦 Installation

Make sure you have Python 3.7+ installed.

1. **Clone this repo** or copy the code into a directory.
2. Place the `Parkinsons.csv` file in the same directory.
3. Install required packages:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn

