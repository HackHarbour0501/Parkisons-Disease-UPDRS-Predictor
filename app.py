import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Parkinsons.csv')

# Preprocess
X = df.drop(columns=['subject', 'test_time', 'motor_updrs', 'total_updrs'])
y = df['total_updrs']
X['sex'] = X['sex'].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# 🧪 Prediction function
def predict_updrs(age, sex, jitter, shimmer, hnr, rpde, dfa, ppe):
    input_data = pd.DataFrame([{
        'age': age,
        'sex': int(sex),
        'jitter': jitter,
        'shimmer': shimmer,
        'hnr': hnr,
        'rpde': rpde,
        'dfa': dfa,
        'ppe': ppe,
        **{col: X[col].mean() for col in X.columns if col not in ['age', 'sex', 'jitter', 'shimmer', 'hnr', 'rpde', 'dfa', 'ppe']}
    }])[X.columns]

    return rf_model.predict(input_data)[0]

# 🌐 Streamlit UI
st.set_page_config(page_title="Parkinson's UPDRS Predictor", layout="wide")

st.title("🧠 Parkinson's Disease UPDRS Score Predictor")

st.sidebar.header("Enter Patient Data:")
age = st.sidebar.slider("Age", 30, 90, 60)
sex = st.sidebar.radio("Sex", ("Male", "Female"))
sex = 1 if sex == "Male" else 0
jitter = st.sidebar.slider("Jitter", 0.0001, 0.02, 0.01)
shimmer = st.sidebar.slider("Shimmer", 0.01, 0.2, 0.1)
hnr = st.sidebar.slider("HNR (Harmonics-to-Noise Ratio)", 5.0, 40.0, 20.0)
rpde = st.sidebar.slider("RPDE", 0.1, 0.6, 0.3)
dfa = st.sidebar.slider("DFA", 0.4, 1.0, 0.6)
ppe = st.sidebar.slider("PPE", 0.0, 0.7, 0.3)

if st.sidebar.button("Predict UPDRS Score"):
    prediction = predict_updrs(age, sex, jitter, shimmer, hnr, rpde, dfa, ppe)
    st.success(f"📣 Predicted Total UPDRS Score: **{prediction:.2f}**")

    # Risk interpretation
    if prediction < 20:
        status = "✅ Normal"
        msg = "No immediate signs of motor impairment. Maintain regular monitoring."
        color = "green"
    elif prediction < 35:
        status = "⚠️ At Risk"
        msg = "Some motor symptoms are evident. Consider consulting a neurologist."
        color = "orange"
    else:
        status = "❗ High Risk"
        msg = "Significant motor impairment. Immediate medical attention is advised."
        color = "red"

    st.markdown(f"### 🩺 Health Status: **{status}**")
    st.info(msg)


# 📊 Metrics and Feature Importance
st.subheader("📈 Model Performance ")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2_score(y_test, y_pred_rf):.2f}")
col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred_rf):.2f}")
col3.metric("RMSE", f"{mean_squared_error(y_test, y_pred_rf):.2f}")



# 🔥 Optional: Correlation Heatmap
with st.expander("📌 Show Feature Correlation Heatmap"):
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax2)
    ax2.set_title("Feature Correlation Heatmap")
    st.pyplot(fig2)
