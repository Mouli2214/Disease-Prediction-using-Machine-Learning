# Disease-Prediction-using-Machine-Learning
# Disease Prediction using Machine Learning  Domain: Healthcare | Tech Stack: Python, Scikit-learn, Pandas, NumPy, XGBoost, Matplotlib, Seaborn
# Developed a machine learning model to predict the likelihood of diabetes in patients based on medical parameters such as glucose, BMI, and blood pressure.
# Performed data preprocessing, feature scaling, and correlation analysis on real-world healthcare data from Kaggle.
# Trained multiple models (Random Forest, SVM, XGBoost) and achieved ~85% accuracy on test data.
# Visualized feature importance and correlation heatmaps for better interpretability.
# Built an interactive web interface using Streamlit allowing users to input health data and get instant disease predictions.
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
# Load dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\diabetes.csv")

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Basic info
print("\nDataset Info:")
print(df.info())

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Features and target
X = df.drop(columns="Outcome")
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------
# Model 1: Random Forest
# ---------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n🎯 Random Forest Results 🎯")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# ---------------------
# Model 2: XGBoost
# ---------------------
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("\n🔥 XGBoost Results 🔥")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))

# ---------------------
# Feature Importance
# ---------------------
importances = rf.feature_importances_
features = df.columns[:-1]
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

# ---------------------
# Predict New Data
# ---------------------
sample_data = np.array([[2, 120, 70, 25, 85, 33.6, 0.627, 45]])  # Example values
sample_scaled = scaler.transform(sample_data)
sample_pred = rf.predict(sample_scaled)

print("\n🔍 Sample Prediction (0 = No Disease, 1 = Disease):", sample_pred[0])
