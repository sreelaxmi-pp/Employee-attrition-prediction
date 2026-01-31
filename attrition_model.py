# Employee Attrition Prediction using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load Dataset
data = pd.read_csv("data/HR-Employee-Attrition.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# 2. Data Preprocessing

# Drop unnecessary columns
data.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode target variable
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical features
categorical_cols = data.select_dtypes(include='object').columns

le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# 3. Exploratory Data Analysis

plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=data)
plt.title("Attrition Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=data)
plt.title("Monthly Income vs Attrition")
plt.show()

plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 4. Train-Test Split
X = data.drop('Attrition', axis=1)
y = data['Attrition']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

    print(f"\n{name} Performance")
    print(classification_report(y_test, y_pred))

# 6. Results Summary
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# 7. Feature Importance (Random Forest)
rf_model = models["Random Forest"]
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
feature_importance.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.ylabel("Importance Score")
plt.show()
