# Roll No: 24BAD074 Mohd Shafique RB
# Experiment 5 – Decision Tree Classification

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

# Inspect dataset
print(data.head())
print(data.info())

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical columns
le = LabelEncoder()

data['Education'] = le.fit_transform(data['Education'])
data['Property_Area'] = le.fit_transform(data['Property_Area'])
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

# Select features
X = data[['ApplicantIncome','LoanAmount','Credit_History','Education','Property_Area']]

# Target
y = data['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Tree Structure Plot
plt.figure(figsize=(12,8))
plot_tree(model,
          feature_names=X.columns,
          class_names=['Rejected','Approved'],
          filled=True)
plt.show()

# Feature Importance
importance = model.feature_importances_

plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()
