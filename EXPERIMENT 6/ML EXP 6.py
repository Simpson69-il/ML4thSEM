import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

## Scenario 1: Bagging (Diabetes Prediction)
# Load dataset
df = pd.read_csv("diabetes_bagging.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Bagging
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
bag.fit(X_train, y_train)
y_pred_bag = bag.predict(X_test)

# Accuracy
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_bag = accuracy_score(y_test, y_pred_bag)

print("Decision Tree:", acc_dt)
print("Bagging:", acc_bag)

plt.bar(["Decision Tree", "Bagging"], [acc_dt, acc_bag])
plt.title("Accuracy Comparison")
plt.show()

sns.heatmap(confusion_matrix(y_test, y_pred_bag), annot=True)
plt.title("Confusion Matrix (Bagging)")
plt.show()

## Scenario 2: Boosting (Customer Churn)
df = pd.read_csv("churn_boosting.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ada = AdaBoostClassifier()
gb = GradientBoostingClassifier()

ada.fit(X_train, y_train)
gb.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)
y_pred_gb = gb.predict(X_test)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, gb.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
plt.title("ROC Curve - Gradient Boosting")
plt.show()

# Feature Importance
plt.barh(X.columns, gb.feature_importances_)
plt.title("Feature Importance")
plt.show()

## Scenario 3: Random Forest (Income Prediction)

df = pd.read_csv("income_random_forest.csv")

X = df.drop("income", axis=1)
y = df["income"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

scores = []
trees = [10, 50, 100]

for t in trees:
    rf = RandomForestClassifier(n_estimators=t)
    rf.fit(X_train, y_train)
    scores.append(accuracy_score(y_test, rf.predict(X_test)))
    
plt.plot(trees, scores, marker='o')
plt.title("Accuracy vs Trees")
plt.xlabel("Trees")
plt.ylabel("Accuracy")
plt.show()

plt.barh(X.columns[:10], rf.feature_importances_[:10])
plt.title("Feature Importance")
plt.show()


## Scenario 4: Stacking (Heart Disease)

df = pd.read_csv("heart_stacking.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

base_models = [
    ('lr', LogisticRegression()),
    ('svm', SVC(probability=True)),
    ('dt', DecisionTreeClassifier())
]

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

stack.fit(X_train, y_train)

y_pred_stack = stack.predict(X_test)    

models = ["Stacking"]
scores = [accuracy_score(y_test, y_pred_stack)]

plt.bar(models, scores)
plt.title("Model Comparison")
plt.show()


## Scenario 5: SMOTE (Fraud Detection)
df = pd.read_csv("fraud_smote.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Before SMOTE:", np.bincount(y))

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

print("After SMOTE:", np.bincount(y_res))

sns.countplot(x=y)
plt.title("Before SMOTE")
plt.show()

sns.countplot(x=y_res)
plt.title("After SMOTE")
plt.show()

