print("Student Name: Mohd Shafique RB")
print("Roll Number: 24BAD074")

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)


# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_message'])


# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])


# Train-test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)


# Train Multinomial NB
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)


# Predict classes
y_pred = model.predict(X_test)


# Evaluate performance
print("\nMULTINOMIAL NAÏVE BAYES RESULTS")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – SMS Spam")
plt.show()


# Misclassified examples
mis_mask = (y_test != y_pred)
misclassified = df.loc[idx_test[mis_mask]]
print("\nSample Misclassified Messages:")
print(misclassified[['label', 'message']].head())


# Top spam words visualization
feature_names = vectorizer.get_feature_names_out()
spam_probs = model.feature_log_prob_[1]
top_indices = np.argsort(spam_probs)[-20:]

plt.figure()
plt.barh(feature_names[top_indices], spam_probs[top_indices])
plt.title("Top Words Indicative of Spam")
plt.show()
