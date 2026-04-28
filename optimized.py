import kagglehub
import pandas as pd
import numpy as np
import os
import re

from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# -------------------------------
# CONFIG
# -------------------------------
CONFIG = {
    "nrows": 200000,
    "max_features": 15000,
    "high_threshold": 5,
    "medium_threshold": 2
}

# Improved keyword weights
KEYWORD_WEIGHTS = {
    'error': 3, 'fail': 3, 'failed': 3, 'crash': 3,
    'not working': 4, 'blocked': 4, 'refund': 3,

    # medium signals weaker
    'slow': 1, 'delay': 1, 'late': 1,
    'problem': 1, 'waiting': 1
}

# -------------------------------
# LOAD DATA
# -------------------------------
path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")

csv_file = None
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_file = os.path.join(root, file)

print("Using file:", csv_file)

df = pd.read_csv(csv_file, nrows=CONFIG["nrows"])
df = df[['text']].dropna()

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    return text.lower()

df['text'] = df['text'].apply(clean_text)

# -------------------------------
# IMPROVED LABELING
# -------------------------------
def assign_priority(text):
    score = 0

    # keyword scoring
    for word, weight in KEYWORD_WEIGHTS.items():
        if word in text:
            score += weight

    # sentiment scoring
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment < -0.5:
        score += 2
    elif sentiment < -0.1:
        score += 1

    # urgency
    if "!" in text:
        score += 1

    # 🔥 Special fix: prevent delay → High
    if "delay" in text and not any(x in text for x in ['error', 'fail', 'crash']):
        score = min(score, 2)

    # thresholds
    if score >= CONFIG["high_threshold"]:
        return "High"
    elif score >= CONFIG["medium_threshold"]:
        return "Medium"
    else:
        return "Low"

df['priority'] = df['text'].apply(assign_priority)

print("\nLabel Distribution:\n", df['priority'].value_counts())

# -------------------------------
# UPSAMPLING
# -------------------------------
df_high = df[df.priority == "High"]
df_medium = df[df.priority == "Medium"]
df_low = df[df.priority == "Low"]

target = len(df_low)

df_high = resample(df_high, replace=True, n_samples=target, random_state=42)
df_medium = resample(df_medium, replace=True, n_samples=target, random_state=42)

df_balanced = pd.concat([df_low, df_high, df_medium])

print("\nBalanced Distribution:\n", df_balanced['priority'].value_counts())

# -------------------------------
# VISUALIZATION
# -------------------------------
df['priority'].value_counts().plot(kind='bar', title="Before Balancing")
plt.show()

df_balanced['priority'].value_counts().plot(kind='bar', title="After Balancing")
plt.show()

# -------------------------------
# FEATURES
# -------------------------------
df_balanced['length'] = df_balanced['text'].apply(len)
df_balanced['caps'] = df_balanced['text'].apply(lambda x: sum(1 for c in x if c.isupper()))

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced[['text','length','caps']],
    df_balanced['priority'],
    test_size=0.2,
    random_state=42
)

# -------------------------------
# TF-IDF (with n-grams)
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=CONFIG["max_features"],
    ngram_range=(1,2)
)

X_train_text = vectorizer.fit_transform(X_train['text'])
X_test_text = vectorizer.transform(X_test['text'])

X_train_final = hstack([X_train_text, np.array(X_train[['length','caps']])])
X_test_final = hstack([X_test_text, np.array(X_test[['length','caps']])])

# -------------------------------
# SCALE FEATURES (Fix convergence)
# -------------------------------
scaler = StandardScaler(with_mean=False)
X_train_final = scaler.fit_transform(X_train_final)
X_test_final = scaler.transform(X_test_final)

# -------------------------------
# MODEL (Fixed convergence)
# -------------------------------
model = LogisticRegression(
    max_iter=500,
    solver='saga',
    class_weight='balanced'
)

model.fit(X_train_final, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test_final)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, y_pred, labels=["High","Medium","Low"])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["High","Medium","Low"],
            yticklabels=["High","Medium","Low"])

plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_priority(text):
    cleaned = clean_text(text)

    length = len(cleaned)
    caps = sum(1 for c in text if c.isupper())

    vec = vectorizer.transform([cleaned])
    final_input = hstack([vec, np.array([[length, caps]])])
    final_input = scaler.transform(final_input)

    return model.predict(final_input)[0]

# -------------------------------
# TEST CASES
# -------------------------------
tests = [
    "APP NOT WORKING!!!",
    "My delivery is delayed",
    "How do I reset my password?",
    "System crashed and payment failed",
    "Thanks for the help"
]

print("\n--- Sample Predictions ---")
for t in tests:
    print(t, "->", predict_priority(t))