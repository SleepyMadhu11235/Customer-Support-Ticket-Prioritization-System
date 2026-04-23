import kagglehub
import pandas as pd
import numpy as np
import os
import re

from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from scipy.sparse import hstack

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")

csv_file = None
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_file = os.path.join(root, file)

print("Using file:", csv_file)

df = pd.read_csv(csv_file, nrows=200000)
print("Dataset Loaded:", df.shape)

# -------------------------------
# STEP 2: Keep Text
# -------------------------------
df = df[['text']].dropna()

# -------------------------------
# STEP 3: Clean Text
# -------------------------------
def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    return text.lower()

df['text'] = df['text'].apply(clean_text)

# -------------------------------
# STEP 4: HYBRID LABELING
# -------------------------------
def assign_priority(text):
    score = 0

    high_keywords = ['error','fail','failed','crash','not working','blocked','refund']
    medium_keywords = ['slow','delay','late','problem','waiting']

    # keyword scoring
    for word in high_keywords:
        if word in text:
            score += 2

    for word in medium_keywords:
        if word in text:
            score += 1

    # sentiment scoring
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment < -0.5:
        score += 2
    elif sentiment < 0:
        score += 1

    # urgency signals
    if "!" in text:
        score += 1
    if text.isupper():
        score += 1

    # better thresholds
    if score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

df['priority'] = df['text'].apply(assign_priority)

print("\nLabel Distribution:\n", df['priority'].value_counts())

# -------------------------------
# STEP 5: UPSAMPLE (NOT DOWNSAMPLE)
# -------------------------------
df_high = df[df.priority == "High"]
df_medium = df[df.priority == "Medium"]
df_low = df[df.priority == "Low"]

target_size = len(df_low)  # match largest class

df_high_up = resample(df_high, replace=True, n_samples=target_size, random_state=42)
df_medium_up = resample(df_medium, replace=True, n_samples=target_size, random_state=42)

df_balanced = pd.concat([df_low, df_high_up, df_medium_up])

print("\nBalanced Distribution:\n", df_balanced['priority'].value_counts())

# -------------------------------
# STEP 6: EXTRA FEATURES
# -------------------------------
df_balanced['length'] = df_balanced['text'].apply(len)
df_balanced['caps'] = df_balanced['text'].apply(lambda x: sum(1 for c in x if c.isupper()))

# -------------------------------
# STEP 7: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced[['text', 'length', 'caps']],
    df_balanced['priority'],
    test_size=0.2,
    random_state=42
)

# -------------------------------
# STEP 8: TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

X_train_text = vectorizer.fit_transform(X_train['text'])
X_test_text = vectorizer.transform(X_test['text'])

# Combine text + numeric features
X_train_final = hstack([X_train_text, np.array(X_train[['length','caps']])])
X_test_final = hstack([X_test_text, np.array(X_test[['length','caps']])])

# -------------------------------
# STEP 9: Train Model
# -------------------------------
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train_final, y_train)

# -------------------------------
# STEP 10: Evaluation
# -------------------------------
y_pred = model.predict(X_test_final)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# STEP 11: Prediction Function
# -------------------------------
def predict_priority(text):
    cleaned = clean_text(text)

    length = len(cleaned)
    caps = sum(1 for c in text if c.isupper())

    vec = vectorizer.transform([cleaned])
    final_input = hstack([vec, np.array([[length, caps]])])

    return model.predict(final_input)[0]

# -------------------------------
# STEP 12: Test Cases
# -------------------------------
test_cases = [
    "APP NOT WORKING!!!",
    "My delivery is delayed",
    "How do I reset my password?",
    "System crashed and payment failed",
    "Thanks for the help"
]

print("\n--- Sample Predictions ---")
for t in test_cases:
    print(f"Text: {t}")
    print(f"Prediction: {predict_priority(t)}\n")