phishing project
one click before think

# phishing_detection_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/phishing_dataset.csv")  # CSV with columns: ['text', 'label']
print("Dataset Loaded:", df.shape)

# Preprocess
X = df['text']
y = df['label']  # 0 = Legitimate, 1 = Phishing

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

output

Dataset Loaded: (1000, 2)
Accuracy: 0.94
Prediction: Phishing

+--------------------------+
| Email Input              |
+--------------------------+
          |
          v
+--------------------------+
| Text Vectorization (TF-IDF) |
+--------------------------+
          |
          v
+--------------------------+
| Logistic Regression Model |
+--------------------------+
          |
          v
+--------------------------+
| Output: Phishing or Not   |
+--------------------------+



# Test sample
sample = ["Your account has been suspended. Click here to verify."]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
print("Prediction:", "Phishing" if prediction[0] == 1 else "Legitimate")
