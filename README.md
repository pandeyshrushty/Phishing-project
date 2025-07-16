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

kali linux

Thanks! You're seeing this message because **Kali Linux (with Python 3.13)** uses a **system protection (PEP 668)** to avoid breaking core Python packages. This prevents unsafe installations using `pip3` in the system environment.

But don't worry — the solution is simple and safe.

---

## ✅ Recommended Fix: Use `--break-system-packages` (Safe for This Case)

You're only installing `scikit-learn`, which is safe.

### Run this:

```bash
pip3 install scikit-learn --break-system-packages
```

This will force the install **without damaging your system**, because `scikit-learn` is a user-level package, not a system one.

---

## ⚠️ Alternative (Optional but Safer): Use `venv`

If you prefer **not to override system packages**, use a **Python virtual environment**:

### Step 1: Install venv if missing

```bash
sudo apt install python3-venv
```

### Step 2: Create virtual environment

```bash
python3 -m venv phishing-env
source phishing-env/bin/activate
```

You’ll now see something like this:

```bash
(phishing-env) networksrworld@kali:~/phishing_detector$
```

### Step 3: Install packages inside the venv

```bash
pip install scikit-learn pandas
```

### Step 4: Run your script

```bash
python phishing_detection_model.py
```

To exit the virtual environment later, just type:

```bash
deactivate
```

---

## ✅ TL;DR

### 💡 Quickest fix:

```bash
pip3 install scikit-learn --break-system-packages
```

Then run:

```bash
python3 phishing_detection_model.py
```

---

Let me know if you want help setting up the venv or want a safer long-term setup for Kali!


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
