"""
train_model.py
Baseline classifier: TF-IDF + Logistic Regression on Gmail email categories.
Saves model artefacts to model.pkl for use in the Streamlit app.
"""

import sqlite3
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import numpy as np
import matplotlib.pyplot as plt

DB_FILE   = "emails.db"
MODEL_FILE = "model.pkl"

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data …")
conn = pd.read_sql("SELECT * FROM emails", sqlite3.connect(DB_FILE))
df   = conn.dropna(subset=["subject", "label"])

# Combine subject + body_preview into one text field.
# Subject alone is often enough for classification, but adding the body preview
# gives the model more signal — especially for spam and purchases.
df["text"] = df["subject"].fillna("") + " " + df["body_preview"].fillna("")

print(f"  {len(df)} emails | {df['label'].nunique()} classes")
print(df["label"].value_counts().to_string())

# ── 2. Features ───────────────────────────────────────────────────────────────
print("\nBuilding features ...")

# TF-IDF on subject + body; bigrams capture phrases like "click here" or "free offer"
tfidf = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_text = tfidf.fit_transform(df["text"])

# --- Numeric features ---
numeric_cols = ["body_length", "link_count", "caps_ratio"]
X_numeric = df[numeric_cols].fillna(0).values

X = hstack([X_text, X_numeric])
y = df["label"].values

# ── 3. Train / test split ─────────────────────────────────────────────────────
# stratify=y ensures rare classes appear in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 4. Train model ────────────────────────────────────────────────────────────
print("\nTraining Logistic Regression ...")
clf = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    multi_class="multinomial",
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n── Evaluation ───────────────────────────────────────────────────────")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
labels = sorted(df["label"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Email Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved → confusion_matrix.png")

# ── 6. Save artefacts ────────────────────────────────────────────────────────
with open(MODEL_FILE, "wb") as f:
    pickle.dump({"tfidf": tfidf, "clf": clf, "numeric_cols": numeric_cols}, f)

print(f"Model saved → {MODEL_FILE}")
