"""
export_stats.py
Computes aggregated statistics from emails.db and writes them to stats.json.
Run this locally whenever the DB is updated. Commit stats.json to the repo.
No raw email content (subjects, bodies, senders) is included in the output.
"""

import json
import sqlite3
import pandas as pd

DB_FILE  = "emails.db"
OUT_FILE = "stats.json"

conn = sqlite3.connect(DB_FILE)
df   = pd.read_sql("SELECT * FROM emails", conn)
conn.close()

df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df = df.dropna(subset=["label"])

stats = {}

# Label counts
stats["label_counts"] = df["label"].value_counts().to_dict()

# Top spam sender domains (domains only, no personal addresses)
stats["top_spam_domains"] = (
    df[df["label"] == "spam"]["sender_domain"]
    .value_counts().head(10).to_dict()
)

# Daily email counts per label
df_time = df.dropna(subset=["date"]).copy()
df_time["day"] = df_time["date"].dt.strftime("%Y-%m-%d")
stats["daily_counts"] = (
    df_time.groupby(["day", "label"])
    .size().reset_index(name="count")
    .to_dict(orient="records")
)

# Average features per label
stats["feature_averages"] = (
    df.groupby("label")[["link_count", "caps_ratio", "body_length"]]
    .mean().round(3).to_dict()
)

# Hourly heatmap (weekday x hour counts)
df_time["hour"]    = df_time["date"].dt.hour
df_time["weekday"] = df_time["date"].dt.day_name()
stats["hourly_heatmap"] = (
    df_time.groupby(["weekday", "hour"])
    .size().reset_index(name="count")
    .to_dict(orient="records")
)

# Body length distribution per label (percentiles, no raw values)
stats["body_length_percentiles"] = {}
for label, group in df.groupby("label"):
    q = group["body_length"].quantile([0.25, 0.5, 0.75, 0.95]).round(0).astype(int)
    stats["body_length_percentiles"][label] = {
        "q25": int(q[0.25]), "q50": int(q[0.50]),
        "q75": int(q[0.75]), "q95": int(q[0.95]),
    }

# Monthly spam rate
df_time["month"] = df_time["date"].dt.to_period("M").astype(str)
monthly       = df_time.groupby(["month", "label"]).size().reset_index(name="count")
monthly_total = monthly.groupby("month")["count"].sum().reset_index(name="total")
monthly_spam  = monthly[monthly["label"] == "spam"].merge(monthly_total, on="month")
monthly_spam["spam_rate"] = (monthly_spam["count"] / monthly_spam["total"] * 100).round(1)
stats["monthly_spam_rate"] = monthly_spam[["month", "spam_rate"]].to_dict(orient="records")

# Top subject keywords per label (word frequencies only, no full subjects)
import re
from collections import Counter

STOPWORDS = {
    "the","a","an","is","in","it","of","to","and","for","you","your","on","at",
    "with","this","that","are","be","was","re","we","have","our","new","get",
    "can","all","or","de","ich","sie","die","der","das","und","ein","eine",
    "mit","von","fur","im","wird","hat","ihr",
}

stats["top_keywords"] = {}
for label, group in df.groupby("label"):
    words = [
        w for subj in group["subject"].dropna().str.lower()
        for w in re.findall(r"[a-zäöüß]{3,}", subj)
        if w not in STOPWORDS
    ]
    stats["top_keywords"][label] = dict(Counter(words).most_common(20))

with open(OUT_FILE, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Stats exported to {OUT_FILE}")
print(f"  Labels:  {list(stats['label_counts'].keys())}")
print(f"  Total:   {sum(stats['label_counts'].values()):,} emails aggregated")
