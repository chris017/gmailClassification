import os
import re
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

st.set_page_config(page_title="Gmail Classifier", layout="wide")

DB_FILE  = "emails.db"
BERT_DIR = "distilbert_model" if os.path.exists("distilbert_model") else "chris017/emailClassification"
LR_FILE  = "model.pkl"

CATEGORY_COLORS = {
    "spam":       "#e74c3c",
    "personal":   "#2ecc71",
    "promotions": "#f39c12",
    "social":     "#3498db",
    "updates":    "#9b59b6",
    "forums":     "#1abc9c",
    "purchases":  "#e67e22",
}

STOPWORDS = {
    "the","a","an","is","in","it","of","to","and","for","you","your","on","at",
    "with","this","that","are","be","was","re","we","have","our","new","get",
    "can","all","or","de","ich","sie","die","der","das","und","ein","eine",
    "mit","von","fur","im","wird","hat","ihr",
}

SPAM_TLDS = {".ru", ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".click", ".loan"}
SPAM_KEYWORDS = {
    "winner", "won", "free", "claim", "prize", "congratulations",
    "click here", "limited time", "act now", "urgent", "verify your account",
    "you have been selected", "nigerian", "inheritance", "wire transfer",
    "guaranteed", "no risk", "make money", "earn cash",
}


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_bert():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR)
    model.to(device)
    model.eval()

    if os.path.exists(BERT_DIR):
        with open(f"{BERT_DIR}/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    else:
        from huggingface_hub import hf_hub_download
        pkl_path = hf_hub_download(repo_id=BERT_DIR, filename="label_encoder.pkl")
        with open(pkl_path, "rb") as f:
            le = pickle.load(f)

    return tokenizer, model, le, device

@st.cache_resource
def load_lr():
    with open(LR_FILE, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_stats():
    import json
    with open("stats.json") as f:
        return json.load(f)


# ── Rule-based spam detection ─────────────────────────────────────────────────
def rule_based_spam(subject, body, sender=""):
    text  = (subject + " " + body).lower()
    score = 0.0
    rules = []

    for tld in SPAM_TLDS:
        if tld in sender.lower() or tld in text:
            score += 0.4
            rules.append(f"suspicious TLD ({tld})")
            break

    hits = [kw for kw in SPAM_KEYWORDS if kw in text]
    if hits:
        score += min(0.15 * len(hits), 0.45)
        rules.append(f"spam keywords: {', '.join(hits[:3])}")

    letters = [c for c in subject if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.4:
        score += 0.2
        rules.append("excessive caps in subject")

    if (subject + body).count("!") > 3:
        score += 0.15
        rules.append("excessive exclamation marks")

    return {"score": min(round(score, 2), 1.0), "rules": rules, "is_spam": score >= 0.5}


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(subject, body="", sender=""):
    tokenizer, model, le, device = load_bert()

    rules = rule_based_spam(subject, body, sender)
    if rules["is_spam"]:
        probs = {c: 0.01 for c in le.classes_}
        probs["spam"] = rules["score"]
        return "spam", probs, rules["rules"]

    text = subject + " " + body
    enc  = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    enc  = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs_arr = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    probs = dict(zip(le.classes_, probs_arr.round(3).tolist()))
    label = max(probs, key=probs.get)
    return label, probs, []


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Gmail Email Classifier")
st.caption("7-class email classifier using fine-tuned DistilBERT with a rule-based spam override layer.")

tab1, tab2 = st.tabs(["Inbox Overview", "Predict"])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    s = load_stats()

    label_counts = s["label_counts"]
    total    = sum(label_counts.values())
    n_spam   = label_counts.get("spam", 0)
    spam_pct = n_spam / total * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Emails", f"{total:,}")
    col2.metric("Spam", f"{n_spam:,}")
    col3.metric("Spam Rate", f"{spam_pct:.1f}%")
    col4.metric("Categories", len(label_counts))

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Label Distribution")
        counts = pd.DataFrame(label_counts.items(), columns=["label", "count"])
        fig = px.pie(
            counts, values="count", names="label",
            color="label", color_discrete_map=CATEGORY_COLORS, hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top Spam Sender Domains")
        top_senders = pd.DataFrame(s["top_spam_domains"].items(), columns=["domain", "count"])
        top_senders = top_senders.sort_values("count")
        fig2 = px.bar(
            top_senders, x="count", y="domain", orientation="h",
            color_discrete_sequence=["#e74c3c"],
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Email Volume Over Time")
    daily = pd.DataFrame(s["daily_counts"])
    if not daily.empty:
        fig3 = px.bar(
            daily, x="day", y="count", color="label",
            color_discrete_map=CATEGORY_COLORS, barmode="stack",
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    st.subheader("Arrival Heatmap — Hour x Weekday")
    heat = pd.DataFrame(s["hourly_heatmap"])
    if not heat.empty:
        weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat["weekday"] = pd.Categorical(heat["weekday"], categories=weekday_order, ordered=True)
        heat = heat.sort_values("weekday")
        fig_heat = px.density_heatmap(
            heat, x="hour", y="weekday", z="count",
            color_continuous_scale="Blues", nbinsx=24,
        )
        fig_heat.update_layout(xaxis_title="Hour of Day", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    st.subheader("Email Body Length by Category (Median)")
    perc = s["body_length_percentiles"]
    perc_df = pd.DataFrame([
        {"label": k, "q25": v["q25"], "median": v["q50"], "q75": v["q75"]}
        for k, v in perc.items()
    ]).sort_values("median", ascending=False)
    fig_box = px.bar(
        perc_df, x="median", y="label", orientation="h",
        color="label", color_discrete_map=CATEGORY_COLORS,
        error_x=perc_df["q75"] - perc_df["median"],
    )
    fig_box.update_layout(showlegend=False, xaxis_title="Characters (median)", yaxis_title="")
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    st.subheader("Top Subject Keywords by Category")
    sel_cat = st.selectbox("Category", sorted(s["top_keywords"].keys()), key="kw_cat")
    kw_data = pd.DataFrame(s["top_keywords"][sel_cat].items(), columns=["word", "count"])
    kw_data = kw_data.sort_values("count")
    fig_kw = px.bar(
        kw_data, x="count", y="word", orientation="h",
        color_discrete_sequence=[CATEGORY_COLORS.get(sel_cat, "#95a5a6")],
    )
    fig_kw.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig_kw, use_container_width=True)



# ── Tab 2: Predict ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Classify an Email")

    col_a, col_b = st.columns(2)
    with col_a:
        subject = st.text_input("Subject", placeholder="Your invoice #1234 is ready")
        sender  = st.text_input("Sender (optional)", placeholder="noreply@example.com")
    with col_b:
        body = st.text_area("Body", placeholder="Hi, please find attached...", height=120)

    if st.button("Classify", type="primary", use_container_width=True):
        if not subject and not body:
            st.warning("Enter at least a subject or body.")
        else:
            with st.spinner("Classifying..."):
                label, probs, triggered_rules = predict(subject, body, sender)

            color = CATEGORY_COLORS.get(label, "#95a5a6")
            st.markdown(f"<h2 style='color:{color}'>{label.upper()}</h2>", unsafe_allow_html=True)

            if triggered_rules:
                st.error(f"Rule-based override: {' | '.join(triggered_rules)}")

            prob_df = pd.DataFrame(
                sorted(probs.items(), key=lambda x: -x[1]),
                columns=["category", "confidence"]
            )
            fig = px.bar(
                prob_df, x="confidence", y="category", orientation="h",
                color="category", color_discrete_map=CATEGORY_COLORS, range_x=[0, 1],
            )
            fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
