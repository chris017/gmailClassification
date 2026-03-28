# Gmail Email Classifier

A 7-class email classifier built on my personal Gmail inbox using fine-tuned DistilBERT, with a rule-based spam override layer and a Streamlit dashboard for live predictions and inbox analytics.

**Live demo:** [Streamlit App](https://share.streamlit.io)
**Model:** [chris017/emailClassification](https://huggingface.co/chris017/emailClassification)

---

## What it does

- Fetches emails from Gmail via the Gmail API and classifies them into 7 categories: `spam`, `personal`, `promotions`, `social`, `updates`, `forums`, `purchases`
- Fine-tunes `distilbert-base-multilingual-cased` on the collected emails (German + English)
- Adds a rule-based spam layer on top to catch obvious spam the model might miss (suspicious TLDs, spam keywords, excessive caps)
- Serves predictions and inbox analytics through a Streamlit dashboard

---

## Categories & Data

Emails are labeled using Gmail's built-in category tabs. Final dataset after enrichment:

| Category   | Emails |
|------------|--------|
| social     | 5,000  |
| promotions | 4,998  |
| updates    | 4,708  |
| personal   | 1,589  |
| purchases  |   445  |
| spam       | ~2,020 (306 Gmail + ~1,714 Enron) |
| forums     |   163  |

---

## Architecture

```
Gmail API
    │
    ▼
fetch_emails.py          ← pulls emails per category, stores in SQLite
    │
    ▼
emails.db (local only)
    │
    ├──► export_stats.py  ← aggregates anonymous stats → stats.json (committed)
    │
    └──► enrich_spam.ipynb + finetune_distilbert.ipynb
                │
                ▼
         distilbert_model/  ← hosted on Hugging Face
                │
                ▼
            app.py          ← Streamlit dashboard
                ├── Overview tab  (reads stats.json)
                └── Predict tab   (loads model from HF Hub)
```

---

## Model Comparison

Two models were trained and compared. The baseline uses TF-IDF features + Logistic Regression. The final model fine-tunes DistilBERT on the same data.

### Logistic Regression (baseline)

| Category   | Precision | Recall | F1   |
|------------|-----------|--------|------|
| forums     | 0.130     | 0.781  | 0.222 |
| personal   | 0.285     | 0.116  | 0.165 |
| promotions | 0.662     | 0.333  | 0.443 |
| purchases  | 0.058     | 0.551  | 0.105 |
| social     | 0.638     | 0.902  | 0.747 |
| spam       | 0.083     | 0.426  | 0.138 |
| updates    | 0.261     | 0.013  | 0.024 |
| **weighted avg** | **0.480** | **0.402** | **0.375** |

### DistilBERT fine-tuned (final model)

| Category   | Precision | Recall | F1    |
|------------|-----------|--------|-------|
| forums     | 0.955     | 0.636  | 0.764 |
| personal   | 0.716     | 0.594  | 0.649 |
| promotions | 0.917     | 0.944  | 0.930 |
| purchases  | 0.746     | 0.494  | 0.595 |
| social     | 0.999     | 1.000  | 1.000 |
| spam       | 0.987     | 0.933  | 0.959 |
| updates    | 0.800     | 0.874  | 0.835 |
| **weighted avg** | **0.896** | **0.897** | **0.895** |

### Why DistilBERT is better

TF-IDF treats every word independently and cannot capture context. `updates` and `promotions` share similar vocabulary, causing the Logistic Regression model to almost never predict `updates` (F1: 0.024). DistilBERT's self-attention mechanism understands word context and separates these categories cleanly (F1: 0.835).

`class_weight="balanced"` helped the Logistic Regression recall for small classes like `forums` and `spam`, but at the cost of very low precision — the model predicted these classes too aggressively.

---

## Why these specific choices

**`distilbert-base-multilingual-cased`**
My inbox is a mix of German and English. The multilingual variant handles both languages simultaneously. `cased` preserves capitalisation, which is a useful spam signal (ALL CAPS subjects).

**Fine-tuning instead of training from scratch**
DistilBERT was pre-trained on 104 languages (~billions of tokens). Fine-tuning adds a classification head and nudges the existing weights — I only needed ~14k emails instead of millions.

**`learning_rate=2e-5`**
Standard range for BERT-family fine-tuning (2e-5 to 5e-5). Too high destroys the pre-trained weights (catastrophic forgetting); too low and the model barely adapts.

**3 epochs**
With a pre-trained model and ~14k samples, 3 epochs is enough to converge without overfitting. The training loss went from 0.71 → 0.32 → 0.22.

**Spam enrichment with Enron dataset**
My Gmail spam folder had only 306 examples — too few for the model to generalise. I added ~1,714 spam emails from the [Enron spam dataset](https://huggingface.co/datasets/SetFit/enron_spam) (spam class only) to bring spam to ~2,020 samples. This improved spam F1 from 0.602 to 0.959.

**Rule-based spam layer on top of the model**
Even after enrichment, the model sometimes missed obvious spam with Russian domains (e.g. `totally-legit.ru`). BERT looks at semantics, not URL patterns. The rule-based layer checks for suspicious TLDs (`.ru`, `.xyz`, `.tk`, etc.), spam keywords, excessive caps, and exclamation marks — and overrides the model prediction to `spam` if triggered.

**TF-IDF bigrams**
`ngram_range=(1,2)` captures phrases like "click here" or "free offer" which are stronger spam signals than individual words alone.

**`stratify=y` in train/test split**
Without stratification, all `forums` emails (163 total) could end up only in training, making evaluation impossible for that class.

---

## Challenges encountered

### Google Cloud Console & Gmail API setup
Setting up OAuth2 for a desktop application requires creating a project in Google Cloud Console, enabling the Gmail API, creating OAuth credentials, and adding your account as a test user (while the app is in testing mode). The client secret JSON file must never be committed to the repository.

### Class imbalance
The dataset is heavily skewed: `social` and `promotions` have 5,000 samples each while `forums` has 163. This required `class_weight="balanced"` for Logistic Regression and careful evaluation using weighted and macro F1 rather than accuracy.

### Databricks — abandoned
Two separate issues blocked Databricks:
1. **SQLite not supported**: Databricks does not support local SQLite files. Had to convert `emails.db` to Parquet via `export_to_parquet.py` and upload to DBFS.
3. **Public DBFS root disabled**: When trying to create a Delta table via the Databricks UI, the workspace threw `UnsupportedOperationException: Public DBFS root is disabled`. The correct fix is to use Unity Catalog Volumes, but this added unnecessary complexity for a prototype.

**Decision**: switched to Google Colab with a T4 GPU, which was simpler, free, and fast enough (~20 minutes for 3 epochs).

### Google Colab
Colab required uploading `emails.parquet` via `google.colab.files.upload()` since the local SQLite DB was not accessible. The trained model was downloaded as a zip and extracted locally.

### Privacy — no raw emails in the repo
`emails.db` and `emails.parquet` are in `.gitignore`. Instead, `export_stats.py` computes aggregated statistics (label counts, domain frequencies, keyword frequencies, hourly heatmaps) with no raw email content and writes them to `stats.json`, which is committed to the repo. The Streamlit Overview tab reads only from `stats.json`.

### Model hosting
The fine-tuned model (~500MB) is too large for GitHub. It is hosted on [Hugging Face Hub](https://huggingface.co/chris017/emailClassification). The app detects whether a local `distilbert_model/` folder exists (development) and falls back to the HF Hub (production).

---

## Project structure

```
├── app.py                     # Streamlit dashboard
├── fetch_emails.py            # Gmail API data collection
├── train_model.py             # Baseline: TF-IDF + Logistic Regression
├── finetune_distilbert.ipynb  # Fine-tuning notebook (run on Google Colab)
├── enrich_spam.ipynb          # Adds Enron spam data to the dataset
├── explore_data.ipynb         # Initial data exploration
├── compare_models.ipynb       # Side-by-side model comparison with rule-based layer
├── export_stats.py            # Generates stats.json from emails.db (run locally)
├── export_to_parquet.py       # Converts emails.db to emails.parquet for Colab
├── stats.json                 # Aggregated inbox statistics (no raw email content)
├── requirements.txt
└── .gitignore
```

---

## Running this yourself

### 1. Gmail API credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable the **Gmail API**
4. Create OAuth credentials → **Desktop application**
5. Download the client secret JSON and place it in the project root
6. Add your Gmail address as a test user under **OAuth consent screen**

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Additional packages needed for local data collection and training
pip install google-api-python-client google-auth-oauthlib scikit-learn scipy datasets
```

### 3. Collect emails

```bash
python fetch_emails.py
```

A browser window opens for Google login on the first run. The token is saved to `token.json` for subsequent runs.

### 4. Enrich spam data (optional but recommended)

Run all cells in `enrich_spam.ipynb`. This adds ~2,000 spam emails from the Enron dataset to improve spam detection.

### 5. Train baseline model

```bash
python train_model.py
```

Saves `model.pkl`.

### 6. Fine-tune DistilBERT

Export to Parquet first:

```bash
python export_to_parquet.py
```

Upload `emails.parquet` to Google Colab, open `finetune_distilbert.ipynb`, set runtime to **T4 GPU**, and run all cells. Download the resulting `distilbert_model.zip` and extract it locally.

### 7. Generate stats

```bash
python export_stats.py
```

Saves `stats.json` for the Streamlit Overview tab.

### 8. Run the app

```bash
streamlit run app.py
```

---

## Deployment

The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io):

- **Code**: GitHub repository
- **Model**: [Hugging Face Hub](https://huggingface.co/chris017/emailClassification) — loaded automatically at startup
- **Data**: `stats.json` in the repository (aggregated, no personal content)

To deploy your own instance:
1. Fork the repository
2. Upload your own fine-tuned model to Hugging Face Hub
3. Update `BERT_DIR` in `app.py` to your HF repo ID
4. Run `export_stats.py` locally and commit `stats.json`
5. Connect the repository at [share.streamlit.io](https://share.streamlit.io)
6. In the Streamlit Cloud dashboard → **Settings** → **Advanced settings** → set **Python version to 3.11** (required for PyTorch compatibility — `runtime.txt` is not supported by Streamlit Cloud)
7. Deploy
