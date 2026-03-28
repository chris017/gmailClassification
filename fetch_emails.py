"""
fetch_emails.py
Fetches emails from Gmail API and saves them to a local SQLite database.
"""

import os
import base64
import sqlite3
import re
from datetime import datetime

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ── Config ────────────────────────────────────────────────────────────────────
CLIENT_SECRET_FILE = "client_secret_329521649187-h38bku79cs72ss58de9h6emvmco30231.apps.googleusercontent.com.json"
TOKEN_FILE = "token.json"
DB_FILE = "emails.db"
MAX_PER_CATEGORY = 5000               # max emails to pull per category

# Read-only access is sufficient
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


# ── Auth ──────────────────────────────────────────────────────────────────────
def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_body(payload):
    """Extract plain-text body from a Gmail message payload."""
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        # fallback: recurse into nested parts
        for part in payload["parts"]:
            result = decode_body(part)
            if result:
                return result
    else:
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    return ""


def get_header(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def extract_sender_domain(sender):
    match = re.search(r"@([\w.\-]+)", sender)
    return match.group(1).lower() if match else ""


def count_links(text):
    return len(re.findall(r"https?://", text))


def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


# ── DB ────────────────────────────────────────────────────────────────────────
def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            message_id   TEXT PRIMARY KEY,
            date         TEXT,
            sender       TEXT,
            sender_domain TEXT,
            subject      TEXT,
            body_preview TEXT,
            body_length  INTEGER,
            link_count   INTEGER,
            caps_ratio   REAL,
            label        TEXT
        )
    """)
    conn.commit()


# ── Main fetch ────────────────────────────────────────────────────────────────
def fetch_emails(service, query="", max_results=MAX_PER_CATEGORY):
    messages = []
    response = service.users().messages().list(
        userId="me", q=query, maxResults=min(max_results, 500)
    ).execute()

    messages.extend(response.get("messages", []))
    while "nextPageToken" in response and len(messages) < max_results:
        response = service.users().messages().list(
            userId="me", q=query,
            pageToken=response["nextPageToken"],
            maxResults=min(max_results - len(messages), 500)
        ).execute()
        messages.extend(response.get("messages", []))

    return messages[:max_results]


def process_message(service, msg_id, label):
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()

    headers = msg["payload"].get("headers", [])
    body = decode_body(msg["payload"])
    sender = get_header(headers, "from")

    return {
        "message_id":    msg_id,
        "date":          get_header(headers, "date"),
        "sender":        sender,
        "sender_domain": extract_sender_domain(sender),
        "subject":       get_header(headers, "subject"),
        "body_preview":  body[:300],
        "body_length":   len(body),
        "link_count":    count_links(body),
        "caps_ratio":    round(caps_ratio(body), 4),
        "label":         label,
    }


def save_to_db(conn, row):
    conn.execute("""
        INSERT OR REPLACE INTO emails
        VALUES (:message_id, :date, :sender, :sender_domain, :subject,
                :body_preview, :body_length, :link_count, :caps_ratio, :label)
    """, row)
    conn.commit()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    service = get_gmail_service()
    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    existing = {row[0] for row in conn.execute("SELECT message_id FROM emails")}

    # (gmail_query, label, max_to_fetch)
    jobs = [
        ("in:spam",                  "spam",       MAX_PER_CATEGORY),
        ("category:primary",         "personal",   MAX_PER_CATEGORY),
        ("category:promotions",      "promotions", MAX_PER_CATEGORY),
        ("category:social",          "social",     MAX_PER_CATEGORY),
        ("category:updates",         "updates",    MAX_PER_CATEGORY),
        ("category:forums",          "forums",     MAX_PER_CATEGORY),
        ("category:purchases",       "purchases",  MAX_PER_CATEGORY),
    ]

    total_new = 0
    for query, label, limit in jobs:
        print(f"\nFetching {limit} '{label}' emails (query: '{query}') …")
        msgs = fetch_emails(service, query=query, max_results=limit)
        new = [m for m in msgs if m["id"] not in existing]
        print(f"  {len(msgs)} found, {len(new)} new to process")

        for i, m in enumerate(new, 1):
            try:
                row = process_message(service, m["id"], label)
                save_to_db(conn, row)
                if i % 25 == 0:
                    print(f"  … {i}/{len(new)} saved")
            except Exception as e:
                print(f"  [skip] {m['id']}: {e}")

        total_new += len(new)

    conn.close()
    print(f"\nDone. {total_new} new emails saved to {DB_FILE}")

    # Quick preview
    df = pd.read_sql("SELECT label, COUNT(*) as n FROM emails GROUP BY label",
                     sqlite3.connect(DB_FILE))
    print("\nLabel distribution:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
