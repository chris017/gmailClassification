import sqlite3
import pandas as pd

df = pd.read_sql("SELECT * FROM emails", sqlite3.connect("emails.db"))
df.to_parquet("emails.parquet", index=False)
print(f"Exported {len(df)} rows → emails.parquet")
print(df["label"].value_counts())
