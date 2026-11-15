"""Load the raw CSVs into a local SQLite warehouse (data/dhandrishti.db) with indexes."""
import os
import sqlite3
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DB = os.path.join(ROOT, "data", "dhandrishti.db")

TABLES = ["users", "merchants", "banks", "campaigns", "transactions", "refunds", "support_tickets"]
INDEXES = [
    "CREATE INDEX ix_tx_user ON transactions(user_id)",
    "CREATE INDEX ix_tx_ts ON transactions(ts)",
    "CREATE INDEX ix_tx_payee ON transactions(payee_id)",
    "CREATE INDEX ix_ref_txn ON refunds(txn_id)",
    "CREATE INDEX ix_tk_user ON support_tickets(user_id)",
]

if __name__ == "__main__":
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    for t in TABLES:
        df = pd.read_csv(os.path.join(RAW, f"{t}.csv"))
        df.to_sql(t, con, index=False)
        print(f"{t:<16} {len(df):>9,} rows")
    for q in INDEXES:
        con.execute(q)
    con.commit()
    con.close()
    print(f"-> {DB}")
