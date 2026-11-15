"""Run every named query in sql/*.sql, print results, and save each to reports/sql_outputs/<name>.csv

Queries are delimited by a line '-- name: <query_name>'.
Usage: python src/run_sql.py            (all files)
       python src/run_sql.py 03         (only files starting with 03)
"""
import glob
import os
import re
import sqlite3
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "dhandrishti.db")
OUT = os.path.join(ROOT, "reports", "sql_outputs")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def load_queries(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    con = sqlite3.connect(DB)
    for f in sorted(glob.glob(os.path.join(ROOT, "sql", f"{prefix}*.sql"))):
        print(f"\n{'=' * 90}\n{os.path.basename(f)}\n{'=' * 90}")
        for name, q in load_queries(f):
            df = pd.read_sql_query(q, con)
            df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
            print(f"\n-- {name}\n{df.head(25).to_string(index=False)}")
