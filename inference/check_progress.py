import sqlite3
from pathlib import Path
import sys

db = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("G:/Projects/model_outputs/run_pilot/processing_log.sqlite")

if not db.exists():
    print("Log not found yet:", db)
    sys.exit(0)

conn = sqlite3.connect(str(db))
rows = conn.execute("SELECT status, COUNT(*) FROM processing_log GROUP BY status").fetchall()
total = sum(n for _, n in rows)
for status, n in sorted(rows):
    print(f"  {status:10s}: {n}")
print(f"  {'total':10s}: {total}")
