"""
Send a run status notification to ntfy.sh (or any webhook).

Usage:
  python notify.py --log G:/Projects/model_outputs/run_20260428/processing_log.sqlite
                   --topic lake-ontario-fish-cv-abc123
                   --total 258955

Schedule via Windows Task Scheduler to run every 30 minutes while the
inference run is in progress.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, metavar="PATH",
                    help="Path to processing_log.sqlite")
    ap.add_argument("--topic", required=True, metavar="TOPIC",
                    help="ntfy.sh topic name (e.g. lake-ontario-fish-cv-abc123)")
    ap.add_argument("--total", type=int, default=258955, metavar="N",
                    help="Total videos in corpus (default 258955)")
    ap.add_argument("--ntfy-url", default="https://ntfy.sh", metavar="URL",
                    help="ntfy server base URL (default: https://ntfy.sh)")
    args = ap.parse_args()

    db = Path(args.log)
    if not db.exists():
        print(f"Log not found: {db}")
        sys.exit(1)

    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM processing_log GROUP BY status"
    ).fetchall()
    conn.close()

    counts = {r[0]: r[1] for r in rows}
    n_success = counts.get("success", 0)
    n_error   = counts.get("error", 0)
    n_skipped = counts.get("skipped", 0)
    n_done    = n_success + n_error + n_skipped
    pct       = 100.0 * n_done / args.total if args.total > 0 else 0.0

    # Rough ETA based on success rate (errors/skips are fast, successes dominate)
    # Can't compute fps here without timing data, so just show completion %
    status_line = "OK" if n_error == 0 else f"ERRORS: {n_error:,}"
    title = f"Fish CV - {pct:.1f}% done  [{status_line}]"
    body = (
        f"Success : {n_success:,}\n"
        f"Error   : {n_error:,}\n"
        f"Skipped : {n_skipped:,}\n"
        f"Total   : {n_done:,} / {args.total:,} ({pct:.1f}%)\n"
        f"Updated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    url = f"{args.ntfy_url.rstrip('/')}/{args.topic}"
    req = urllib.request.Request(
        url,
        data=body.encode(),
        headers={"Title": title, "Priority": "default"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"Sent to {url}  [{resp.status}]")
            print(f"  {title}")
            print(f"  {body.splitlines()[2]}")  # total line
    except Exception as e:
        print(f"Notification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
