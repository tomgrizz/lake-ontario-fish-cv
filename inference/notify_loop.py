"""
Run notify.py every 30 minutes in a loop.
Leave this running in a terminal — no admin or Task Scheduler needed.

Usage: python notify_loop.py [--interval 30]
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parent / "notify.py"
LOG    = r"G:\Projects\model_outputs\run_20260428\processing_log.sqlite"
TOPIC  = "ontario-fish-update"
TOTAL  = 258955

interval_minutes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--interval" else 30
interval_seconds = interval_minutes * 60

print(f"Sending notifications to ntfy.sh/{TOPIC} every {interval_minutes} minutes.")
print("Leave this window open. Ctrl+C to stop.\n")

while True:
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--log", LOG, "--topic", TOPIC, "--total", str(TOTAL)],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    time.sleep(interval_seconds)
