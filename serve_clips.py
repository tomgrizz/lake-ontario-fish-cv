"""
Simple static file server for Label Studio clips.
Serves G:\Projects without authentication so the browser video player
can load clips remotely over Tailscale.

Usage: python serve_clips.py
Clips accessible at: http://<tailscale-ip>:8888/label_batches_ts/clips/clip_xxx.mp4
"""
import http.server
import os

PORT = 8888
ROOT = r"G:\Projects"

os.chdir(ROOT)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

print(f"Serving {ROOT} on 0.0.0.0:{PORT}")
print(f"Clips at http://100.68.58.2:{PORT}/label_batches_ts/clips/")
http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
