"""Minimal stand-in for api.airouter.io: picks a fixed model and records usage reports."""

import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

PICK = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.2"
CATALOG = ["gpt-5-mini", "gpt-5.2", "claude-4-5-sonnet"]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": m, "object": "model"} for m in CATALOG]})
        else:
            self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/select":
            print("SELECT", json.dumps(body), flush=True)
            pick = PICK if PICK in body["models"] else body["models"][0]
            self._send(200, {"decision_id": str(uuid.uuid4()), "predicted_savings": None, "candidates": [
                {"model": pick, "effort": None, "quality": 1, "costs": 1, "latency": 1,
                 "quality_score": 1, "costs_score": 1, "latency_score": 1, "score": 1}]})
        elif self.path == "/v1/usage-reports":
            print("REPORTS", json.dumps(body), flush=True)
            self._send(202, {"accepted": len(body["reports"]), "duplicate": 0, "rejected": []})
        else:
            self._send(404, {})

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
