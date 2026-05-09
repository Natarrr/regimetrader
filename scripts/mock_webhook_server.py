"""scripts/mock_webhook_server.py
Minimal HTTP server that captures Discord webhook POST payloads.

Used by test_daily_toplists_absence.yml and local testing to verify
the send_toplists_discord.py alert path without hitting real Discord.

Usage:
  python scripts/mock_webhook_server.py --port 9999 --out /tmp/captured.json

  # In another terminal:
  DISCORD_WEBHOOK_URL=http://localhost:9999 \
    python scripts/send_toplists_discord.py --input nonexistent.json --log-dir /tmp

The server writes each captured request body (JSON) to --out and returns HTTP 204.
Exits after receiving --count requests (default: 1) or SIGTERM/Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List

log = logging.getLogger("mock_webhook")


class _Handler(BaseHTTPRequestHandler):
    captures: List[dict] = []
    max_count: int = 1
    out_path: Path = Path("/tmp/captured.json")

    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", errors="replace")}

        self.__class__.captures.append(payload)
        log.info("Captured request #%d", len(self.__class__.captures))

        # Flush captures to file after every request
        self.__class__.out_path.write_text(
            json.dumps(self.__class__.captures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self.send_response(204)
        self.end_headers()

        # Shutdown after expected number of requests
        if len(self.__class__.captures) >= self.__class__.max_count:
            log.info("Received %d request(s) — shutting down", len(self.__class__.captures))
            # Schedule shutdown from a thread to avoid deadlock inside handler
            import threading
            threading.Thread(target=self.server.shutdown, daemon=True).start()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mock Discord webhook server")
    parser.add_argument("--port",  type=int,  default=9999)
    parser.add_argument("--out",   type=Path, default=Path("/tmp/captured.json"))
    parser.add_argument("--count", type=int,  default=1,
                        help="Exit after receiving this many requests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _Handler.captures  = []
    _Handler.max_count = args.count
    _Handler.out_path  = args.out

    server = HTTPServer(("localhost", args.port), _Handler)
    log.info("Mock webhook server listening on http://localhost:%d", args.port)
    log.info("Writing captures to %s", args.out)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    captures = _Handler.captures
    if captures:
        log.info("Captured %d request(s)", len(captures))
        return 0
    log.warning("No requests received")
    return 1


if __name__ == "__main__":
    sys.exit(main())
