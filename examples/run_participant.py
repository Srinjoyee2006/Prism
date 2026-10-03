"""
Local Web Participant Server for LiveKit End-to-End Testing (Stage 7).

Serves the minimal local web participant at http://127.0.0.1:8080 and provides
an automatic token generation endpoint (/api/token) using local dev credentials.

Usage:
    python examples/run_participant.py

Then in browser:
    Click 'Connect & Start Mic' to join 'prism-test' and speak to Prism!
"""

import argparse
import json
import logging
import os
import sys
import urllib.parse
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.livekit.token import generate_participant_token

STATIC_DIR = Path(__file__).resolve().parent / "web_participant"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("participant_server")


class ParticipantRequestHandler(SimpleHTTPRequestHandler):
    """Serves the static web participant and local token generation API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)

        if parsed_url.path == "/api/token":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            room = query_params.get("room", ["prism-test"])[0]
            identity = query_params.get("identity", ["human-participant"])[0]

            token = generate_participant_token(
                room_name=room,
                identity=identity,
                name="Human Participant",
            )

            response_data = {
                "token": token,
                "room": room,
                "identity": identity,
                "ws_url": os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"),
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        super().do_GET()

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy HTTP asset access logs
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prism Local LiveKit Web Participant")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve web participant (default: 8080)")
    parser.add_argument("--room", type=str, default="prism-test", help="Default LiveKit room (default: prism-test)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    token = generate_participant_token(room_name=args.room)
    server_address = ("127.0.0.1", args.port)

    print("=" * 70)
    print("  PRISM LOCAL WEBRTC PARTICIPANT (BROWSER CLIENT)")
    print("=" * 70)
    print(f"Room:        {args.room}")
    print("Server URL:  ws://127.0.0.1:7880")
    print(f"Web Client:  http://127.0.0.1:{args.port}")
    print(f"JWT Token:   {token[:35]}...{token[-15:]}")
    print("=" * 70)
    print(f"Ready! Open http://127.0.0.1:{args.port} in your browser and click 'Connect'.")
    print("Press Ctrl+C to stop the participant server.\n")

    httpd = ThreadingHTTPServer(server_address, ParticipantRequestHandler)

    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web participant server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
