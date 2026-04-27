"""
main.py — Entry point for the multi-agent coding assistant.

Run with:
    python main.py

Expects:
    ../secret      — Puter auth token
    ../SysPrompt   — (optional) custom orchestrator system prompt override
"""
import argparse
import http.server
import json
import pathlib
import socketserver
import sys
import threading
import webbrowser
from http import HTTPStatus

from orchestrator import Orchestrator
import config

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           🤖  Multi-Agent Coding Assistant v2               ║
║           Powered by DeepSeek v4 Pro via Puter               ║
╠══════════════════════════════════════════════════════════════╣
║  Orchestrator  →  CodeExpert  /  FileExpert                  ║
║                →  ShellExpert /  DebugExpert                 ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:  exit | quit | /clear  (wipe conversation)        ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek multi-agent assistant runner."
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run using the terminal CLI UI.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run using the browser UI via ui.html.",
    )
    parser.add_argument(
        "--model",
        choices=config.AVAILABLE_MODELS,
        default=config.MODEL,
        help="Select the model to use.",
    )
    return parser.parse_args()


def run_cli(orchestrator: Orchestrator) -> None:
    print(BANNER)
    print(f"✅  Token:     {len(orchestrator._token)} chars (starts: {orchestrator._token[:4]}…)")
    print(f"🌐  Model:     {config.MODEL}")
    print(f"📁  Workspace: {config.WORKSPACE}")
    print(f"🔁  Max loops: {config.MAX_TOOL_ITERATIONS} per agent\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break

        if user_input.lower() == "/clear":
            orchestrator._conversation.clear()
            print("🗑️   Conversation history cleared.\n")
            continue

        orchestrator.chat(user_input)
        print()   # blank line between turns


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BrowserHandler(http.server.BaseHTTPRequestHandler):
    orchestrator: Orchestrator

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_path: pathlib.Path) -> None:
        if not html_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "UI file not found")
            return
        body = html_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        if not raw_body:
            return {}
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return {}

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/ui.html"}:
            self._send_html(config.BASE_DIR / "ui.html")
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/chat":
            payload = self._parse_json()
            if not payload or "message" not in payload:
                self.send_error(HTTPStatus.BAD_REQUEST, "'message' is required")
                return
            model = payload.get("model")
            if model:
                model_str = str(model).strip()
                if model_str not in config.AVAILABLE_MODELS:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid model selection")
                    return
                config.MODEL = model_str
            message = str(payload["message"]).strip()
            if not message:
                self.send_error(HTTPStatus.BAD_REQUEST, "'message' must not be empty")
                return

            reply = self.orchestrator.chat(message)
            self._send_json({"reply": reply})
            return

        if path == "/clear":
            self.orchestrator._conversation.clear()
            self._send_json({"status": "ok"})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        # Suppress default logging for a cleaner terminal output.
        return


def run_web(orchestrator: Orchestrator, port: int = 0) -> None:
    handler = BrowserHandler
    handler.orchestrator = orchestrator

    with ThreadedHTTPServer(("127.0.0.1", port), handler) as server:
        host, actual_port = server.server_address
        url = f"http://{host}:{actual_port}/"
        print(BANNER)
        print(f"🌐  Starting browser UI at {url}")
        print("🔐  Loading token and hosting local interface...\n")

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            webbrowser.open(url)
        except Exception:
            print("⚠️  Could not open browser automatically. Open the URL above manually.")

        try:
            thread.join()
        except KeyboardInterrupt:
            print("\n👋  Shutting down browser UI...")
            server.shutdown()


def main() -> None:
    args = parse_args()
    if args.cli and args.web:
        print("❌  Please choose only one mode: --cli or --web")
        sys.exit(2)

    try:
        token = config.load_token()
    except FileNotFoundError as exc:
        print(f"❌  {exc}")
        sys.exit(1)

    config.MODEL = args.model
    orchestrator = Orchestrator(token=token)

    if args.web:
        run_web(orchestrator)
        return

    run_cli(orchestrator)


if __name__ == "__main__":
    main()
