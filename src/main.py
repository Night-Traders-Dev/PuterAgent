"""
main.py — Entry point for PuterAgent.

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
import time
import urllib.parse
import webbrowser
from http import HTTPStatus

from orchestrator import Orchestrator
from file_tools import _safe_path
from shell_tools import _run_shell
import config

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                    PuterAgent v2                            ║
║           Multi-model coding agents via Puter               ║
╠══════════════════════════════════════════════════════════════╣
║  Orchestrator  →  CodeExpert  /  FileExpert                  ║
║                →  ShellExpert /  DebugExpert                 ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:  exit | quit | /clear  (wipe conversation)        ║
╚══════════════════════════════════════════════════════╝
"""


class MemoryManager:
    def __init__(self, base_path: pathlib.Path | None = None) -> None:
        self.base = base_path or config.MEMORY_PATH
        self.profile_path = self.base / "profile.json"
        self.sessions_dir = self.base / "sessions"
        self.base.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: pathlib.Path, default: object) -> object:
        try:
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json(self, path: pathlib.Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _sanitize_model(self, model: str) -> str:
        return str(model).replace("/", "_").replace(" ", "_")

    def _session_path(self, model: str) -> pathlib.Path:
        return self.sessions_dir / f"{self._sanitize_model(model)}-current.json"

    def _archive_dir(self, model: str) -> pathlib.Path:
        return self.sessions_dir / self._sanitize_model(model) / "archive"

    def load_profile(self) -> dict:
        profile = self._read_json(self.profile_path, default=config.DEFAULT_USER_PROFILE.copy())
        if not isinstance(profile, dict):
            profile = config.DEFAULT_USER_PROFILE.copy()
        if "theme" not in profile:
            profile["theme"] = config.DEFAULT_USER_PROFILE["theme"]
        self.save_profile(profile)
        return profile

    def save_profile(self, profile: dict) -> None:
        profile = {**config.DEFAULT_USER_PROFILE, **profile}
        self._write_json(self.profile_path, profile)

    def load_history(self, model: str) -> list[dict]:
        payload = self._read_json(self._session_path(model), default={
            "model": model,
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
        })
        if not isinstance(payload, dict):
            payload = {"model": model, "created_at": time.time(), "updated_at": time.time(), "messages": []}
        return payload.get("messages", [])

    def save_history(self, model: str, messages: list[dict]) -> None:
        payload = {
            "model": model,
            "updated_at": time.time(),
            "messages": messages,
        }
        self._write_json(self._session_path(model), payload)

    def clear_history(self, model: str) -> None:
        messages = self.load_history(model)
        archive_dir = self._archive_dir(model)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_file = archive_dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}.json"
        self._write_json(archive_file, {"model": model, "archived_at": time.time(), "messages": messages})
        self.save_history(model, [])

    def available_themes(self) -> list[str]:
        themes_path = config.BASE_DIR.parent / "themes"
        if not themes_path.exists():
            return [config.DEFAULT_USER_PROFILE["theme"]]
        return sorted([f.stem for f in themes_path.glob("*.css") if f.is_file()]) or [config.DEFAULT_USER_PROFILE["theme"]]

    def theme_path(self, theme_name: str) -> pathlib.Path:
        return config.BASE_DIR.parent / "themes" / f"{theme_name}.css"

    def list_directory(self, path: str, recursive: bool = False) -> dict:
        try:
            p = _safe_path(path)
            if not p.exists() or not p.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}
            iterator = p.rglob("*") if recursive else p.iterdir()
            entries = sorted(
                [
                    {
                        "name": str(item.relative_to(p)) if recursive else item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    }
                    for item in iterator
                ],
                key=lambda x: (x["type"] == "file", x["name"]),
            )
            return {"success": True, "path": str(p), "count": len(entries), "entries": entries}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def analyze_path(self, target: str) -> dict:
        if target.lower().startswith("http://") or target.lower().startswith("https://"):
            if "github.com" not in target:
                return {"success": False, "error": "Only GitHub repository URLs are supported for remote analysis."}
            repo_name = target.rstrip("/\n").split("/")[-1]
            clone_path = config.WORKSPACE / repo_name
            if clone_path.exists():
                result = _run_shell(f"git -C {clone_path} pull --ff-only")
                if not result.get("success"):
                    return {"success": False, "error": result.get("stderr") or result.get("stdout")}
            else:
                result = _run_shell(f"git clone --depth 1 {target} {repo_name}")
                if not result.get("success"):
                    return {"success": False, "error": result.get("stderr") or result.get("stdout")}
            return self.list_directory(str(clone_path), recursive=True)

        return self.list_directory(target, recursive=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PuterAgent multi-agent assistant runner."
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
    history: list[dict] = []
    session_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    state_lock = threading.Lock()

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _add_usage(target: dict[str, int], usage: dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            target[key] = int(target.get(key, 0)) + int(usage.get(key, 0))

    @staticmethod
    def _safe_attachment_name(value: object) -> str:
        name = str(value or "attachment").replace("\\", "/").split("/")[-1].strip()
        return name[:120] or "attachment"

    def _coerce_attachments(self, payload: dict) -> list[dict]:
        raw_attachments = payload.get("attachments", [])
        if not isinstance(raw_attachments, list):
            return []

        attachments: list[dict] = []
        total_chars = 0
        for raw in raw_attachments[:config.MAX_ATTACHMENTS]:
            if not isinstance(raw, dict):
                continue

            content = str(raw.get("content") or "")
            remaining = max(config.MAX_TOTAL_ATTACHMENT_CHARS - total_chars, 0)
            if remaining <= 0:
                break

            max_chars = min(config.MAX_ATTACHMENT_CHARS, remaining)
            truncated = bool(raw.get("truncated")) or len(content) > max_chars
            content = content[:max_chars]
            total_chars += len(content)

            attachments.append({
                "name": self._safe_attachment_name(raw.get("name")),
                "type": str(raw.get("type") or "text/plain")[:80],
                "size": int(raw.get("size") or len(content)),
                "content": content,
                "truncated": truncated,
            })

        return attachments

    @staticmethod
    def _attachment_summaries(attachments: list[dict]) -> list[dict]:
        return [
            {
                "name": item["name"],
                "type": item["type"],
                "size": item["size"],
                "truncated": item["truncated"],
            }
            for item in attachments
        ]

    def _message_with_attachments(self, message: str, attachments: list[dict]) -> str:
        if not attachments:
            return message

        sections = [message or "Use the attached files as context for this turn."]
        sections.append(
            "\nUploaded file contents are included inline below. "
            "They are not workspace paths unless the user explicitly says they are."
        )
        for item in attachments:
            truncation = " truncated" if item["truncated"] else ""
            sections.append(
                f"\n--- Uploaded file: {item['name']} ({item['type']}, {item['size']} bytes{truncation}) ---\n"
                f"{item['content']}"
            )
        return "\n".join(sections)

    def _state_payload(self) -> dict:
        return {
            "messages": type(self).history,
            "model": config.MODEL,
            "session_metrics": dict(type(self).session_usage),
            "profile": self.memory_manager.load_profile(),
            "available_models": config.AVAILABLE_MODELS,
            "themes": self.memory_manager.available_themes(),
        }

    def _load_model_history(self, model: str) -> None:
        messages = self.memory_manager.load_history(model)
        self.orchestrator._conversation = list(messages)
        type(self).history = list(messages)

    def _refresh_profile_prompt(self) -> None:
        profile = self.memory_manager.load_profile()
        if hasattr(self.orchestrator, "set_profile"):
            self.orchestrator.set_profile(profile)

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

        if path.startswith("/themes/"):
            theme_name = path.split("/", 2)[-1]
            theme_path = self.memory_manager.theme_path(theme_name)
            if theme_path.exists():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/css; charset=utf-8")
                body = theme_path.read_bytes()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Theme not found")
            return

        if path == "/history":
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            model = params.get("model", [config.MODEL])[0]
            if model not in config.AVAILABLE_MODELS:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid model selection")
                return
            config.MODEL = model
            self._load_model_history(model)
            with self.state_lock:
                self._send_json(self._state_payload())
            return

        if path == "/profile":
            with self.state_lock:
                self._send_json({
                    "profile": self.memory_manager.load_profile(),
                    "themes": self.memory_manager.available_themes(),
                })
            return

        if path == "/themes":
            self._send_json({"themes": self.memory_manager.available_themes()})
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
                if model_str != config.MODEL:
                    config.MODEL = model_str
                    self._load_model_history(model_str)
            message = str(payload["message"]).strip()
            attachments = self._coerce_attachments(payload)
            if not message and not attachments:
                self.send_error(HTTPStatus.BAD_REQUEST, "'message' or attachments are required")
                return

            prompt = self._message_with_attachments(message, attachments)
            reply = self.orchestrator.chat(prompt)
            metrics = self.orchestrator.last_turn_metrics
            usage = metrics.get("usage", self._empty_usage())
            used_model = metrics.get("last_model") or config.MODEL

            created_at = time.time()
            attachment_summaries = self._attachment_summaries(attachments)
            user_record = {
                "id": f"msg-{int(created_at * 1000)}-user",
                "role": "user",
                "content": message,
                "attachments": attachment_summaries,
                "model": config.MODEL,
                "created_at": created_at,
            }
            assistant_record = {
                "id": f"msg-{int(created_at * 1000)}-assistant",
                "role": "assistant",
                "content": reply,
                "model": config.MODEL,
                "used_model": used_model,
                "models_used": metrics.get("models", []),
                "metrics": usage,
                "created_at": time.time(),
            }

            with self.state_lock:
                type(self).history.extend([user_record, assistant_record])
                self._add_usage(type(self).session_usage, usage)
                session_usage = dict(type(self).session_usage)
                self.memory_manager.save_history(config.MODEL, type(self).history)

            self._send_json({
                "reply": reply,
                "model": config.MODEL,
                "used_model": used_model,
                "models_used": metrics.get("models", []),
                "metrics": usage,
                "session_metrics": session_usage,
                "messages": [user_record, assistant_record],
            })
            return

        if path == "/clear":
            self.orchestrator._conversation.clear()
            self.memory_manager.clear_history(config.MODEL)
            with self.state_lock:
                type(self).history.clear()
                type(self).session_usage = self._empty_usage()
                session_usage = dict(type(self).session_usage)
            self._send_json({"status": "ok", "session_metrics": session_usage})
            return

        if path == "/profile":
            payload = self._parse_json()
            if not isinstance(payload, dict):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid profile payload")
                return
            profile = self.memory_manager.load_profile()
            profile.update({k: v for k, v in payload.items() if k in profile})
            if "theme" in payload:
                profile["theme"] = str(payload["theme"]).strip() or profile.get("theme", config.DEFAULT_USER_PROFILE["theme"])
            self.memory_manager.save_profile(profile)
            self._refresh_profile_prompt()
            self._send_json({"status": "ok", "profile": profile})
            return

        if path == "/analyze":
            payload = self._parse_json()
            target = str(payload.get("target") or payload.get("path") or "").strip()
            if not target:
                self.send_error(HTTPStatus.BAD_REQUEST, "'target' or 'path' is required")
                return
            result = self.memory_manager.analyze_path(target)
            if not result.get("success"):
                self.send_error(HTTPStatus.BAD_REQUEST, result.get("error", "Analysis failed"))
                return
            self._send_json({"status": "ok", "analysis": result})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        # Suppress default logging for a cleaner terminal output.
        return


def run_web(orchestrator: Orchestrator, memory_manager: MemoryManager, port: int = 0) -> None:
    handler = BrowserHandler
    handler.orchestrator = orchestrator
    handler.memory_manager = memory_manager
    handler.history = memory_manager.load_history(config.MODEL)
    orchestrator._conversation = list(handler.history)
    handler.session_usage = BrowserHandler._empty_usage()
    handler.state_lock = threading.Lock()

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
    memory_manager = MemoryManager()
    profile = memory_manager.load_profile()
    orchestrator = Orchestrator(token=token, profile=profile)
    orchestrator.set_profile(profile)

    if args.web:
        run_web(orchestrator, memory_manager)
        return

    run_cli(orchestrator)


if __name__ == "__main__":
    main()
