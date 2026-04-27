"""
config.py — Central configuration. All tunables live here.
"""
import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent
SECRET_PATH = BASE_DIR.parent / "secret"
SYSPROMPT_PATH = BASE_DIR.parent / "SysPrompt"
WORKSPACE   = pathlib.Path.cwd()   # All file tools are scoped to this dir

# ── API ───────────────────────────────────────────────────────────────────────
API_URL         = "https://api.puter.com/puterai/openai/v1/chat/completions"
MEMORY_PATH     = BASE_DIR.parent / ".memory"
AVAILABLE_MODELS = [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1-0528",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-6",
    "openai/gpt-5.5-pro",
    "moonshotai/kimi-k2.6",
    "x-ai/grok-4.20-multi-agent",
    "minimax/minimax-m2.7",
    "nvidia/nemotron-3-super-120b-a12b",
    "google/gemini-3.1-pro-preview",
]
DEFAULT_USER_PROFILE = {
    "name": "Developer",
    "pronouns": "they/them",
    "role": "Software engineer",
    "company": "",
    "timezone": "",
    "bio": "A practical, detail-oriented software engineer who appreciates concise technical answers and clear explanations.",
    "preferences": "Use my name, remember my role and preferences, and keep recommendations actionable.",
    "theme": "default",
}
MODEL           = "deepseek/deepseek-v4-pro"
TOOL_FALLBACK_MODELS = [
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat",
    "openai/gpt-5.5",
]
REQUEST_TIMEOUT = 90          # seconds per HTTP request
MAX_TOOL_ITERATIONS = 12      # hard cap on agentic loops per agent
MAX_ATTACHMENTS = 6
MAX_ATTACHMENT_CHARS = 120_000
MAX_TOTAL_ATTACHMENT_CHARS = 360_000

API_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent":   "Mozilla/5.0 (PuterAgent/2.0)",
    "Origin":       "https://puter.com",
    "Referer":      "https://puter.com/",
}

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_token() -> str:
    if not SECRET_PATH.exists():
        raise FileNotFoundError(f"'secret' file not found: {SECRET_PATH}")
    return SECRET_PATH.read_text("utf-8").replace('\n', '').replace('\r', '').replace('"', '').replace("'", "")


def load_orchestrator_prompt() -> str:
    if SYSPROMPT_PATH.exists():
        return SYSPROMPT_PATH.read_text("utf-8").strip()
    return (
        "You are a senior engineering orchestrator managing a team of expert AI agents. "
        "Break down complex tasks and delegate precisely."
    )
