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
LOCAL_API_URL   = "http://localhost:11434/api/chat"
CLOUD_API_URL   = "https://api.puter.com/v1/ai/chat"
USAGE_API_URL   = "https://api.puter.com/drivers/usage"
API_URL         = LOCAL_API_URL  # Default to local

MEMORY_PATH     = BASE_DIR.parent / ".memory"

LOCAL_MODELS = [
    "qwen2.5-coder:14b",
    "qwen2.5-coder:7b",
    "deepseek-coder:6.7b",
    "neural-chat:7b",
    "mistral:7b",
    "llama2:13b",
]

CLOUD_MODELS = [
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

AVAILABLE_MODELS = LOCAL_MODELS + CLOUD_MODELS

DEFAULT_USER_PROFILE = {
    "name": "Developer",
    "pronouns": "they/them",
    "role": "Software engineer",
    "company": "",
    "timezone": "",
    "bio": "A practical, detail-oriented software engineer who appreciates concise technical answers and clear explanations.",
    "preferences": "Use my name, remember my role and preferences, and keep recommendations actionable.",
    "theme": "sunset-boulevard",
}
MODEL           = "qwen2.5-coder:14b"
TOOL_FALLBACK_MODELS = [
    "qwen2.5-coder:7b",
    "deepseek-coder:6.7b",
]
REQUEST_TIMEOUT = 90          # seconds per HTTP request
MAX_TOOL_ITERATIONS = 12      # hard cap on agentic loops per agent
MAX_ATTACHMENTS = 6
MAX_ATTACHMENT_CHARS = 120_000
MAX_TOTAL_ATTACHMENT_CHARS = 360_000

API_HEADERS = {
    "Content-Type": "application/json",
}

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_token() -> str:
    if SECRET_PATH.exists():
        try:
            token = SECRET_PATH.read_text("utf-8").strip()
            if token:
                return token
        except Exception:
            pass
    return "ollama-local"


def load_orchestrator_prompt() -> str:
    if SYSPROMPT_PATH.exists():
        return SYSPROMPT_PATH.read_text("utf-8").strip()
    return (
        "You are a senior engineering orchestrator managing a team of expert AI agents. "
        "Break down complex tasks and delegate precisely."
    )
