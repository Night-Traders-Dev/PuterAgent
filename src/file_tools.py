"""
tools/file_tools.py — Safe file system operations.

SECURITY: every path is resolved and checked against WORKSPACE before any
I/O. Attempts to escape the workspace (e.g. ../../etc/passwd) are blocked.
"""
from __future__ import annotations
import pathlib
from registry import Tool, ToolParam
from config import WORKSPACE

MAX_READ_BYTES = 1_000_000   # 1 MB read cap


# ── Path guard ────────────────────────────────────────────────────────────────

def _safe_path(rel: str) -> pathlib.Path:
    """Resolve *rel* relative to WORKSPACE; raise if it escapes the sandbox."""
    resolved = (WORKSPACE / rel).resolve()
    if not str(resolved).startswith(str(WORKSPACE.resolve())):
        raise PermissionError(f"Path traversal blocked: '{rel}'")
    return resolved


# ── Implementations ───────────────────────────────────────────────────────────

def _read_file(path: str, encoding: str = "utf-8") -> dict:
    try:
        p = _safe_path(path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        size = p.stat().st_size
        if size > MAX_READ_BYTES:
            return {"success": False, "error": f"File too large ({size} bytes). Use run_shell with head/tail/sed."}
        content = p.read_text(encoding=encoding, errors="replace")
        return {"success": True, "path": str(p), "content": content, "lines": len(content.splitlines())}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _write_file(path: str, content: str, mode: str = "write") -> dict:
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with p.open("a", encoding="utf-8") as fh:
                fh.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(p), "bytes_written": len(content.encode())}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


from config import WORKSPACE
import functools

# Cache for list_directory (TTL: 5s)
@functools.lru_cache(maxsize=16)
def _cached_list_dir(path: str, recursive: bool, mtime: float) -> dict:
    # This dummy mtime argument ensures invalidation when file system changes
    return _list_directory_impl(path, recursive)

def _list_directory_impl(path: str, recursive: bool) -> dict:
    p = _safe_path(path)
    if not p.is_dir():
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

def _list_directory(path: str = ".", recursive: bool = False) -> dict:
    try:
        p = _safe_path(path)
        mtime = p.stat().st_mtime if p.exists() else 0.0
        # Quantize mtime to 5s windows for cache invalidation
        mtime_window = int(mtime / 5)
        return _cached_list_dir(path, recursive, mtime_window)
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _delete_file(path: str) -> dict:
    try:
        p = _safe_path(path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if p.is_dir():
            return {"success": False, "error": "Won't delete directories. Use run_shell for that."}
        p.unlink()
        return {"success": True, "deleted": str(p)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Tool definitions ──────────────────────────────────────────────────────────

READ_FILE_TOOL = Tool(
    name="read_file",
    description=(
        "WHEN: Use to read the contents of any text file — source code, configs, logs, "
        "READMEs, JSON, YAML. Call this before editing a file so you preserve existing content. "
        "HOW: Provide the path relative to the workspace root. "
        "Returns the full text and line count. Files over 1 MB are rejected; use run_shell "
        "with head/tail/sed/grep for large files instead."
    ),
    fn=_read_file,
    params={
        "path":     ToolParam("string", "Workspace-relative file path to read"),
        "encoding": ToolParam("string", "Character encoding (default: utf-8)", required=False),
    },
)

WRITE_FILE_TOOL = Tool(
    name="write_file",
    description=(
        "WHEN: Use to create a new file or fully overwrite an existing one. "
        "Use mode='append' to add content without clobbering existing lines (e.g. log entries). "
        "HOW: Provide the path and complete text content. "
        "Parent directories are created automatically. "
        "Always read_file first before overwriting to avoid losing existing content."
    ),
    fn=_write_file,
    params={
        "path":    ToolParam("string", "Workspace-relative path to write (created if absent)"),
        "content": ToolParam("string", "Full text content to write"),
        "mode":    ToolParam("string", "'write' (overwrite, default) or 'append'", required=False),
    },
)

LIST_DIRECTORY_TOOL = Tool(
    name="list_directory",
    description=(
        "WHEN: Use at the start of any task to understand project structure. "
        "Use before reading/writing files to discover what exists. "
        "HOW: Provide a path (default: workspace root). "
        "Set recursive=true for a full file tree. Returns name, type (file/dir), and size."
    ),
    fn=_list_directory,
    params={
        "path":      ToolParam("string",  "Directory path to list (default: '.')", required=False),
        "recursive": ToolParam("boolean", "Include all nested entries (default: false)", required=False),
    },
)

DELETE_FILE_TOOL = Tool(
    name="delete_file",
    description=(
        "WHEN: Use only when explicitly asked to delete a specific file. "
        "This action is irreversible — confirm intent before calling. "
        "HOW: Provide the file path. Only works on files, not directories."
    ),
    fn=_delete_file,
    params={
        "path": ToolParam("string", "Workspace-relative path of the file to delete"),
    },
)

FILE_TOOLS = [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_DIRECTORY_TOOL, DELETE_FILE_TOOL]
