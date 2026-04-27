"""
src/sandbox.py — Manages a Git worktree for sandboxed refactoring.
"""
import subprocess
import pathlib
import config

SANDBOX_DIR = config.BASE_DIR.parent / ".sandbox"

def setup_sandbox():
    """Initializes a new git worktree for the sandbox."""
    if not SANDBOX_DIR.exists():
        # Check if worktree directory exists but git index doesn't
        try:
            subprocess.run(["git", "worktree", "add", str(SANDBOX_DIR), "main"], check=True)
        except subprocess.CalledProcessError:
            # If main is already checked out, we can use a orphan branch for the sandbox
            subprocess.run(["git", "worktree", "add", "--detach", str(SANDBOX_DIR)], check=True)

def revert_sandbox():
    """Discards all changes in the sandbox."""
    subprocess.run(["git", "-C", str(SANDBOX_DIR), "reset", "--hard", "HEAD"], check=True)
    subprocess.run(["git", "-C", str(SANDBOX_DIR), "clean", "-fd"], check=True)

def commit_and_merge():
    """Commits changes in sandbox and merges to main."""
    subprocess.run(["git", "-C", str(SANDBOX_DIR), "add", "."], check=True)
    subprocess.run(["git", "-C", str(SANDBOX_DIR), "commit", "-m", "Auto-refactor: improved codebase."], check=True)
    subprocess.run(["git", "merge", "main"], cwd=SANDBOX_DIR, check=True)
