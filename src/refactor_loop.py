"""
src/refactor_loop.py — Runs an autonomous refactor loop in a sandbox.
"""
import threading
import time
from sandbox import setup_sandbox, revert_sandbox, commit_and_merge
from orchestrator import Orchestrator

class RefactorLoop(threading.Thread):
    def __init__(self, token: str):
        super().__init__(daemon=True)
        self.token = token
        self.running = False

    def run(self):
        self.running = True
        setup_sandbox()
        orch = Orchestrator(self.token)
        self.completed_tasks = []
        self.current_task = "None"

        while self.running:
            # 1. Ask Orchestrator for refactoring task
            self.current_task = "Identifying debt"
            completed_str = ", ".join(self.completed_tasks)
            # Add a constraint to prevent infinite loops on the same task
            prompt = f"Identify one new, concrete technical debt item (e.g., specific file, specific function refactor) NOT in this list: [{completed_str}]. Perform the refactor in the sandbox. If no more technical debt exists, report 'SUCCESS_FINISHED'."
            result = orch.chat(prompt)

            # 2. Check if refactor succeeded
            if "SUCCESS_FINISHED" in result.upper():
                self._log("\n✨ All refactoring complete. Stopping loop.")
                self.running = False
            elif "success" in result.lower():
                # Extract task name from result if possible, or use a placeholder
                task_name = result.split(":")[-1].strip()[:50]
                self.completed_tasks.append(task_name)
                commit_and_merge()
            else:
                revert_sandbox()

            self.current_task = "Idle"
            time.sleep(60)

    def _log(self, msg: str) -> None:
        print(msg)

