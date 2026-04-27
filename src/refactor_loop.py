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
        while self.running:
            # 1. Ask Orchestrator for refactoring task
            result = orch.chat("Identify one technical debt item and refactor it in the sandbox.")
            
            # 2. Check if refactor succeeded (would involve test check)
            if "success" in result.lower():
                commit_and_merge()
            else:
                revert_sandbox()
            
            time.sleep(10) # 10s overhead loop
