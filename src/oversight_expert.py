"""
agents/oversight_expert.py — Supervises the orchestrator to ensure progress.
"""
from base_agent import BaseAgent
import config
import json

OVERSIGHT_PROMPT = """
You are the OversightExpert. Your job is to monitor the Orchestrator's progress.
If the Orchestrator has reached the maximum iteration limit, analyze the conversation history.

- Did the agent make constructive progress (writing code, fixing files, passing tests)?
- Is it stuck in a loop or hallucinating tool calls?

Output ONLY a JSON object:
{
  "action": "increase_limit" | "abort",
  "reason": "Brief explanation",
  "score": 0-10
}
"""

class OversightExpert(BaseAgent):
    def __init__(self, token: str) -> None:
        super().__init__(
            name="OversightExpert",
            system_prompt=OVERSIGHT_PROMPT,
            token=token,
        )

    def run(self, history_summary: str, verbose: bool = False) -> str:
        messages = [{"role": "user", "content": f"Analyze this Orchestrator session progress:\n{history_summary}"}]
        resp = self._stream(messages)
        return resp
