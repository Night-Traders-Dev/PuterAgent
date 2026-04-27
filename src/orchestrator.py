"""
orchestrator.py — Master coordinator that routes tasks to expert agents.

The orchestrator itself is an agent whose "tools" are the four expert agents.
It receives the user's message, decides which expert(s) to call (via the LLM),
dispatches to them, and prints a synthesised final answer.

Conversation history is maintained across turns so the orchestrator has full context.
"""
from __future__ import annotations
import config
from base_agent import BaseAgent
from code_expert import CodeExpert
from file_expert import FileExpert
from shell_expert import ShellExpert
from debug_expert import DebugExpert
from registry import Tool, ToolParam

ORCHESTRATOR_PROMPT = """\
You are a senior engineering lead coordinating a team of specialised AI agents.

YOUR TEAM
  • CodeExpert   — writing, reading, reviewing, refactoring, and testing code
  • FileExpert   — exploring project structure, reading/writing/organising files
  • ShellExpert  — git, builds, test runners, package management, CLI tools
  • DebugExpert  — diagnosing errors, tracing bugs, fixing issues with root-cause rigour

YOUR ROLE
  1. Understand exactly what the user wants.
  2. Break complex requests into clear sub-tasks.
  3. Delegate each sub-task to the best expert with a precise, self-contained description.
  4. Chain experts when needed (e.g. FileExpert reads a file → CodeExpert modifies it → ShellExpert runs tests).
  5. Synthesise the expert results into a clear, concise response for the user.

DELEGATION RULES
  • Always delegate — never answer technical questions from your own knowledge.
  • Include all relevant context in each delegation: file paths, error messages, code snippets.
  • If expert A's output is needed by expert B, include it verbatim in B's task description.
  • For ambiguous requests, delegate to FileExpert first to understand the project structure.
  • For errors, always delegate to DebugExpert with the full stack trace.
"""


class Orchestrator(BaseAgent):
    def __init__(self, token: str, profile: dict | None = None) -> None:
        system_prompt = ORCHESTRATOR_PROMPT
        if profile:
            profile_lines = [f"{k.title()}: {v}" for k, v in profile.items() if v and k != "theme"]
            system_prompt += "\n\nUSER PROFILE\n" + "\n".join(profile_lines)
            system_prompt += "\n\nINSTRUCTIONS\n- Always refer to the user by name when available.\n- Remember the user’s role, preferences, and orientation.\n- Adapt answers to the user’s background and keep recommendations actionable.\n"

        super().__init__(
            name="Orchestrator",
            system_prompt=system_prompt,
            token=token,
        )
        self._token        = token
        self._conversation: list[dict] = []   # persistent across chat() calls
        self.last_turn_metrics = self.turn_metrics()
        self._register_routing_tools()

    def set_profile(self, profile: dict | None) -> None:
        if not profile:
            return
        profile_lines = [f"{k.title()}: {v}" for k, v in profile.items() if v and k != "theme"]
        system_prompt = ORCHESTRATOR_PROMPT + "\n\nUSER PROFILE\n" + "\n".join(profile_lines)
        system_prompt += "\n\nINSTRUCTIONS\n- Always refer to the user by name when available.\n- Remember the user’s role, preferences, and orientation.\n- Adapt answers to the user’s background and keep recommendations actionable.\n"
        self.system_prompt = system_prompt

    # ── Register expert delegation as callable tools ──────────────────────────

    def _register_routing_tools(self) -> None:

        def _delegate(ExpertClass, label: str, task: str) -> str:
            print(f"\n  ➤  Delegating to {label}: {task[:90]}{'…' if len(task)>90 else ''}")
            expert = ExpertClass(self._token)
            result = expert.run(task, verbose=True)
            self._merge_turn_metrics_from(expert)
            print(f"\n  ✓  {label} finished.")
            return result

        routing_tools = [
            Tool(
                name="delegate_to_code_expert",
                description=(
                    "WHEN: Any task involving writing new code, reading/understanding existing code, "
                    "implementing features, reviewing for bugs or style, refactoring, adding tests, "
                    "or producing documentation. "
                    "HOW: Write a clear task description that includes the programming language, "
                    "target file paths, exact requirements, and any constraints."
                ),
                fn=lambda task: _delegate(CodeExpert, "CodeExpert", task),
                params={"task": ToolParam("string", "Full description of the coding task with file paths and requirements")},
            ),
            Tool(
                name="delegate_to_file_expert",
                description=(
                    "WHEN: Exploring an unknown project structure, batch file operations, managing "
                    "config files, searching for content across many files, or organising directories. "
                    "Use this first when the user mentions a project you haven't seen yet. "
                    "HOW: Describe what to find, read, create, or reorganise and why."
                ),
                fn=lambda task: _delegate(FileExpert, "FileExpert", task),
                params={"task": ToolParam("string", "Full description of the file management task")},
            ),
            Tool(
                name="delegate_to_shell_expert",
                description=(
                    "WHEN: Running builds, test suites, linters, formatters, git operations, "
                    "installing packages, scaffolding projects, or any CLI/terminal task. "
                    "HOW: Describe the goal and any specific commands or tools expected. "
                    "Include the project root path if relevant."
                ),
                fn=lambda task: _delegate(ShellExpert, "ShellExpert", task),
                params={"task": ToolParam("string", "Full description of the shell/DevOps task")},
            ),
            Tool(
                name="delegate_to_debug_expert",
                description=(
                    "WHEN: Any error, exception, unexpected output, failing test, or bug report. "
                    "Also use for performance analysis or investigating unexpected behaviour. "
                    "HOW: Include the FULL error message, complete stack trace, "
                    "the file and line number where it originates, and any relevant code context."
                ),
                fn=lambda task: _delegate(DebugExpert, "DebugExpert", task),
                params={"task": ToolParam("string", "Error description with full stack trace and code context")},
            ),
        ]

        for tool in routing_tools:
            self.registry.register(tool)

    # ── Public chat interface ─────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Process one user turn.
        - Appends to persistent conversation history.
        - Runs orchestrator tool loop (dispatching to experts as needed).
        - Prints the final synthesised answer to stdout.
        - Returns the final answer string.
        """
        self._reset_turn_metrics()
        self._conversation.append({"role": "user", "content": user_input})

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._conversation,
        ]

        for iteration in range(config.MAX_TOOL_ITERATIONS):
            data = self._call(messages)
            if data is None:
                err = f"Orchestrator: {self._last_error or 'API call failed.'}"
                self._conversation.append({"role": "assistant", "content": err})
                self.last_turn_metrics = self.turn_metrics()
                return err

            text, tool_calls = self._parse_response(data)

            # ── Final response ────────────────────────────────────────────────
            if not tool_calls:
                final = text or "Orchestrator: Empty response from model."
                messages.append({"role": "assistant", "content": final})
                print("\n🤖 Orchestrator: ", end="", flush=True)
                print(final)
                self._conversation.append({"role": "assistant", "content": final})
                self.last_turn_metrics = self.turn_metrics()
                return final

            # ── Routing decisions — log and execute ───────────────────────────
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "?")
                print(f"\n  🔀 Orchestrator routing → {fn_name}")

            messages.append({
                "role":       "assistant",
                "content":    text or "",
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                result = self.registry.dispatch(
                    tc["function"]["name"],
                    tc["function"].get("arguments", "{}"),
                )
                messages.append({
                    "role":         "tool",
                    "tool_call_id":  tc["id"],
                    "content":       result,
                })

        err = "Orchestrator: Max tool iterations reached."
        self._conversation.append({"role": "assistant", "content": err})
        self.last_turn_metrics = self.turn_metrics()
        return err
