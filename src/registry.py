"""
tools/registry.py — Central registry for all agent tools.

Each Tool carries its LLM-facing schema (name, WHEN/HOW description, parameter
types) alongside the Python callable. Agents build a ToolRegistry from a
subset of tools relevant to their specialisation.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ToolParam:
    """Describes one parameter of a Tool for the LLM schema."""
    type:        str            # "string" | "integer" | "boolean" | "array" | "object"
    description: str
    required:    bool = True
    enum:        list[str] | None = None


@dataclass
class Tool:
    """
    A callable tool with a self-describing LLM schema.

    `description` must answer two questions the model will ask:
      WHEN — which situations call for this tool?
      HOW  — what input format does it expect?
    """
    name:        str
    description: str
    fn:          Callable
    params:      dict[str, ToolParam] = field(default_factory=dict)

    # ── Schema generation (OpenAI-compatible function-call format) ────────────

    def to_schema(self) -> dict:
        properties: dict = {}
        required:   list = []
        for pname, p in self.params.items():
            prop: dict = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[pname] = prop
            if p.required:
                required.append(pname)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def execute(self, **kwargs) -> str:
        """Run the tool and always return a JSON string."""
        try:
            result = self.fn(**kwargs)
            return json.dumps(result) if isinstance(result, dict) else str(result)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Holds the tools available to one agent instance."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._cached_schemas: list[dict] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_schemas = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        """Return OpenAI-compatible tool schemas for the API payload."""
        if self._cached_schemas is None:
            self._cached_schemas = [t.to_schema() for t in self._tools.values()]
        return self._cached_schemas

    def dispatch(self, name: str, arguments: str | dict) -> str:
        """Execute a tool by name, decoding JSON arguments."""
        tool = self.get(name)
        if not tool:
            return json.dumps({"success": False, "error": f"Unknown tool: '{name}'"})
        try:
            if isinstance(arguments, dict):
                kwargs = arguments
            else:
                kwargs = json.loads(arguments) if arguments.strip() else {}
            return tool.execute(**kwargs)
        except json.JSONDecodeError as exc:
            return json.dumps({"success": False, "error": f"Malformed arguments JSON: {exc}"})
