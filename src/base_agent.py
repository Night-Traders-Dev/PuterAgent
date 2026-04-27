"""
agents/base_agent.py — Core agentic loop shared by every expert and the orchestrator.

Responsibilities:
  1. Construct API payloads (messages + tool schemas).
  2. Parse both Puter-format and OpenAI-format responses.
  3. Run the tool-call loop until the model produces a text-only response.
  4. Print and return the final user-facing response.
"""
from __future__ import annotations
import json
import sys
import requests
from registry import ToolRegistry, Tool
import config


class BaseAgent:
    """
    All agents extend this class.

    Subclasses define:
      - ``name``          – displayed in log output
      - ``system_prompt`` – expert persona and rules
      - call ``register_tools(tools)`` in ``__init__``
    """

    def __init__(self, name: str, system_prompt: str, token: str) -> None:
        self.name          = name
        self.system_prompt = system_prompt
        self.registry      = ToolRegistry()
        self._headers      = {**config.API_HEADERS, "Authorization": f"Bearer {token}"}
        self._last_error: str | None = None
        self._last_model: str | None = None
        self._last_usage = self._empty_usage()
        self._turn_usage = self._empty_usage()
        self._turn_models: list[str] = []

    # ── Tool registration ─────────────────────────────────────────────────────

    def register_tools(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.registry.register(tool)

    # ── API layer ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @classmethod
    def _normalise_usage(cls, data: object) -> dict[str, int]:
        if not isinstance(data, dict):
            return cls._empty_usage()

        usage = data.get("usage")
        if not isinstance(usage, dict):
            return cls._empty_usage()

        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        total = usage.get("total_tokens", prompt + completion) or 0
        return {
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
        }

    @staticmethod
    def _add_usage(target: dict[str, int], usage: dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            target[key] = int(target.get(key, 0)) + int(usage.get(key, 0))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)

    def _estimated_usage(self, messages: list[dict], data: dict) -> dict[str, int]:
        prompt_text = json.dumps(messages, ensure_ascii=False)
        text, tool_calls = self._parse_response(data)
        completion_text = text
        if tool_calls:
            completion_text += json.dumps(tool_calls, ensure_ascii=False)

        prompt_tokens = self._estimate_tokens(prompt_text)
        completion_tokens = self._estimate_tokens(completion_text)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _reset_turn_metrics(self) -> None:
        self._last_model = None
        self._last_usage = self._empty_usage()
        self._turn_usage = self._empty_usage()
        self._turn_models = []

    def _record_response_metadata(self, data: dict, fallback_model: str, messages: list[dict]) -> None:
        self._last_model = str(data.get("model") or fallback_model)
        self._last_usage = self._normalise_usage(data)
        if self._last_usage["total_tokens"] <= 0:
            self._last_usage = self._estimated_usage(messages, data)
        self._add_usage(self._turn_usage, self._last_usage)
        if self._last_model not in self._turn_models:
            self._turn_models.append(self._last_model)

    def _merge_turn_metrics_from(self, agent: "BaseAgent") -> None:
        self._add_usage(self._turn_usage, agent._turn_usage)
        for model in agent._turn_models:
            if model not in self._turn_models:
                self._turn_models.append(model)

    def turn_metrics(self) -> dict:
        return {
            "usage": dict(self._turn_usage),
            "models": list(self._turn_models),
            "last_model": self._last_model,
        }

    def _api_error_from_payload(self, data: object) -> str | None:
        """Return a readable error if the API produced an error-shaped payload."""
        if not isinstance(data, dict):
            return None

        error = data.get("error")
        message = data.get("message")
        code = data.get("code")

        if isinstance(error, dict):
            message = error.get("message") or message
            code = error.get("code") or code
        elif isinstance(error, str):
            message = error

        if data.get("success") is False or error or (message and code):
            parts = [str(message or "API request failed")]
            if code:
                parts.append(f"code: {code}")

            attempts = data.get("attempts")
            if isinstance(attempts, list) and attempts:
                summaries = []
                for attempt in attempts[:2]:
                    if not isinstance(attempt, dict):
                        continue
                    provider = attempt.get("provider") or attempt.get("model") or "provider"
                    detail = str(attempt.get("error") or "").replace("\n", " ")
                    if len(detail) > 180:
                        detail = detail[:177] + "..."
                    summaries.append(f"{provider}: {detail}")
                if summaries:
                    parts.append("; ".join(summaries))

            return " | ".join(parts)

        return None

    def _record_error(self, message: str) -> None:
        self._last_error = message
        print(f"\n[{self.name}] {message}")

    def _decode_json_response(self, resp: requests.Response) -> dict | None:
        try:
            data = resp.json()
        except ValueError:
            preview = resp.text[:400].replace("\n", " ")
            self._record_error(f"HTTP {resp.status_code}: non-JSON API response: {preview}")
            return None

        api_error = self._api_error_from_payload(data)
        if resp.status_code >= 400 or api_error:
            message = api_error or resp.text[:400].replace("\n", " ")
            self._record_error(f"HTTP {resp.status_code}: {message}")
            return None

        return data

    def _call(self, messages: list[dict]) -> dict | None:
        """Run one OpenAI-compatible chat-completion request."""

        self._last_error = None
        payload_base: dict = {
            "messages": messages,
            "stream":   False,
        }

        schemas = self.registry.schemas()
        if schemas:
            payload_base["tools"] = schemas
            payload_base["tool_choice"] = "auto"

        models = [config.MODEL]
        if schemas:
            for model in config.TOOL_FALLBACK_MODELS:
                if model not in models:
                    models.append(model)

        for attempt, model in enumerate(models):
            payload = {**payload_base, "model": model}
            if attempt:
                print(f"\n[{self.name}] Retrying tool request with {model}.")

            try:
                resp = requests.post(
                    config.API_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=config.REQUEST_TIMEOUT,
                    stream=False,
                )
                data = self._decode_json_response(resp)
            except Exception as exc:
                self._record_error(f"Request error: {exc}")
                data = None

            if data is None:
                continue

            if schemas:
                text, tool_calls = self._parse_response(data)
                if not text.strip() and not tool_calls and attempt < len(models) - 1:
                    self._record_error(f"Model {model} returned an empty tool response.")
                    continue

            self._last_error = None
            self._record_response_metadata(data, model, messages)
            return data

        return None



    def _stream(self, messages: list[dict]) -> str:
        """
        Streaming API call — prints tokens to stdout as they arrive.
        Returns the complete assembled text.
        Strips tool schemas for the final answer pass (no tool calls during streaming).
        """
        payload = {"model": config.MODEL, "messages": messages, "stream": True}
        full: list[str] = []

        try:
            resp = requests.post(
                config.API_URL,
                headers=self._headers,
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
                stream=True,
            )
            if resp.status_code >= 400 or "application/json" in resp.headers.get("Content-Type", ""):
                data = self._decode_json_response(resp)
                if data is None:
                    return f"[{self.name}] {self._last_error or 'API request failed.'}"
                text, _tool_calls = self._parse_response(data)
                return text or f"[{self.name}] Empty API response."

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    api_error = self._api_error_from_payload(chunk)
                    if api_error:
                        self._record_error(f"API error: {api_error}")
                        return f"[{self.name}] {self._last_error}"
                    # Puter SSE format: {"text": "..."} or {"reasoning": "..."}
                    text = chunk.get("text") or chunk.get("reasoning") or ""
                    # OpenAI SSE format: choices[0].delta.content
                    if not text and "choices" in chunk:
                        text = chunk["choices"][0].get("delta", {}).get("content", "") or ""
                    if text:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        full.append(text)
                except json.JSONDecodeError:
                    continue

        except Exception as exc:
            self._record_error(f"Stream error: {exc}")
            return f"[{self.name}] {self._last_error}"

        print()   # terminal newline
        return "".join(full)

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, data: dict) -> tuple[str, list[dict]]:
        """
        Returns (text_content, tool_calls).
        Handles both Puter and OpenAI response shapes, including nested 'result'.
        """
        if not data:
            return "", []

        # OpenAI format: {"choices": [{"message": {...}}]}
        if "choices" in data:
            msg        = data["choices"][0].get("message", {})
            text       = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            return text, tool_calls

        # Puter simple format: {"text": "..."}
        if "text" in data:
            return data["text"], []

        # Puter nested: {"result": {...}}
        if "result" in data:
            return self._parse_response(data["result"])

        return str(data), []

    # ── Agentic loop ──────────────────────────────────────────────────────────

    def run(self, task: str, verbose: bool = True) -> str:
        """
        Execute a task with the full tool-use loop.

        - Loops until the model emits a text-only response (no tool_calls).
        - Prints the final response to stdout if ``verbose=True``.
        - Returns the final text string (used by the orchestrator).
        """
        self._reset_turn_metrics()
        messages: list[dict] = [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": task},
        ]

        for iteration in range(config.MAX_TOOL_ITERATIONS):
            data = self._call(messages)
            if data is None:
                return f"[{self.name}] {self._last_error or 'API call failed.'}"

            text, tool_calls = self._parse_response(data)

            # ── No tool calls → final answer ──────────────────────────────────
            if not tool_calls:
                final = text or f"[{self.name}] Empty response from model."
                if verbose:
                    print(f"\n[{self.name}]: ", end="", flush=True)
                    print(final)
                return final

            # ── Tool calls → execute and loop ─────────────────────────────────
            if verbose:
                for tc in tool_calls:
                    fn   = tc.get("function", {})
                    args = fn.get("arguments", "")[:100]
                    print(f"  🔧 [{self.name}] {fn.get('name')}({args}{'...' if len(fn.get('arguments',''))>100 else ''})")

            # Append assistant message (may include reasoning text)
            messages.append({
                "role":       "assistant",
                "content":    text or "",
                "tool_calls": tool_calls,
            })

            # Execute each tool and collect results
            for tc in tool_calls:
                result = self.registry.dispatch(
                    tc["function"]["name"],
                    tc["function"].get("arguments", "{}"),
                )
                if verbose:
                    preview = result[:200] + "…" if len(result) > 200 else result
                    print(f"     ↳ {preview}")

                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc["id"],
                    "content":     result,
                })

        return f"[{self.name}] Reached max tool iterations ({config.MAX_TOOL_ITERATIONS})."
