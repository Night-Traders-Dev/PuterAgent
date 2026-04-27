# PuterAgent

PuterAgent is a local multi-agent coding assistant built around a Puter-compatible orchestrator and expert agents. It supports both a terminal CLI mode and a browser-based web UI.

## Features

- Terminal CLI interaction via `--cli`
- Browser UI via `--web`
- Model selection support for multiple vendors
- Expert routing across Code, File, Shell, and Debug agents
- Web chat history and clear session support
- Text file uploads in the browser UI
- Active model display with provider avatar switching
- Per-turn and session token metrics

## Requirements

- Python 3.11+ recommended
- `requests` library

Install dependencies:

```bash
pip install requests
```

## Setup

1. Place your API token in the `secret` file at the repository root.
2. Optionally add a custom orchestrator prompt in `SysPrompt`.

The project expects:

- `secret` — Puter auth token
- `SysPrompt` — optional custom orchestrator prompt

## Running

### Terminal CLI

```bash
python src/main.py --cli
```

### Browser UI

```bash
python src/main.py --web
```

The web UI will launch locally and serve `src/ui.html`.

## Model Selection

### CLI mode

Choose a model with `--model`:

```bash
python src/main.py --cli --model anthropic/claude-opus-4-7
```

### Web mode

Use the model dropdown in the sidebar.

### Supported models

- `deepseek/deepseek-v4-pro`
- `deepseek/deepseek-chat-v3.1`
- `deepseek/deepseek-chat`
- `deepseek/deepseek-r1-0528`
- `anthropic/claude-opus-4-7`
- `anthropic/claude-opus-4-6`
- `openai/gpt-5.5-pro`
- `moonshotai/kimi-k2.6`
- `x-ai/grok-4.20-multi-agent`
- `minimax/minimax-m2.7`
- `nvidia/nemotron-3-super-120b-a12b`
- `google/gemini-3.1-pro-preview`

## Commands

In CLI mode, use:

- `exit` or `quit` to stop
- `/clear` to reset the conversation

In web mode, use the `Clear conversation` button. Uploaded files are attached as bounded text context for the current turn.

## Project Layout

- `src/main.py` — application entrypoint and CLI/web launcher
- `src/config.py` — configuration, available models, and token loading
- `src/base_agent.py` — core API integration and tool loop
- `src/orchestrator.py` — orchestrator that routes tasks to expert agents
- `src/ui.html` — browser user interface

## Notes

- Keep the `secret` file private.
- Make sure the chosen model is supported by your configured backend.
