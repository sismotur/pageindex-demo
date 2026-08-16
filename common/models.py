#!/usr/bin/env python3
"""
common/models.py — Canonical LLM model identifiers for the project.

Single source of truth for the models used by the assistant reference
implementation (run_eval.py / chat_demo.py) and named in the docs so the
Android/iOS ports have one table to mirror.

Model strings are litellm-style `openai/<id>` routed to the local oMLX
server (http://127.0.0.1:8000/v1).  Ollama remains a supported
alternative — use its tags instead (`openai/gemma4:26b`, …) with
OPENAI_API_BASE=http://localhost:11434/v1.
"""

from __future__ import annotations

# oMLX model ids (see ~/.omlx/models)
MODEL_E2B = "openai/gemma-4-E2B-it-MLX-8bit"    # mobile deployment target
MODEL_E4B = "openai/gemma-4-E4B-it-MLX-4bit"    # mid-size fallback
MODEL_26B = "openai/gemma-4-26B-A4B-it-MLX-4bit"  # server / quality ceiling

MODELS: dict[str, dict] = {
    MODEL_E2B: {
        "short": "gemma-4-E2B",
        "size_gb": 7.2,
        "context": 128_000,
        "role": "on-device (Android/iOS, fully offline)",
    },
    MODEL_E4B: {
        "short": "gemma-4-E4B",
        "size_gb": 9.6,
        "context": 128_000,
        "role": "fallback / comparison",
    },
    MODEL_26B: {
        "short": "gemma-4-26B",
        "size_gb": 18.0,
        "context": 256_000,
        "role": "server reference / eval ceiling",
    },
}

# Defaults used by the entry points.  Eval targets the mobile model (that
# is the deployment constraint); interactive chat defaults to the server
# model for answer quality.
DEFAULT_EVAL_MODEL = MODEL_E2B
DEFAULT_CHAT_MODEL = MODEL_26B
