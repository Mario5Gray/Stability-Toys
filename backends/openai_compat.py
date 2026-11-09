"""Shared wire helpers for OpenAI-compatible chat/completions endpoints.

The text chat client and the multimodal analysis client speak the same
protocol. URL construction, bearer auth, and auth-failure explanation live
here so the two clients cannot drift.
"""
from __future__ import annotations

import os
from typing import Dict

import httpx


class AuthError(RuntimeError):
    """An endpoint rejected (or never received) the configured bearer token."""


def chat_completions_url(endpoint: str) -> str:
    """Join an OpenAI-compatible base URL with the chat/completions path."""
    return f"{endpoint.rstrip('/')}/chat/completions"


def bearer_headers(api_key_env: str) -> Dict[str, str]:
    """Build request headers, adding Authorization only when the token is set.

    The variable is read at request time so a rotated token is picked up
    without rebuilding the client.
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get(api_key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def ensure_success(
    resp: httpx.Response,
    *,
    endpoint: str,
    api_key_env: str,
) -> None:
    """Raise on a non-2xx response, naming the credential on 401/403.

    The generic httpx message says only that the request failed. An operator
    needs the two facts that live in configuration: which endpoint rejected
    the call, and which environment variable supplies its token.
    """
    status = getattr(resp, "status_code", None)
    if status in (401, 403):
        if os.environ.get(api_key_env, ""):
            state = f"the bearer token in {api_key_env} was rejected"
        else:
            state = f"the bearer token environment variable {api_key_env} is unset"
        raise AuthError(
            f"{endpoint} returned HTTP {status}: {state}. "
            f"Set {api_key_env} in the server environment, or remove "
            "api_key_env from this connection in modes.yml if the endpoint "
            "does not require authentication."
        )
    resp.raise_for_status()
