"""Thin HTTP/WebSocket client for the Skipper platform's voice API.

The voice service is a companion *satellite*: it does audio capture/playback,
wake-word detection, and OpenAI Realtime streaming **locally**, and delegates
the brain / tools / database to the platform over the network. No platform
code, database connection, or MCP server runs on the voice device — everything
that needs them happens on the platform via these endpoints (served by the
platform's agent.py):

    GET  /api/config/picovoice    -> {"access_key": "..."}
    POST /api/voice/session       -> session config incl. an ephemeral OpenAI
                                     Realtime token, model, voice, tools,
                                     instructions, and a server-side session_id
    POST /api/voice/switch-app    -> a session.update payload for an app switch
    POST /api/voice/end           -> end the server-side session
    WS   /ws/voice/{session_id}   -> sideband relay. We send:
                                       {type:"tool_call", call_id, name, arguments}
                                       {type:"transcript", role, text}
                                     The platform executes the tool (MCP + DB)
                                     and streams back events we hand to the
                                     caller: tool_result / session_update /
                                     confirmation_required / end_session.

Only the stdlib + websocket-client are used here — deliberately no platform
imports, no psycopg, no MCP.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request

logger = logging.getLogger("skipperbot_voice")

DEFAULT_API_BASE = os.getenv("SKIPPER_API_BASE", "http://localhost:8000").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("SKIPPER_API_TIMEOUT", "15"))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(method: str, path: str, payload: dict | None, *, api_base: str) -> dict:
    url = f"{api_base}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def get_picovoice_config(*, api_base: str = DEFAULT_API_BASE) -> dict:
    """Wake-word config from the platform (Picovoice access key, if any)."""
    return _request("GET", "/api/config/picovoice", None, api_base=api_base)


def create_session(user_id: str, device_info: dict | None = None, *,
                   api_base: str = DEFAULT_API_BASE) -> dict | None:
    """Create a server-side voice session and get its config (incl. the
    ephemeral OpenAI Realtime token). Returns None on failure."""
    res = _request("POST", "/api/voice/session",
                   {"user_id": user_id, "device_info": device_info or {}},
                   api_base=api_base)
    if not res or res.get("error"):
        logger.error("voice/session failed: %s", (res or {}).get("error", "no response"))
        return None
    return res


def switch_app(session_id: str, app: str, *, api_base: str = DEFAULT_API_BASE) -> dict:
    """Ask the platform for the session.update payload to switch apps."""
    return _request("POST", "/api/voice/switch-app",
                    {"session_id": session_id, "app": app}, api_base=api_base)


def end_session(session_id: str, *, api_base: str = DEFAULT_API_BASE) -> None:
    """End the server-side session (best-effort)."""
    try:
        _request("POST", "/api/voice/end", {"session_id": session_id}, api_base=api_base)
    except Exception as exc:  # noqa: BLE001 — cleanup must never raise
        logger.debug("voice/end failed: %s", exc)


# ---------------------------------------------------------------------------
# Sideband WebSocket
# ---------------------------------------------------------------------------

def _ws_url(api_base: str, session_id: str) -> str:
    base = api_base.replace("https://", "wss://").replace("http://", "ws://")
    return f"{base}/ws/voice/{session_id}"


class Sideband:
    """Sideband WebSocket to the platform.

    Forwards tool calls and transcripts to the platform, and invokes
    ``on_event(event_dict)`` for every event the platform sends back
    (``tool_result`` / ``session_update`` / ``confirmation_required`` /
    ``end_session``). It's event-driven, not request/response: a tool call is
    fired and its result arrives later on the same socket — exactly the bridge
    pattern the platform expects from the mobile client.
    """

    def __init__(self, session_id: str, on_event, *, api_base: str = DEFAULT_API_BASE):
        self.session_id = session_id
        self.on_event = on_event
        self.api_base = api_base
        self._ws = None
        self._thread = None
        self._closed = False

    def start(self) -> None:
        try:
            import websocket  # websocket-client
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: websocket-client (pip install websocket-client)"
            ) from exc
        self._ws = websocket.WebSocketApp(
            _ws_url(self.api_base, self.session_id),
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever, name="skipper-sideband", daemon=True)
        self._thread.start()

    def _on_message(self, _ws, message) -> None:
        try:
            event = json.loads(message)
        except (ValueError, TypeError):
            logger.debug("sideband: non-JSON message ignored")
            return
        try:
            self.on_event(event)
        except Exception as exc:  # noqa: BLE001 — a bad event must not kill the socket
            logger.error("sideband on_event failed: %s", exc, exc_info=True)

    def _on_error(self, _ws, error) -> None:
        logger.error("sideband WS error: %s", error)

    def _on_close(self, _ws, *_args) -> None:
        if not self._closed:
            logger.warning("sideband WS closed unexpectedly for %s", self.session_id[:8])

    def _send(self, obj: dict) -> None:
        if self._ws is None or self._closed:
            return
        try:
            self._ws.send(json.dumps(obj))
        except Exception as exc:  # noqa: BLE001
            logger.error("sideband send failed: %s", exc)

    def send_tool_call(self, call_id: str, name: str, arguments) -> None:
        self._send({"type": "tool_call", "call_id": call_id,
                    "name": name, "arguments": arguments})

    def send_transcript(self, role: str, text: str) -> None:
        self._send({"type": "transcript", "role": role, "text": text})

    def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
