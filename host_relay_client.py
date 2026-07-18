"""Host-relay voice client (thin-satellite mode).

In this mode the satellite does NOT talk to OpenAI. It streams 2-way PCM to the
platform's audio relay (`/ws/voice/audio/{session_id}`), which holds the OpenAI
Realtime session server-side, runs the tools, and does speaker identification.
The satellite is just wake word + AEC + audio I/O.

Protocol (this client's side; see app_platform/voice/relay.py on the host):
  send:    binary frame = mic PCM16 mono @ REALTIME_AUDIO_RATE
           text  frame  = JSON control: {"type": "end"}
  receive: binary frame = output PCM16 to play
           text  frame  = JSON: {"type":"transcript"|"status"|"session_ended"|"error", ...}
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from typing import Any

from realtime_voice_test import REALTIME_AUDIO_RATE, load_websocket_dep
import skipper_voice_client as skc


class RelayClient:
    """Streams the mic to the host audio relay and plays what comes back."""

    def __init__(self, session_config: dict, audio_bridge: Any,
                 stop_event: threading.Event, *, api_base: str):
        self.session_config = session_config
        self.audio_bridge = audio_bridge
        self.stop_event = stop_event
        self.api_base = api_base.rstrip("/")
        self.session_id = session_config["session_id"]
        self.ws = None
        self._mic_thread: threading.Thread | None = None
        # Interface mirrored from the direct client so the wake-service timeout
        # loop can treat both the same way.
        self.first_user_transcript_at: float | None = None
        self._last_activity = time.monotonic()
        # Console ordering: the brain answers immediately (fail-open) while the user's
        # transcript is produced by a separate model a beat later, so "Skipper:" can
        # arrive before "You:". We hold response lines (Skipper: / tool:) until the
        # matching "You:" prints, so the log reads in the order things actually happened.
        self._pending_user = False
        self._buffered_lines: list[str] = []
        self._buf_lock = threading.Lock()
        self._buf_timer: "threading.Timer | None" = None

    # --- timeout-loop interface (matches the direct client) ---
    def mark_activity(self) -> None:
        self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def waiting_for_first_user_transcript_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    # --- console ordering: hold response lines until the You: line prints ---
    def _flush_buffered(self) -> None:
        with self._buf_lock:
            lines = self._buffered_lines
            self._buffered_lines = []
            self._pending_user = False
            if self._buf_timer is not None:
                self._buf_timer.cancel()
                self._buf_timer = None
        for line in lines:
            print(line)

    def _emit_response_line(self, line: str) -> None:
        """Print a response line (Skipper's speech / a tool call). If a user turn just
        ended but its `You:` hasn't printed yet, hold it so it prints AFTER `You:`; a
        timer flushes it if the transcript never arrives (e.g. a dropped/empty turn)."""
        with self._buf_lock:
            if self._pending_user:
                self._buffered_lines.append(line)
                if self._buf_timer is not None:
                    self._buf_timer.cancel()
                self._buf_timer = threading.Timer(1.5, self._flush_buffered)
                self._buf_timer.daemon = True
                self._buf_timer.start()
                return
        print(line)

    # --- websocket lifecycle ---
    def _url(self) -> str:
        base = self.api_base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/ws/voice/audio/{urllib.parse.quote(self.session_id)}"  # token rides the Authorization header, not the URL

    def run(self) -> None:
        websocket = load_websocket_dep()
        self.ws = websocket.WebSocketApp(
            self._url(),
            header=skc.ws_auth_headers(),
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever()

    def _on_open(self, ws) -> None:
        print("Connected to host audio relay. Streaming microphone.")
        self.mark_activity()
        self._mic_thread = threading.Thread(target=self._send_mic_loop, daemon=True)
        self._mic_thread.start()

    def _send_mic_loop(self) -> None:
        import websocket as _ws  # for the BINARY opcode
        while not self.stop_event.is_set():
            chunk = self.audio_bridge.get_mic_chunk()
            if not chunk:
                continue
            try:
                self.ws.send(chunk, opcode=_ws.ABNF.OPCODE_BINARY)
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"Relay: failed sending mic chunk: {exc}")
                self.stop_event.set()
                break

    def _on_message(self, ws, message) -> None:
        # Binary = output PCM to play; text = JSON control/transcript.
        if isinstance(message, (bytes, bytearray)):
            self.mark_activity()
            self.audio_bridge.append_output_audio(bytes(message))
            return
        try:
            event = json.loads(message)
        except (ValueError, TypeError):
            return
        etype = event.get("type")
        if etype == "transcript":
            role = event.get("role", "user")
            text = (event.get("text") or "").strip()
            if role == "user":
                if self.first_user_transcript_at is None:
                    self.first_user_transcript_at = time.monotonic()
                # Print You: first, then release any response lines held for this turn.
                with self._buf_lock:
                    held = self._buffered_lines
                    self._buffered_lines = []
                    self._pending_user = False
                    if self._buf_timer is not None:
                        self._buf_timer.cancel()
                        self._buf_timer = None
                print(f"You: {text}")
                for line in held:
                    print(line)
            else:
                self._emit_response_line(f"Skipper: {text}")
            self.mark_activity()
        elif etype == "status":
            status = event.get("status")
            if status == "speech_started":
                print("Listening...")
            elif status == "speech_stopped":
                print("Thinking...")
                with self._buf_lock:
                    self._pending_user = True   # a You: is expected for the turn just ended
            self.mark_activity()
        elif etype == "tool_call":
            name = event.get("name", "?")
            args = event.get("args")
            try:
                args_str = json.dumps(args, default=str) if args else "{}"
            except Exception:
                args_str = str(args)
            self._emit_response_line(f"  -> tool: {name} {args_str}")
            self.mark_activity()
        elif etype == "host_info":
            print(f"Host: {event.get('text', '')}")
        elif etype == "session_ended":
            print("Session ended by host.")
            self.stop_event.set()
            try:
                ws.close()
            except Exception:
                pass
        elif etype == "error":
            print(f"Relay error: {event.get('error')}")

    def _on_error(self, ws, error) -> None:
        if not self.stop_event.is_set():
            print(f"Relay websocket error: {error}")

    def _on_close(self, ws, status_code, msg) -> None:
        self.stop_event.set()

    def request_end(self) -> None:
        """Tell the host to end (e.g. local 'goodbye' / wake-timeout)."""
        try:
            if self.ws is not None:
                self.ws.send(json.dumps({"type": "end"}))
        except Exception:
            pass
        self.stop_event.set()

    def close(self) -> None:
        """Stop streaming and close the relay socket (cleanup)."""
        self.stop_event.set()
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass
