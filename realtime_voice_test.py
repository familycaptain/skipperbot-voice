from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basic_audio_test import (  # noqa: E402
    DEFAULT_DEVICE_NAME,
    DEFAULT_INPUT_NAME,
    DEFAULT_OUTPUT_NAME,
    describe_selected_devices,
    find_device,
    list_audio_devices,
    load_audio_deps,
    resolve_audio_settings,
    convert_audio,
)
import logging  # noqa: E402
from speaker_isolation import SpeakerIsolationConfig, SpeakerIsolationGate  # noqa: E402
import skipper_voice_client as skc  # noqa: E402

# Thin client: the brain/tools/DB live on the platform. No platform imports.
logger = logging.getLogger("skipperbot_voice")


REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
REALTIME_AUDIO_RATE = 24000
REALTIME_CHANNELS = 1
DEFAULT_FRAME_MS = int(os.getenv("VOICE_FRAME_MS", "20"))
DEFAULT_SPEAKER_ISOLATION = os.getenv("VOICE_SPEAKER_ISOLATION", "off").strip().lower()
DEFAULT_SPEAKER_SIMILARITY = float(os.getenv("VOICE_SPEAKER_SIMILARITY", "0.68"))
DEFAULT_SPEAKER_RMS = float(os.getenv("VOICE_SPEAKER_RMS", "0.015"))
DEFAULT_SPEAKER_SILENCE_MS = int(os.getenv("VOICE_SPEAKER_SILENCE_MS", "700"))
DEFAULT_SPEAKER_ENROLL_MIN_MS = int(os.getenv("VOICE_SPEAKER_ENROLL_MIN_MS", "900"))
DEFAULT_SPEAKER_VERIFY_MIN_MS = int(os.getenv("VOICE_SPEAKER_VERIFY_MIN_MS", "450"))
DEFAULT_SUPPRESS_MIC_DURING_PLAYBACK = os.getenv(
    "VOICE_SUPPRESS_MIC_DURING_PLAYBACK",
    "true",
).strip().lower() not in {"0", "false", "off", "no"}
DEFAULT_PLAYBACK_TAIL_MS = int(os.getenv("VOICE_PLAYBACK_TAIL_MS", "250"))
DEFAULT_PLAYBACK_BARGE_IN_GRACE_MS = int(os.getenv("VOICE_PLAYBACK_BARGE_IN_GRACE_MS", "700"))


class RealtimeAudioBridge:
    """Streams EMEET mic frames to Realtime and speaker frames back out."""

    def __init__(
        self,
        np: Any,
        sd: Any,
        *,
        input_device: int | None,
        output_device: int | None,
        input_sample_rate: int,
        input_channels: int,
        output_sample_rate: int,
        output_channels: int,
        frame_ms: int,
        speaker_gate: SpeakerIsolationGate | None = None,
    ) -> None:
        self.np = np
        self.sd = sd
        self.input_device = input_device
        self.output_device = output_device
        self.input_sample_rate = input_sample_rate
        self.input_channels = input_channels
        self.output_sample_rate = output_sample_rate
        self.output_channels = output_channels
        self.frame_ms = frame_ms
        self.speaker_gate = speaker_gate
        self.input_blocksize = max(1, int(input_sample_rate * frame_ms / 1000))
        self.output_blocksize = max(1, int(output_sample_rate * frame_ms / 1000))
        self.realtime_frame_bytes = max(1, int(REALTIME_AUDIO_RATE * frame_ms / 1000)) * REALTIME_CHANNELS * 2
        self.mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=1000)
        self._speaker_buffer = bytearray()
        self._speaker_lock = threading.Lock()
        self._last_output_audio_at = 0.0
        self._playback_started_at = 0.0
        self._playback_tail_seconds = max(0.0, DEFAULT_PLAYBACK_TAIL_MS / 1000.0)
        self._barge_in_grace_seconds = max(0.0, DEFAULT_PLAYBACK_BARGE_IN_GRACE_MS / 1000.0)
        self._suppress_mic_during_playback = DEFAULT_SUPPRESS_MIC_DURING_PLAYBACK
        self._input_stream = None
        self._output_stream = None

    def start(self) -> None:
        self._input_stream = self.sd.InputStream(
            samplerate=self.input_sample_rate,
            blocksize=self.input_blocksize,
            device=self.input_device,
            channels=self.input_channels,
            dtype="int16",
            callback=self._input_callback,
        )
        self._output_stream = self.sd.OutputStream(
            samplerate=self.output_sample_rate,
            blocksize=self.output_blocksize,
            device=self.output_device,
            channels=self.output_channels,
            dtype="int16",
            callback=self._output_callback,
        )
        self._output_stream.start()
        self._input_stream.start()

    def stop(self) -> None:
        if self.speaker_gate is not None:
            for chunk in self.speaker_gate.flush():
                self._enqueue_mic_bytes(chunk)
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def get_mic_chunk(self, timeout: float = 0.2) -> bytes | None:
        try:
            return self.mic_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def inject_preroll_pcm(self, pcm: Any, *, source_sample_rate: int) -> int:
        """Resample int16 mono PCM and pre-load mic_queue so the WS sender ships
        these bytes ahead of the live mic. Returns the number of samples enqueued
        at the realtime sample rate."""
        if pcm is None or source_sample_rate <= 0:
            return 0
        try:
            length = int(pcm.shape[0])
        except AttributeError:
            length = len(pcm)
        if length <= 0:
            return 0
        audio = pcm.reshape(-1, 1) if pcm.ndim == 1 else pcm
        converted = convert_audio(
            self.np,
            audio,
            input_sample_rate=source_sample_rate,
            output_sample_rate=REALTIME_AUDIO_RATE,
            output_channels=REALTIME_CHANNELS,
        )
        self._enqueue_mic_bytes(converted.tobytes())
        return int(converted.shape[0])

    def append_output_audio(self, pcm_24k_mono: bytes) -> None:
        audio = self.np.frombuffer(pcm_24k_mono, dtype=self.np.int16).reshape(-1, 1)
        converted = convert_audio(
            self.np,
            audio,
            input_sample_rate=REALTIME_AUDIO_RATE,
            output_sample_rate=self.output_sample_rate,
            output_channels=self.output_channels,
        )
        with self._speaker_lock:
            if not self._speaker_buffer:
                self._playback_started_at = time.monotonic()
            self._speaker_buffer.extend(converted.tobytes())
            self._last_output_audio_at = time.monotonic()

    def clear_output_audio(self) -> None:
        with self._speaker_lock:
            self._speaker_buffer.clear()

    def is_output_active(self) -> bool:
        with self._speaker_lock:
            has_buffered_audio = bool(self._speaker_buffer)
            last_output_audio_at = self._last_output_audio_at
        return has_buffered_audio or (
            last_output_audio_at > 0
            and time.monotonic() - last_output_audio_at <= self._playback_tail_seconds
        )

    def is_playback_barge_in_grace_active(self) -> bool:
        with self._speaker_lock:
            playback_started_at = self._playback_started_at
        return (
            playback_started_at > 0
            and time.monotonic() - playback_started_at <= self._barge_in_grace_seconds
        )

    def _input_callback(self, indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if status:
            logger.debug("HOME_VOICE: input status: %s", status)
        if self._suppress_mic_during_playback and self.is_playback_barge_in_grace_active():
            return
        realtime_audio = convert_audio(
            self.np,
            indata,
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=REALTIME_AUDIO_RATE,
            output_channels=REALTIME_CHANNELS,
        )
        chunks = [realtime_audio.tobytes()]
        if self.speaker_gate is not None:
            chunks = self.speaker_gate.process(realtime_audio.tobytes())
        for chunk in chunks:
            self._enqueue_mic_bytes(chunk)

    def _enqueue_mic_bytes(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        for offset in range(0, len(audio_bytes), self.realtime_frame_bytes):
            chunk = audio_bytes[offset:offset + self.realtime_frame_bytes]
            if not chunk:
                continue
            if len(chunk) < self.realtime_frame_bytes:
                chunk += b"\x00" * (self.realtime_frame_bytes - len(chunk))
            try:
                self.mic_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def _output_callback(self, outdata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if status:
            logger.debug("HOME_VOICE: output status: %s", status)

        bytes_needed = frames * self.output_channels * 2
        with self._speaker_lock:
            chunk = self._speaker_buffer[:bytes_needed]
            del self._speaker_buffer[:bytes_needed]
            has_more_audio = bool(self._speaker_buffer)

        played_audio = bool(chunk) and any(chunk)
        if len(chunk) < bytes_needed:
            chunk += b"\x00" * (bytes_needed - len(chunk))
        if has_more_audio or played_audio:
            self._last_output_audio_at = time.monotonic()

        outdata[:] = self.np.frombuffer(chunk, dtype=self.np.int16).reshape(
            frames,
            self.output_channels,
        )


class RealtimeHomeVoiceClient:
    def __init__(
        self,
        *,
        session_config: dict,
        audio_bridge: RealtimeAudioBridge,
        stop_event: threading.Event,
    ) -> None:
        self.session_config = session_config
        self.audio_bridge = audio_bridge
        self.stop_event = stop_event
        # Sideband WS to the platform (relays tool calls + transcripts).
        # Set by the run flow after the session is created.
        self.sideband = None
        self.ws = None
        self.sender_thread: threading.Thread | None = None
        self.pending_tool_names: dict[str, str] = {}
        self.started_at = time.monotonic()
        self.last_activity_at = self.started_at
        self.first_user_transcript_at = 0.0
        self._activity_lock = threading.Lock()

    def mark_activity(self, *, user_transcript: bool = False) -> None:
        now = time.monotonic()
        with self._activity_lock:
            self.last_activity_at = now
            if user_transcript and not self.first_user_transcript_at:
                self.first_user_transcript_at = now

    def idle_seconds(self) -> float:
        with self._activity_lock:
            last_activity_at = self.last_activity_at
        return time.monotonic() - last_activity_at

    def waiting_for_first_user_transcript_seconds(self) -> float:
        with self._activity_lock:
            if self.first_user_transcript_at:
                return 0.0
            started_at = self.started_at
        return time.monotonic() - started_at

    def run(self) -> None:
        websocket = load_websocket_dep()
        model = self.session_config["model"]
        token = self.session_config["ephemeral_token"]
        url = f"{REALTIME_WS_URL}?model={quote_plus(model)}"

        self.ws = websocket.WebSocketApp(
            url,
            header=[
                f"Authorization: Bearer {token}",
            ],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever(ping_interval=20, ping_timeout=10)

    def close(self) -> None:
        self.stop_event.set()
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass

    def send_event(self, event: dict) -> None:
        if self.ws is None:
            return
        self.ws.send(json.dumps(event))

    def _on_open(self, ws) -> None:
        print("Realtime WebSocket connected.")
        self._send_session_update(
            self.session_config["base_instructions"],
            self.session_config["base_tools"],
        )
        self.sender_thread = threading.Thread(target=self._send_mic_loop, daemon=True)
        self.sender_thread.start()

    def _on_message(self, ws, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("HOME_VOICE: non-JSON realtime message: %r", message[:200])
            return

        event_type = event.get("type", "")

        if event_type in {"session.updated", "session.created"}:
            tools = event.get("session", {}).get("tools") or []
            print(f"Realtime {event_type}; tools={len(tools)}")
        elif event_type in {"input_audio_buffer.speech_started"}:
            if self.audio_bridge.is_playback_barge_in_grace_active():
                logger.debug("HOME_VOICE: ignored speech_started during playback grace window")
            else:
                self.mark_activity()
                print("Listening...")
                self.audio_bridge.clear_output_audio()
        elif event_type in {"input_audio_buffer.speech_stopped"}:
            self.mark_activity()
            print("Thinking...")
        elif event_type in {
            "response.audio.delta",
            "response.output_audio.delta",
            "response.audio.delta.done",
        }:
            delta = event.get("delta")
            if delta:
                self.mark_activity()
                self.audio_bridge.append_output_audio(base64.b64decode(delta))
        elif event_type in {
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        }:
            transcript = event.get("transcript", "")
            if transcript:
                print(f"Skipper: {transcript}")
                self._record_transcript("assistant", transcript)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                self.mark_activity(user_transcript=True)
                print(f"You: {transcript}")
                self._record_transcript("user", transcript)
        elif event_type == "response.function_call_arguments.done":
            self.mark_activity()
            self._handle_tool_call(event)
        elif event_type == "response.done":
            self.mark_activity()
            print("Ready.")
        elif event_type == "error":
            print(f"Realtime error: {event.get('error', event)}")
            logger.error("HOME_VOICE: realtime error: %s", event)
        else:
            logger.debug("HOME_VOICE: realtime event: %s", event_type)

    def _record_transcript(self, role: str, transcript: str) -> None:
        # The platform owns the DB; relay the transcript over the sideband WS
        # and let it persist (voice_chatlog) server-side.
        if self.sideband is not None:
            self.sideband.send_transcript(role, transcript)

    def _on_error(self, ws, error) -> None:
        if not self.stop_event.is_set():
            print(f"Realtime WebSocket error: {error}")
            logger.error("HOME_VOICE: realtime websocket error: %s", error)

    def _on_close(self, ws, status_code, message) -> None:
        self.stop_event.set()
        print(f"Realtime WebSocket closed ({status_code}): {message}")

    def _send_mic_loop(self) -> None:
        print("Streaming EMEET microphone. Press Ctrl+C to stop.")
        while not self.stop_event.is_set():
            chunk = self.audio_bridge.get_mic_chunk()
            if not chunk:
                continue
            try:
                self.send_event({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                })
            except Exception as exc:
                if not self.stop_event.is_set():
                    logger.error("HOME_VOICE: failed sending mic chunk: %s", exc)
                self.stop_event.set()
                break

    def _send_session_update(self, instructions: str, tools: list[dict]) -> None:
        self.send_event({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": instructions,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_AUDIO_RATE},
                        "transcription": {
                            "model": os.getenv("VOICE_REALTIME_TRANSCRIPTION_MODEL", "whisper-1"),
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": float(os.getenv("VOICE_VAD_THRESHOLD", "0.5")),
                            "prefix_padding_ms": int(os.getenv("VOICE_VAD_PREFIX_PADDING_MS", "300")),
                            "silence_duration_ms": int(os.getenv("VOICE_VAD_SILENCE_MS", "500")),
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_AUDIO_RATE},
                        "voice": self.session_config.get("voice", "ash"),
                    },
                },
                "tools": tools,
            },
        })

    def _handle_tool_call(self, event: dict) -> None:
        call_id = event.get("call_id", "")
        tool_name = event.get("name", "")
        raw_args = event.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            arguments = {}

        print(f"Tool call: {tool_name}({arguments})")
        self.pending_tool_names[call_id] = tool_name
        # The platform executes the tool (MCP + DB) and streams the result back
        # over the sideband WS; _dispatch_platform_event handles the reply.
        if self.sideband is not None:
            self.sideband.send_tool_call(call_id, tool_name, arguments)
        else:
            logger.error("HOME_VOICE: no sideband connection; dropping tool call %s", tool_name)

    def _dispatch_platform_event(self, event: dict) -> None:
        """Handle one event the platform streams back over the sideband WS:
        tool_result / session_update / confirmation_required / end_session."""
        event_type = event.get("type")
        if event_type == "session_update":
            app = event.get("app") or "default"
            print(f"Switching voice app: {app}")
            self._send_session_update(
                event.get("instructions", ""),
                event.get("tools", []),
            )
        elif event_type == "tool_result":
            output       = event.get("output", "")
            event_call   = event.get("call_id", "")
            tool_name    = self.pending_tool_names.pop(event_call, "")
            # Echo the tool output to the console so we can see what the
            # model is actually getting back (recall hits, errors, etc).
            # Truncate long outputs so the console stays readable; the
            # full string still goes back to OpenAI below.
            preview = (output or "(empty)").rstrip()
            if len(preview) > 800:
                preview = preview[:800] + f"... ({len(output) - 800} more chars)"
            label = tool_name or event_call or "tool"
            print(f"Tool result [{label}]:\n  " + preview.replace("\n", "\n  "))

            self.send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": event_call,
                    "output": output,
                },
            })
            self.send_event({"type": "response.create"})
        elif event_type == "confirmation_required":
            logger.info("HOME_VOICE: confirmation_required for %s", event.get("action"))
        elif event_type == "end_session":
            print("Ending session.")
            self.stop_event.set()


def load_websocket_dep():
    try:
        import websocket
    except ModuleNotFoundError as exc:
        print(
            "Missing dependency: websocket-client\n"
            "Install Phase 1 audio test dependencies with:\n"
            "  pip install -r home_voice/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return websocket


def build_speaker_gate(np: Any, *, sample_rate: int, frame_ms: int, args: argparse.Namespace) -> SpeakerIsolationGate | None:
    mode = getattr(args, "speaker_isolation", DEFAULT_SPEAKER_ISOLATION)
    if mode in {"", "off", "false", "0", "disabled"}:
        return None
    config = SpeakerIsolationConfig(
        mode=mode,
        similarity_threshold=getattr(args, "speaker_similarity", DEFAULT_SPEAKER_SIMILARITY),
        speech_rms_threshold=getattr(args, "speaker_rms", DEFAULT_SPEAKER_RMS),
        silence_ms=getattr(args, "speaker_silence_ms", DEFAULT_SPEAKER_SILENCE_MS),
        enroll_min_ms=getattr(args, "speaker_enroll_min_ms", DEFAULT_SPEAKER_ENROLL_MIN_MS),
        verify_min_ms=getattr(args, "speaker_verify_min_ms", DEFAULT_SPEAKER_VERIFY_MIN_MS),
    )
    gate = SpeakerIsolationGate(np, sample_rate=sample_rate, frame_ms=frame_ms, config=config)
    return gate if gate.active else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime EMEET voice prototype using the shared Android voice config."
    )
    parser.add_argument("--user-id", default=os.getenv("VOICE_USER_ID", "user1"))
    parser.add_argument("--device-id", default=os.getenv("VOICE_DEVICE_ID", "windows-server-local-emeet"))
    parser.add_argument("--room", default=os.getenv("VOICE_ROOM", "office"))
    parser.add_argument("--friendly-name", default=os.getenv("VOICE_FRIENDLY_NAME", "Office Speaker"))
    parser.add_argument("--device-name", default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="Optional auto-stop duration.")
    parser.add_argument(
        "--speaker-isolation",
        choices=("off", "optional", "required"),
        default=DEFAULT_SPEAKER_ISOLATION,
        help=(
            "Session speaker lock. 'optional' uses resemblyzer if installed and "
            "falls back to normal audio; 'required' fails if unavailable."
        ),
    )
    parser.add_argument("--speaker-similarity", type=float, default=DEFAULT_SPEAKER_SIMILARITY)
    parser.add_argument("--speaker-rms", type=float, default=DEFAULT_SPEAKER_RMS)
    parser.add_argument("--speaker-silence-ms", type=int, default=DEFAULT_SPEAKER_SILENCE_MS)
    parser.add_argument("--speaker-enroll-min-ms", type=int, default=DEFAULT_SPEAKER_ENROLL_MIN_MS)
    parser.add_argument("--speaker-verify-min-ms", type=int, default=DEFAULT_SPEAKER_VERIFY_MIN_MS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    np, sd = load_audio_deps()

    device_info = {
        "platform": "home_voice",
        "device_type": "home_voice",
        "device_id": args.device_id,
        "room": args.room,
        "friendly_name": args.friendly_name,
        "audio_device_name": args.device_name,
    }
    session_config = skc.create_session(args.user_id, device_info)
    if not session_config:
        raise RuntimeError("Could not create a voice session via the platform API.")

    print("Home realtime voice session:")
    print(f"  user_id:         {args.user_id}")
    print(f"  room:            {args.room}")
    print(f"  model:           {session_config.get('model')}")
    print(f"  active_app:      {session_config.get('active_app')}")
    print(f"  active_category: {session_config.get('active_category')}")
    print(f"  tools:           {len(session_config.get('base_tools', []))}")

    input_name = args.input_name or DEFAULT_INPUT_NAME or args.device_name
    output_name = args.output_name or DEFAULT_OUTPUT_NAME or args.device_name
    list_audio_devices(sd)
    input_device = find_device(sd, input_name, "input")
    output_device = find_device(sd, output_name, "output")
    describe_selected_devices(sd, input_device, output_device)

    input_sample_rate, input_channels = resolve_audio_settings(
        sd,
        device_index=input_device,
        kind="input",
        preferred_rate=16000,
        preferred_channels=1,
        use_device_default_rate=False,
    )
    output_sample_rate, output_channels = resolve_audio_settings(
        sd,
        device_index=output_device,
        kind="output",
        preferred_rate=REALTIME_AUDIO_RATE,
        preferred_channels=1,
        use_device_default_rate=False,
    )
    print(f"  input format:    {input_sample_rate} Hz, {input_channels} channel(s)")
    print(f"  output format:   {output_sample_rate} Hz, {output_channels} channel(s)")
    speaker_gate = build_speaker_gate(
        np,
        sample_rate=REALTIME_AUDIO_RATE,
        frame_ms=args.frame_ms,
        args=args,
    )
    print(f"  speaker lock:    {'on' if speaker_gate else 'off'}")

    stop_event = threading.Event()
    audio_bridge = RealtimeAudioBridge(
        np,
        sd,
        input_device=input_device,
        output_device=output_device,
        input_sample_rate=input_sample_rate,
        input_channels=input_channels,
        output_sample_rate=output_sample_rate,
        output_channels=output_channels,
        frame_ms=args.frame_ms,
        speaker_gate=speaker_gate,
    )
    client = RealtimeHomeVoiceClient(
        session_config=session_config,
        audio_bridge=audio_bridge,
        stop_event=stop_event,
    )
    client.sideband = skc.Sideband(
        session_config["session_id"], client._dispatch_platform_event)
    client.sideband.start()

    def _stop(signum=None, frame=None) -> None:
        stop_event.set()
        client.close()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    ws_thread = threading.Thread(target=client.run, daemon=True)
    try:
        audio_bridge.start()
        ws_thread.start()

        started = time.monotonic()
        while not stop_event.is_set():
            if args.duration_seconds > 0 and time.monotonic() - started >= args.duration_seconds:
                stop_event.set()
                client.close()
                break
            time.sleep(0.1)
    finally:
        client.close()
        if client.sideband is not None:
            client.sideband.close()
        audio_bridge.stop()
        skc.end_session(session_config["session_id"])

    print("Realtime home voice test stopped.")
    return 0


ASYNC_LOOP: asyncio.AbstractEventLoop
_ASYNC_THREAD: threading.Thread | None = None


def initialize_async_loop() -> None:
    global ASYNC_LOOP, _ASYNC_THREAD
    ASYNC_LOOP = asyncio.new_event_loop()
    _ASYNC_THREAD = threading.Thread(target=ASYNC_LOOP.run_forever, daemon=True)
    _ASYNC_THREAD.start()


def shutdown_async_loop() -> None:
    if ASYNC_LOOP.is_running():
        ASYNC_LOOP.call_soon_threadsafe(ASYNC_LOOP.stop)
    if _ASYNC_THREAD:
        _ASYNC_THREAD.join(timeout=2)


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, ASYNC_LOOP).result()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)
    except Exception as exc:
        logger.error("HOME_VOICE: realtime voice test failed: %s", exc, exc_info=True)
        print(f"\nRealtime voice test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
