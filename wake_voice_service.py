from __future__ import annotations

import argparse
import collections
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


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
    play_test_tone,
    resolve_audio_settings,
)
import logging  # noqa: E402
from realtime_voice_test import (  # noqa: E402
    DEFAULT_SPEAKER_ENROLL_MIN_MS,
    DEFAULT_SPEAKER_ISOLATION,
    DEFAULT_SPEAKER_RMS,
    DEFAULT_SPEAKER_SILENCE_MS,
    DEFAULT_SPEAKER_SIMILARITY,
    DEFAULT_SPEAKER_VERIFY_MIN_MS,
    REALTIME_AUDIO_RATE,
    RealtimeAudioBridge,
    RealtimeHomeVoiceClient,
    build_speaker_gate,
    initialize_async_loop,
    run_async,
    shutdown_async_loop,
)
import skipper_voice_client as skc  # noqa: E402

# Thin client: brain/tools/DB live on the platform. No platform imports.
logger = logging.getLogger("skipperbot_voice")


DEFAULT_WAKE_KEYWORD_RAW = os.getenv("VOICE_WAKE_KEYWORD_PATH", "").strip()
DEFAULT_WAKE_KEYWORD_PATH = Path(DEFAULT_WAKE_KEYWORD_RAW) if DEFAULT_WAKE_KEYWORD_RAW else None
DEFAULT_BUILTIN_KEYWORD = os.getenv("VOICE_WAKE_BUILTIN_KEYWORD", "computer")
DEFAULT_SENSITIVITY = float(os.getenv("VOICE_WAKE_SENSITIVITY", "0.70"))
DEFAULT_WAKE_BACKEND = os.getenv("VOICE_WAKE_BACKEND", "openwakeword").strip().lower()
DEFAULT_OPENWAKEWORD_THRESHOLD = float(os.getenv("VOICE_OPENWAKEWORD_THRESHOLD", "0.90"))
DEFAULT_OPENWAKEWORD_LABEL = os.getenv("VOICE_OPENWAKEWORD_LABEL", "").strip()
DEFAULT_OPENWAKEWORD_FRAME_MS = int(os.getenv("VOICE_OPENWAKEWORD_FRAME_MS", "80"))
DEFAULT_OPENWAKEWORD_INFERENCE_FRAMEWORK = os.getenv(
    "VOICE_OPENWAKEWORD_INFERENCE_FRAMEWORK",
    "onnx",
)
DEFAULT_COOLDOWN_SECONDS = float(os.getenv("VOICE_WAKE_COOLDOWN_SECONDS", "1.0"))
DEFAULT_FRAME_MS = int(os.getenv("VOICE_FRAME_MS", "20"))
DEFAULT_INITIAL_SPEECH_TIMEOUT_SECONDS = float(os.getenv("VOICE_INITIAL_SPEECH_TIMEOUT_SECONDS", "20"))
DEFAULT_IDLE_TIMEOUT_SECONDS = float(os.getenv("VOICE_IDLE_TIMEOUT_SECONDS", "45"))
DEFAULT_MAX_SESSION_SECONDS = float(os.getenv("VOICE_MAX_SESSION_SECONDS", "300"))
DEFAULT_PREROLL_SECONDS = float(os.getenv("VOICE_PREROLL_SECONDS", "3.0"))
WAKE_WORD_MODEL_DIR = Path(__file__).resolve().parent / "wake_words"


def default_openwakeword_model_paths() -> list[Path]:
    raw_paths = [
        Path(p.strip())
        for p in os.getenv("VOICE_OPENWAKEWORD_MODEL_PATHS", "").replace(";", ",").split(",")
        if p.strip()
    ]
    if raw_paths:
        return raw_paths

    stable_names = [
        WAKE_WORD_MODEL_DIR / "hey-skipper.onnx",
        WAKE_WORD_MODEL_DIR / "hey_skipper.onnx",
    ]
    for path in stable_names:
        if path.exists():
            return [path]

    timestamped = sorted(
        WAKE_WORD_MODEL_DIR.glob("Hey_Skipper*.onnx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return timestamped[:1]


DEFAULT_OPENWAKEWORD_MODEL_PATHS = default_openwakeword_model_paths()


class OpenWakeWordWakeListener:
    """Local wake-word detector using openWakeWord and sounddevice."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        np: Any,
        sd: Any,
        *,
        input_device: int | None,
        model_paths: list[Path],
        threshold: float,
        label: str,
        frame_ms: int,
        download_models: bool,
        inference_framework: str,
        preroll_seconds: float = 0.0,
    ) -> None:
        self.np = np
        self.sd = sd
        self.input_device = input_device
        self.model_paths = model_paths
        self.threshold = threshold
        self.label = label.lower().strip()
        self.frame_ms = frame_ms
        self.download_models = download_models
        self.inference_framework = inference_framework
        self.blocksize = max(1280, int(self.SAMPLE_RATE * frame_ms / 1000))
        self.model = None
        self.stream = None
        self.audio_queue: queue.Queue[Any] = queue.Queue(maxsize=50)
        self.preroll_seconds = max(0.0, preroll_seconds)
        self.preroll_sample_rate = self.SAMPLE_RATE
        self._preroll_frames: collections.deque[Any] | None = None
        if self.preroll_seconds > 0 and self.blocksize > 0:
            total_samples = int(self.preroll_seconds * self.SAMPLE_RATE)
            maxlen = max(1, (total_samples + self.blocksize - 1) // self.blocksize + 2)
            self._preroll_frames = collections.deque(maxlen=maxlen)

    def __enter__(self) -> "OpenWakeWordWakeListener":
        try:
            from openwakeword.model import Model
            from openwakeword import utils as oww_utils
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "openwakeword is not installed. Run: pip install -r requirements.txt "
                "&& pip install --no-deps 'openwakeword>=0.6.0'"
            ) from exc

        if self.download_models:
            print("Downloading openWakeWord pretrained/preprocessing models...")
            oww_utils.download_models()

        kwargs: dict[str, Any] = {}
        if self.model_paths:
            kwargs["wakeword_models"] = [str(path) for path in self.model_paths]
        if self.inference_framework and self.inference_framework != "auto":
            kwargs["inference_framework"] = self.inference_framework

        try:
            self.model = Model(**kwargs)
        except TypeError:
            kwargs.pop("inference_framework", None)
            self.model = Model(**kwargs)

        try:
            self.sd.check_input_settings(
                device=self.input_device,
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenWakeWord wake listener needs 16 kHz mono int16 input. "
                f"{exc}"
            ) from exc

        self.stream = self.sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.blocksize,
            device=self.input_device,
            channels=1,
            dtype="int16",
            callback=self._input_callback,
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def wait_for_detection(self, stop_event: threading.Event) -> bool:
        if self.model is None:
            raise RuntimeError("OpenWakeWord listener is not started.")

        while not stop_event.is_set():
            try:
                frame = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            predictions = self.model.predict(frame)
            name, score = self._best_prediction(predictions)
            if score >= self.threshold:
                print(f"Wake score: {name}={score:.3f}")
                return True
        return False

    def _best_prediction(self, predictions: dict) -> tuple[str, float]:
        best_name = ""
        best_score = 0.0
        for name, raw_score in predictions.items():
            score = float(raw_score)
            normalized_name = str(name).lower()
            if self.label and self.label not in normalized_name:
                continue
            if score > best_score:
                best_name = str(name)
                best_score = score
        return best_name, best_score

    def _input_callback(self, indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if status:
            logger.debug("HOME_VOICE_WAKE_OPENWAKEWORD: input status: %s", status)
        pcm = indata.reshape(-1).astype(self.np.int16, copy=False)
        if self._preroll_frames is not None:
            self._preroll_frames.append(pcm.copy())
        try:
            self.audio_queue.put_nowait(pcm.copy())
        except queue.Full:
            pass

    def snapshot_preroll_pcm(self) -> Any | None:
        if not self._preroll_frames:
            return None
        frames = list(self._preroll_frames)
        if not frames:
            return None
        return self.np.concatenate(frames)


class PorcupineWakeListener:
    """Local wake-word detector using Picovoice Porcupine and sounddevice."""

    def __init__(
        self,
        np: Any,
        sd: Any,
        *,
        access_key: str,
        input_device: int | None,
        keyword_path: Path | None,
        builtin_keyword: str,
        sensitivity: float,
        preroll_seconds: float = 0.0,
    ) -> None:
        self.np = np
        self.sd = sd
        self.access_key = access_key
        self.input_device = input_device
        self.keyword_path = keyword_path
        self.builtin_keyword = builtin_keyword
        self.sensitivity = sensitivity
        self.porcupine = None
        self.stream = None
        self.audio_queue: queue.Queue[Any] = queue.Queue(maxsize=50)
        self.preroll_seconds = max(0.0, preroll_seconds)
        self.preroll_sample_rate = 0
        self._preroll_frames: collections.deque[Any] | None = None

    def __enter__(self) -> "PorcupineWakeListener":
        import pvporcupine

        kwargs = {
            "access_key": self.access_key,
            "sensitivities": [self.sensitivity],
        }
        if self.keyword_path:
            kwargs["keyword_paths"] = [str(self.keyword_path)]
        else:
            kwargs["keywords"] = [self.builtin_keyword]

        try:
            self.porcupine = pvporcupine.create(**kwargs)
        except Exception as exc:
            detail = type(exc).__name__
            if "ActivationLimit" in detail:
                detail += (
                    ": this Picovoice AccessKey appears to have reached its "
                    "activation/device limit. Create or reset an AccessKey in "
                    "Picovoice Console, then update PICOVOICE_API_KEY in the root .env."
                )
            raise RuntimeError(
                "Porcupine failed to initialize. "
                f"{detail}. Check PICOVOICE_API_KEY, Picovoice account status, "
                "and that any custom .ppn matches this platform. For Windows, "
                "use a Windows-trained .ppn; for Raspberry Pi later, train/download "
                "a separate Raspberry Pi .ppn."
            ) from exc
        try:
            self.sd.check_input_settings(
                device=self.input_device,
                samplerate=self.porcupine.sample_rate,
                channels=1,
                dtype="int16",
            )
        except Exception as exc:
            raise RuntimeError(
                "Wake listener needs a mono input format supported by Porcupine: "
                f"{self.porcupine.sample_rate} Hz, 1 channel. {exc}"
            ) from exc

        self.preroll_sample_rate = self.porcupine.sample_rate
        if self.preroll_seconds > 0 and self.porcupine.frame_length > 0:
            block_samples = self.porcupine.frame_length
            total_samples = int(self.preroll_seconds * self.preroll_sample_rate)
            maxlen = max(1, (total_samples + block_samples - 1) // block_samples + 2)
            self._preroll_frames = collections.deque(maxlen=maxlen)

        self.stream = self.sd.InputStream(
            samplerate=self.porcupine.sample_rate,
            blocksize=self.porcupine.frame_length,
            device=self.input_device,
            channels=1,
            dtype="int16",
            callback=self._input_callback,
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.porcupine is not None:
            self.porcupine.delete()
            self.porcupine = None

    def wait_for_detection(self, stop_event: threading.Event) -> bool:
        if self.porcupine is None:
            raise RuntimeError("Wake listener is not started.")

        while not stop_event.is_set():
            try:
                pcm = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                return True
        return False

    def _input_callback(self, indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if status:
            logger.debug("HOME_VOICE_WAKE: input status: %s", status)
        pcm_array = indata.reshape(-1).astype(self.np.int16, copy=False)
        if self._preroll_frames is not None:
            self._preroll_frames.append(pcm_array.copy())
        try:
            self.audio_queue.put_nowait(pcm_array.tolist())
        except queue.Full:
            pass

    def snapshot_preroll_pcm(self) -> Any | None:
        if not self._preroll_frames:
            return None
        frames = list(self._preroll_frames)
        if not frames:
            return None
        return self.np.concatenate(frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Always-on local home voice service: local wake word -> "
            "Realtime EMEET conversation -> back to wake listening."
        )
    )
    parser.add_argument("--user-id", default=os.getenv("VOICE_USER_ID", "user1"))
    parser.add_argument("--device-id", default=os.getenv("VOICE_DEVICE_ID", "windows-server-local-emeet"))
    parser.add_argument("--room", default=os.getenv("VOICE_ROOM", "office"))
    parser.add_argument("--friendly-name", default=os.getenv("VOICE_FRIENDLY_NAME", "Office Speaker"))
    parser.add_argument("--device-name", default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--wake-backend",
        choices=("openwakeword", "porcupine"),
        default=DEFAULT_WAKE_BACKEND,
        help="Wake-word backend. Default: VOICE_WAKE_BACKEND or openwakeword.",
    )
    parser.add_argument(
        "--openwakeword-model",
        type=Path,
        action="append",
        default=list(DEFAULT_OPENWAKEWORD_MODEL_PATHS),
        help="Path to an openWakeWord .tflite/.onnx model. Repeat for multiple models.",
    )
    parser.add_argument(
        "--openwakeword-label",
        default=DEFAULT_OPENWAKEWORD_LABEL,
        help="Optional prediction label/name filter, e.g. hey_jarvis or hey_skipper.",
    )
    parser.add_argument(
        "--openwakeword-threshold",
        type=float,
        default=DEFAULT_OPENWAKEWORD_THRESHOLD,
        help="Activation threshold. Default: VOICE_OPENWAKEWORD_THRESHOLD or 0.85. Higher = fewer false triggers.",
    )
    parser.add_argument(
        "--openwakeword-frame-ms",
        type=int,
        default=DEFAULT_OPENWAKEWORD_FRAME_MS,
        help="Wake inference frame size. OpenWakeWord recommends multiples of 80 ms.",
    )
    parser.add_argument(
        "--openwakeword-download-models",
        action="store_true",
        help="Download openWakeWord pretrained/preprocessing models before starting.",
    )
    parser.add_argument(
        "--openwakeword-inference-framework",
        choices=("auto", "onnx", "tflite"),
        default=DEFAULT_OPENWAKEWORD_INFERENCE_FRAMEWORK,
    )
    parser.add_argument(
        "--keyword-path",
        type=Path,
        default=DEFAULT_WAKE_KEYWORD_PATH,
        help="Path to a custom Porcupine .ppn. Only used with --wake-backend porcupine.",
    )
    parser.add_argument(
        "--builtin-keyword",
        default=DEFAULT_BUILTIN_KEYWORD,
        help="Built-in Porcupine keyword used only with --wake-backend porcupine.",
    )
    parser.add_argument("--sensitivity", type=float, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument(
        "--mode",
        choices=("relay", "direct"),
        default=os.getenv("VOICE_MODE", "relay"),
        help=(
            "relay (default): stream audio to the platform, which runs the OpenAI "
            "Realtime session, the tools, and speaker identification. "
            "direct: legacy — the satellite connects to OpenAI itself."
        ),
    )
    parser.add_argument(
        "--speaker-isolation",
        choices=("off", "optional", "required"),
        default=DEFAULT_SPEAKER_ISOLATION,
        help=(
            "Session speaker lock after wake. 'optional' uses resemblyzer if "
            "installed and falls back to normal audio; 'required' fails if unavailable."
        ),
    )
    parser.add_argument("--speaker-similarity", type=float, default=DEFAULT_SPEAKER_SIMILARITY)
    parser.add_argument("--speaker-rms", type=float, default=DEFAULT_SPEAKER_RMS)
    parser.add_argument("--speaker-silence-ms", type=int, default=DEFAULT_SPEAKER_SILENCE_MS)
    parser.add_argument("--speaker-enroll-min-ms", type=int, default=DEFAULT_SPEAKER_ENROLL_MIN_MS)
    parser.add_argument("--speaker-verify-min-ms", type=int, default=DEFAULT_SPEAKER_VERIFY_MIN_MS)
    parser.add_argument(
        "--max-session-seconds",
        type=float,
        default=DEFAULT_MAX_SESSION_SECONDS,
        help="Maximum conversation duration after wake. 0 disables the hard cap. Default: VOICE_MAX_SESSION_SECONDS or 300.",
    )
    parser.add_argument(
        "--initial-speech-timeout-seconds",
        type=float,
        default=DEFAULT_INITIAL_SPEECH_TIMEOUT_SECONDS,
        help="Return to wake mode if no user transcript arrives after wake. 0 disables. Default: VOICE_INITIAL_SPEECH_TIMEOUT_SECONDS or 20.",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT_SECONDS,
        help="Return to wake mode after this many quiet seconds in a conversation. 0 disables. Default: VOICE_IDLE_TIMEOUT_SECONDS or 45.",
    )
    parser.add_argument(
        "--preroll-seconds",
        type=float,
        default=DEFAULT_PREROLL_SECONDS,
        help=(
            "Rolling pre-roll buffer (seconds) injected into the realtime session "
            "after wake. Lets the user keep talking past the wake word without "
            "waiting for the beep. 0 disables. Default: VOICE_PREROLL_SECONDS or 3.0."
        ),
    )
    parser.add_argument(
        "--self-test-wake",
        action="store_true",
        help="Initialize the selected wake backend and EMEET mic, then exit.",
    )
    return parser


def _detect_alsa_card() -> str:
    """Best-effort ALSA playback card index: a USB card, else the first."""
    try:
        out = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        return ""
    first = ""
    for line in out.splitlines():
        m = re.match(r"card (\d+):", line)
        if not m:
            continue
        idx = m.group(1)
        first = first or idx
        if re.search(r"usb", line, re.IGNORECASE):
            return idx
    return first


def pin_output_volume() -> None:
    """Re-pin the speaker's ALSA output volume on startup.

    Some USB speakers reset their ALSA mixer to the lowest level on USB
    re-enumeration (a host reboot or replug), so the speaker can come up nearly
    silent. Pinning the mixer to VOICE_OUTPUT_VOLUME on every startup restores a
    known level. (A container restart alone does not re-enumerate the device.)

    Opt-in: does nothing unless VOICE_OUTPUT_VOLUME is set. Card/control default
    to a USB card and 'PCM'; override with VOICE_OUTPUT_CARD / VOICE_OUTPUT_MIXER.
    Never raises — a volume-pin failure must not stop the voice service.
    """
    raw = os.getenv("VOICE_OUTPUT_VOLUME", "").strip().rstrip("%")
    if not raw:
        return
    try:
        vol = max(0, min(100, int(raw)))
    except ValueError:
        print(f"[volume] ignoring invalid VOICE_OUTPUT_VOLUME={raw!r}")
        return
    card = os.getenv("VOICE_OUTPUT_CARD", "").strip() or _detect_alsa_card()
    mixer = os.getenv("VOICE_OUTPUT_MIXER", "").strip() or "PCM"
    if not card:
        print("[volume] no ALSA playback card found; skipping volume pin")
        return
    try:
        subprocess.run(
            ["amixer", "-c", str(card), "sset", mixer, f"{vol}%", "unmute"],
            check=True, capture_output=True, text=True,
        )
        print(f"[volume] pinned card {card} '{mixer}' to {vol}%")
    except FileNotFoundError:
        print("[volume] amixer not found (install alsa-utils); skipping volume pin")
    except subprocess.CalledProcessError as exc:
        print(f"[volume] could not set volume: {(exc.stderr or '').strip() or exc}")


def main() -> int:
    args = build_parser().parse_args()
    access_key = os.getenv("PICOVOICE_API_KEY", "").strip()
    if args.wake_backend == "porcupine" and not access_key:
        raise RuntimeError("PICOVOICE_API_KEY is required for Porcupine wake-word detection.")
    if args.wake_backend == "porcupine" and args.keyword_path and not args.keyword_path.exists():
        raise RuntimeError(f"Porcupine wake keyword file not found: {args.keyword_path}")
    if args.wake_backend == "openwakeword":
        for model_path in args.openwakeword_model:
            if not model_path.exists():
                raise RuntimeError(f"OpenWakeWord model file not found: {model_path}")

    np, sd = load_audio_deps()
    initialize_async_loop()
    stop_service = threading.Event()

    def request_stop(signum=None, frame=None) -> None:
        stop_service.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        input_name = args.input_name or DEFAULT_INPUT_NAME or args.device_name
        output_name = args.output_name or DEFAULT_OUTPUT_NAME or args.device_name
        list_audio_devices(sd)
        input_device = find_device(sd, input_name, "input")
        output_device = find_device(sd, output_name, "output")
        describe_selected_devices(sd, input_device, output_device)

        # Restore the speaker volume in case the device reset its ALSA mixer to
        # the lowest level on USB re-enumeration. Opt-in via VOICE_OUTPUT_VOLUME;
        # no-op when unset, never fatal.
        pin_output_volume()

        output_sample_rate, output_channels = resolve_audio_settings(
            sd,
            device_index=output_device,
            kind="output",
            preferred_rate=REALTIME_AUDIO_RATE,
            preferred_channels=1,
            use_device_default_rate=False,
        )

        keyword_label = wake_label(args)
        threshold_value = (
            args.openwakeword_threshold
            if args.wake_backend == "openwakeword"
            else args.sensitivity
        )
        print("\nHome voice wake service is running.")
        print(f"  wake backend: {args.wake_backend}")
        print(f"  wake keyword: {keyword_label}")
        print(f"  threshold:    {threshold_value:.3f}")
        print(f"  room:         {args.room}")
        print("  press Ctrl+C to stop")
        if args.wake_backend == "openwakeword" and not args.openwakeword_model:
            print("\nNo custom OpenWakeWord model configured yet. For quick testing, use a")
            print("bundled model/label such as --openwakeword-label hey_jarvis after")
            print("downloading models with --openwakeword-download-models.")
        elif args.wake_backend == "porcupine" and not args.keyword_path:
            print("\nNo custom .ppn configured. Porcupine will use the built-in keyword.")

        if args.self_test_wake:
            self_test_wake_listener(
                np,
                sd,
                args=args,
                access_key=access_key,
                input_device=input_device,
            )
            return 0

        # Tools run on the platform; nothing to initialize locally. Each wake
        # creates a server-side session whose config carries the tool list.
        while not stop_service.is_set():
            print("\nListening for wake word...")
            listener = create_wake_listener(
                np,
                sd,
                args=args,
                access_key=access_key,
                input_device=input_device,
            )
            session_config = None
            preroll_pcm = None
            preroll_sample_rate = 0
            with listener:
                detected = listener.wait_for_detection(stop_service)
                if not detected or stop_service.is_set():
                    break
                print("Wake word detected.")
                # NOTE: the "ready" chime is NOT played here anymore. Firing it at
                # wake-detection was a false "go" — the mic→model stream isn't live until the
                # conversation's audio bridge starts (after create_session + the relay/session
                # connect), ~1s later. The chime now fires there (see run_*_conversation), so
                # it honestly means "speak now." Pre-roll still covers anything said before it.
                device_info = {
                    "platform": "home_voice",
                    "device_type": "home_voice",
                    "device_id": args.device_id,
                    "room": args.room,
                    "friendly_name": args.friendly_name,
                    "audio_device_name": args.device_name,
                }
                session_config = skc.create_session(args.user_id, device_info)
                if session_config and args.preroll_seconds > 0:
                    preroll_pcm = listener.snapshot_preroll_pcm()
                    preroll_sample_rate = listener.preroll_sample_rate
                    if preroll_pcm is not None and preroll_sample_rate > 0:
                        duration = len(preroll_pcm) / preroll_sample_rate
                        print(f"Captured {duration:.2f}s of pre-roll audio.")

            if not session_config:
                print("Could not create realtime voice session; returning to wake listening.")
                if not stop_service.is_set():
                    time.sleep(args.cooldown_seconds)
                continue

            conversation = (
                run_relay_conversation if args.mode == "relay"
                else run_realtime_conversation
            )
            conversation(
                np,
                sd,
                args=args,
                input_device=input_device,
                output_device=output_device,
                stop_service=stop_service,
                session_config=session_config,
                preroll_pcm=preroll_pcm,
                preroll_sample_rate=preroll_sample_rate,
            )
            if not stop_service.is_set():
                print(f"Cooling down for {args.cooldown_seconds:.1f}s...")
                time.sleep(args.cooldown_seconds)

    finally:
        shutdown_async_loop()

    print("Home voice wake service stopped.")
    return 0


def self_test_wake_listener(
    np: Any,
    sd: Any,
    *,
    args: argparse.Namespace,
    access_key: str,
    input_device: int | None,
) -> None:
    print(f"\nTesting {args.wake_backend} wake listener initialization...")
    listener = create_wake_listener(
        np,
        sd,
        args=args,
        access_key=access_key,
        input_device=input_device,
    )
    with listener:
        print("Wake listener initialized successfully.")
        if args.wake_backend == "porcupine":
            print(f"  sample_rate:  {listener.porcupine.sample_rate}")
            print(f"  frame_length: {listener.porcupine.frame_length}")
        else:
            print(f"  sample_rate:  {listener.SAMPLE_RATE}")
            print(f"  blocksize:    {listener.blocksize}")
            print(f"  threshold:    {listener.threshold}")


def create_wake_listener(
    np: Any,
    sd: Any,
    *,
    args: argparse.Namespace,
    access_key: str,
    input_device: int | None,
):
    if args.wake_backend == "porcupine":
        return PorcupineWakeListener(
            np,
            sd,
            access_key=access_key,
            input_device=input_device,
            keyword_path=args.keyword_path,
            builtin_keyword=args.builtin_keyword,
            sensitivity=args.sensitivity,
            preroll_seconds=args.preroll_seconds,
        )

    return OpenWakeWordWakeListener(
        np,
        sd,
        input_device=input_device,
        model_paths=args.openwakeword_model,
        threshold=args.openwakeword_threshold,
        label=args.openwakeword_label,
        frame_ms=args.openwakeword_frame_ms,
        download_models=args.openwakeword_download_models,
        inference_framework=args.openwakeword_inference_framework,
        preroll_seconds=args.preroll_seconds,
    )


def wake_label(args: argparse.Namespace) -> str:
    if args.wake_backend == "porcupine":
        return str(args.keyword_path) if args.keyword_path else f"built-in '{args.builtin_keyword}'"
    if args.openwakeword_model:
        models = ", ".join(str(path) for path in args.openwakeword_model)
    else:
        models = "bundled/pretrained models"
    if args.openwakeword_label:
        return f"{models}; label contains '{args.openwakeword_label}'"
    return models


def play_wake_chime(
    np: Any,
    sd: Any,
    *,
    output_device: int | None,
    output_sample_rate: int,
    output_channels: int,
) -> None:
    try:
        play_test_tone(
            np,
            sd,
            output_device=output_device,
            sample_rate=output_sample_rate,
            channels=output_channels,
            seconds=0.18,
            frequency=880.0,
            volume=0.12,
        )
    except Exception as exc:
        logger.debug("HOME_VOICE_WAKE: chime failed: %s", exc)


def play_end_chime(
    np: Any,
    sd: Any,
    *,
    output_device: int | None,
    output_sample_rate: int,
    output_channels: int,
) -> None:
    try:
        play_chime_sequence(
            np,
            sd,
            output_device=output_device,
            sample_rate=output_sample_rate,
            channels=output_channels,
            tones=[
                (660.0, 0.10, 0.10),
                (440.0, 0.14, 0.10),
            ],
            gap_seconds=0.03,
        )
    except Exception as exc:
        logger.debug("HOME_VOICE_WAKE: end chime failed: %s", exc)


def play_chime_sequence(
    np: Any,
    sd: Any,
    *,
    output_device: int | None,
    sample_rate: int,
    channels: int,
    tones: list[tuple[float, float, float]],
    gap_seconds: float,
) -> None:
    chunks = []
    for frequency, seconds, volume in tones:
        frames = max(1, int(sample_rate * seconds))
        t = np.arange(frames, dtype=np.float32) / sample_rate
        tone = np.sin(2 * np.pi * frequency * t).astype(np.float32) * volume

        fade_frames = min(int(sample_rate * 0.015), frames // 2)
        if fade_frames:
            fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
            tone[:fade_frames] *= fade
            tone[-fade_frames:] *= fade[::-1]

        chunks.append(tone)
        gap_frames = int(sample_rate * gap_seconds)
        if gap_frames > 0:
            chunks.append(np.zeros(gap_frames, dtype=np.float32))

    audio_mono = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    audio = np.repeat(audio_mono.reshape(-1, 1), channels, axis=1)
    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()


def run_realtime_conversation(
    np: Any,
    sd: Any,
    *,
    args: argparse.Namespace,
    input_device: int | None,
    output_device: int | None,
    stop_service: threading.Event,
    session_config: dict,
    preroll_pcm: Any | None = None,
    preroll_sample_rate: int = 0,
) -> None:
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

    print("Starting realtime conversation.")
    print(f"  active_app: {session_config.get('active_app')}")
    print(f"  tools:      {len(session_config.get('base_tools', []))}")
    speaker_gate = build_speaker_gate(
        np,
        sample_rate=REALTIME_AUDIO_RATE,
        frame_ms=args.frame_ms,
        args=args,
    )
    print(f"  speaker lock: {'on' if speaker_gate else 'off'}")
    print("Say goodbye, stop, or I'm done to return to wake-word mode.")

    session_stop = threading.Event()
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
        stop_event=session_stop,
    )
    client.sideband = skc.Sideband(
        session_config["session_id"], client._dispatch_platform_event)
    client.sideband.start()
    ws_thread = threading.Thread(target=client.run, daemon=True)

    try:
        if preroll_pcm is not None and preroll_sample_rate > 0:
            injected_samples = audio_bridge.inject_preroll_pcm(
                preroll_pcm,
                source_sample_rate=preroll_sample_rate,
            )
            if injected_samples > 0:
                duration = injected_samples / REALTIME_AUDIO_RATE
                print(f"  injected pre-roll: {duration:.2f}s")
        # Chime right BEFORE the bridge opens its output stream. play_test_tone needs its own
        # transient output stream; once the bridge owns the device (in .start()) a second open
        # fails on ALSA. The bridge starts capturing immediately after, so this is still an
        # honest "speak now" cue — speech right after the beep is captured and buffered.
        play_wake_chime(
            np, sd,
            output_device=output_device,
            output_sample_rate=output_sample_rate,
            output_channels=output_channels,
        )
        audio_bridge.start()
        ws_thread.start()
        started = time.monotonic()
        while not stop_service.is_set() and not session_stop.is_set():
            if (
                args.initial_speech_timeout_seconds > 0
                and not client.first_user_transcript_at
                and client.waiting_for_first_user_transcript_seconds()
                >= args.initial_speech_timeout_seconds
            ):
                print("No speech after wake; returning to wake-word mode.")
                break
            if (
                args.idle_timeout_seconds > 0
                and client.first_user_transcript_at
                and client.idle_seconds() >= args.idle_timeout_seconds
            ):
                print("Conversation idle timeout reached; returning to wake-word mode.")
                break
            if args.max_session_seconds > 0 and time.monotonic() - started >= args.max_session_seconds:
                print("Max session duration reached; ending conversation.")
                break
            time.sleep(0.1)
    finally:
        client.close()
        if client.sideband is not None:
            client.sideband.close()
        audio_bridge.stop()
        if not stop_service.is_set():
            play_end_chime(
                np,
                sd,
                output_device=output_device,
                output_sample_rate=output_sample_rate,
                output_channels=output_channels,
            )
        skc.end_session(session_config["session_id"])

    print("Conversation ended; returning to wake-word mode.")


def run_relay_conversation(
    np: Any,
    sd: Any,
    *,
    args: argparse.Namespace,
    input_device: int | None,
    output_device: int | None,
    stop_service: threading.Event,
    session_config: dict,
    preroll_pcm: Any | None = None,
    preroll_sample_rate: int = 0,
) -> None:
    """Relay mode: stream 2-way audio to the platform's /ws/voice/audio relay.

    The platform runs the OpenAI Realtime session, the tools, and speaker
    identification. The satellite is just wake word + AEC + audio I/O — so no
    local speaker gate and no sideband here.
    """
    from host_relay_client import RelayClient

    input_sample_rate, input_channels = resolve_audio_settings(
        sd, device_index=input_device, kind="input",
        preferred_rate=16000, preferred_channels=1, use_device_default_rate=False,
    )
    output_sample_rate, output_channels = resolve_audio_settings(
        sd, device_index=output_device, kind="output",
        preferred_rate=REALTIME_AUDIO_RATE, preferred_channels=1, use_device_default_rate=False,
    )

    print("Starting relay conversation (OpenAI session + tools + speaker ID run on the host).")
    print(f"  active_app: {session_config.get('active_app')}")
    print(f"  host:       {skc.DEFAULT_API_BASE}")
    print("Say goodbye, stop, or I'm done to return to wake-word mode.")

    session_stop = threading.Event()
    audio_bridge = RealtimeAudioBridge(
        np, sd,
        input_device=input_device,
        output_device=output_device,
        input_sample_rate=input_sample_rate,
        input_channels=input_channels,
        output_sample_rate=output_sample_rate,
        output_channels=output_channels,
        frame_ms=args.frame_ms,
        speaker_gate=None,  # speaker ID is the host's job in relay mode
    )
    client = RelayClient(
        session_config=session_config,
        audio_bridge=audio_bridge,
        stop_event=session_stop,
        api_base=skc.DEFAULT_API_BASE,
    )
    ws_thread = threading.Thread(target=client.run, daemon=True)

    try:
        if preroll_pcm is not None and preroll_sample_rate > 0:
            injected_samples = audio_bridge.inject_preroll_pcm(
                preroll_pcm, source_sample_rate=preroll_sample_rate,
            )
            if injected_samples > 0:
                print(f"  injected pre-roll: {injected_samples / REALTIME_AUDIO_RATE:.2f}s")
        # Chime right BEFORE the bridge opens its output stream. play_test_tone needs its own
        # transient output stream; once the bridge owns the device (in .start()) a second open
        # fails on ALSA. The bridge starts capturing immediately after, so this is still an
        # honest "speak now" cue — speech right after the beep is captured and buffered.
        play_wake_chime(
            np, sd,
            output_device=output_device,
            output_sample_rate=output_sample_rate,
            output_channels=output_channels,
        )
        audio_bridge.start()
        ws_thread.start()
        started = time.monotonic()
        while not stop_service.is_set() and not session_stop.is_set():
            if (args.initial_speech_timeout_seconds > 0
                    and not client.first_user_transcript_at
                    and client.waiting_for_first_user_transcript_seconds()
                    >= args.initial_speech_timeout_seconds):
                print("No speech after wake; returning to wake-word mode.")
                break
            if (args.idle_timeout_seconds > 0
                    and client.first_user_transcript_at
                    and client.idle_seconds() >= args.idle_timeout_seconds):
                print("Conversation idle timeout reached; returning to wake-word mode.")
                break
            if args.max_session_seconds > 0 and time.monotonic() - started >= args.max_session_seconds:
                print("Max session duration reached; ending conversation.")
                break
            time.sleep(0.1)
    finally:
        client.close()
        audio_bridge.stop()
        if not stop_service.is_set():
            play_end_chime(
                np, sd,
                output_device=output_device,
                output_sample_rate=output_sample_rate,
                output_channels=output_channels,
            )
        skc.end_session(session_config["session_id"])

    print("Conversation ended; returning to wake-word mode.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)
    except Exception as exc:
        logger.error("HOME_VOICE_WAKE: service failed: %s", exc, exc_info=True)
        print(f"\nHome voice wake service failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
