from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import time
import wave
from pathlib import Path
from typing import Any


DEFAULT_DEVICE_NAME = os.getenv("VOICE_DEVICE_NAME", "EMEET")
DEFAULT_INPUT_NAME = os.getenv("VOICE_DEVICE_INPUT_NAME", DEFAULT_DEVICE_NAME)
DEFAULT_OUTPUT_NAME = os.getenv("VOICE_DEVICE_OUTPUT_NAME", DEFAULT_DEVICE_NAME)
DEFAULT_SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
DEFAULT_CHANNELS = int(os.getenv("VOICE_CHANNELS", "1"))
DEFAULT_FRAME_MS = int(os.getenv("VOICE_FRAME_MS", "20"))
DEFAULT_RECORDING_PATH = Path(__file__).with_name("last_recording.wav")


def load_audio_deps() -> tuple[Any, Any]:
    try:
        import numpy as np
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        missing = exc.name or "audio dependency"
        print(f"Missing dependency: {missing}", file=sys.stderr)
        print(
            "Install Phase 1 audio test dependencies with:\n"
            "  pip install -r home_voice/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    return np, sd


def list_audio_devices(sd: Any) -> None:
    devices = sd.query_devices()
    default_input, default_output = _default_device_indexes(sd)

    print("Audio devices detected by Python/PortAudio:\n")
    for index, dev in enumerate(devices):
        markers = []
        if index == default_input:
            markers.append("default input")
        if index == default_output:
            markers.append("default output")

        marker_text = f" ({', '.join(markers)})" if markers else ""
        print(f"{index}: {dev['name']}{marker_text}")
        print(f"    host api:            {sd.query_hostapis(dev['hostapi'])['name']}")
        print(f"    max input channels:  {dev['max_input_channels']}")
        print(f"    max output channels: {dev['max_output_channels']}")
        print(f"    default samplerate:  {dev['default_samplerate']}")


def _default_device_indexes(sd: Any) -> tuple[int | None, int | None]:
    default_device = sd.default.device
    if isinstance(default_device, (list, tuple)):
        input_index = default_device[0] if len(default_device) > 0 else None
        output_index = default_device[1] if len(default_device) > 1 else None
        return input_index, output_index
    return None, None


def find_device(sd: Any, name_contains: str | None, kind: str) -> int | None:
    if not name_contains or name_contains.lower() == "default":
        return None

    needle = name_contains.lower()
    devices = sd.query_devices()

    for index, dev in enumerate(devices):
        name = dev["name"].lower()
        if needle not in name:
            continue
        if kind == "input" and dev["max_input_channels"] > 0:
            return index
        if kind == "output" and dev["max_output_channels"] > 0:
            return index

    candidates = [
        f"{index}: {dev['name']}"
        for index, dev in enumerate(devices)
        if (kind == "input" and dev["max_input_channels"] > 0)
        or (kind == "output" and dev["max_output_channels"] > 0)
    ]
    raise RuntimeError(
        f"Could not find {kind} device containing {name_contains!r}.\n"
        f"Available {kind} devices:\n  " + "\n  ".join(candidates)
    )


def describe_selected_devices(sd: Any, input_device: int | None, output_device: int | None) -> None:
    print("\nSelected devices:")
    print(f"  input:  {_device_label(sd, input_device, 'input')}")
    print(f"  output: {_device_label(sd, output_device, 'output')}")


def _device_label(sd: Any, device_index: int | None, kind: str) -> str:
    if device_index is None:
        default_input, default_output = _default_device_indexes(sd)
        device_index = default_input if kind == "input" else default_output
        if device_index is None or device_index < 0:
            return "system default"
        return f"{device_index}: {sd.query_devices(device_index)['name']} (system default)"
    return f"{device_index}: {sd.query_devices(device_index)['name']}"


def device_default_sample_rate(sd: Any, device_index: int | None, kind: str) -> int:
    resolved_index = resolve_device_index(sd, device_index, kind)
    if resolved_index is None:
        return DEFAULT_SAMPLE_RATE
    return int(sd.query_devices(resolved_index)["default_samplerate"])


def resolve_device_index(sd: Any, device_index: int | None, kind: str) -> int | None:
    if device_index is not None:
        return device_index
    default_input, default_output = _default_device_indexes(sd)
    resolved_index = default_input if kind == "input" else default_output
    if resolved_index is None or resolved_index < 0:
        return None
    return resolved_index


def resolve_audio_settings(
    sd: Any,
    *,
    device_index: int | None,
    kind: str,
    preferred_rate: int,
    preferred_channels: int,
    use_device_default_rate: bool,
) -> tuple[int, int]:
    resolved_index = resolve_device_index(sd, device_index, kind)
    default_rate = device_default_sample_rate(sd, device_index, kind)
    max_channels = _max_channels(sd, resolved_index, kind)
    channel_candidates = _unique_ints(
        preferred_channels,
        min(max_channels, 2) if max_channels else preferred_channels,
        1,
    )
    rate_candidates = (
        _unique_ints(default_rate, preferred_rate)
        if use_device_default_rate
        else _unique_ints(preferred_rate, default_rate, 48000, 44100, 16000)
    )

    errors = []
    for sample_rate in rate_candidates:
        for channels in channel_candidates:
            try:
                if kind == "input":
                    sd.check_input_settings(
                        device=device_index,
                        samplerate=sample_rate,
                        channels=channels,
                        dtype="int16",
                    )
                else:
                    sd.check_output_settings(
                        device=device_index,
                        samplerate=sample_rate,
                        channels=channels,
                        dtype="float32",
                    )
                if sample_rate != preferred_rate or channels != preferred_channels:
                    print(
                        f"  {kind} format fallback: "
                        f"{preferred_rate} Hz/{preferred_channels} ch -> "
                        f"{sample_rate} Hz/{channels} ch"
                    )
                return sample_rate, channels
            except Exception as exc:
                errors.append(f"{sample_rate} Hz/{channels} ch: {exc}")

    raise RuntimeError(
        f"Could not find usable {kind} audio settings for {_device_label(sd, device_index, kind)}.\n"
        + "\n".join(f"  {error}" for error in errors)
    )


def _max_channels(sd: Any, device_index: int | None, kind: str) -> int:
    if device_index is None:
        return 2
    dev = sd.query_devices(device_index)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    return int(dev[key])


def _unique_ints(*values: int) -> list[int]:
    result = []
    for value in values:
        value = int(value)
        if value > 0 and value not in result:
            result.append(value)
    return result


def play_test_tone(
    np: Any,
    sd: Any,
    *,
    output_device: int | None,
    sample_rate: int,
    channels: int,
    seconds: float,
    frequency: float,
    volume: float,
) -> None:
    print(f"\nPlaying {seconds:.1f}s test tone at {frequency:.0f} Hz...")
    frames = max(1, int(sample_rate * seconds))
    t = np.arange(frames, dtype=np.float32) / sample_rate
    tone = np.sin(2 * np.pi * frequency * t).astype(np.float32) * volume

    fade_frames = min(int(sample_rate * 0.02), frames // 2)
    if fade_frames:
        fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
        tone[:fade_frames] *= fade
        tone[-fade_frames:] *= fade[::-1]

    audio = np.repeat(tone.reshape(-1, 1), channels, axis=1)
    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()
    print("Tone playback finished.")


def record_audio(
    np: Any,
    sd: Any,
    *,
    input_device: int | None,
    sample_rate: int,
    channels: int,
    seconds: float,
) -> Any:
    print(f"\nRecording {seconds:.1f}s from microphone...")
    print("Speak near the EMEET now.")
    frames = max(1, int(sample_rate * seconds))
    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=input_device,
    )
    sd.wait()
    print("Recording finished.")
    print_audio_levels(np, recording)
    return recording


def print_audio_levels(np: Any, audio: Any) -> None:
    if audio.size == 0:
        print("Audio level: no samples captured.")
        return

    normalized = audio.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    peak = float(np.max(np.abs(normalized)))
    rms_db = 20 * math.log10(max(rms, 1e-12))
    peak_db = 20 * math.log10(max(peak, 1e-12))

    print("Captured audio level:")
    print(f"  RMS:  {rms:.4f} ({rms_db:.1f} dBFS)")
    print(f"  Peak: {peak:.4f} ({peak_db:.1f} dBFS)")

    if peak < 0.01:
        print("  Note: the recording is very quiet. Check input device, mute, or gain.")
    elif peak > 0.95:
        print("  Note: the recording is close to clipping. Lower input gain if it sounds distorted.")


def save_wav(path: Path, audio: Any, *, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    print(f"Saved recording: {path}")


def play_audio(
    np: Any,
    sd: Any,
    audio: Any,
    *,
    output_device: int | None,
    input_sample_rate: int,
    output_sample_rate: int,
    output_channels: int,
) -> None:
    print("\nPlaying recorded sample back through speaker...")
    output_audio = convert_audio(
        np,
        audio,
        input_sample_rate=input_sample_rate,
        output_sample_rate=output_sample_rate,
        output_channels=output_channels,
    )
    sd.play(output_audio, samplerate=output_sample_rate, device=output_device)
    sd.wait()
    print("Playback finished.")


def convert_audio(
    np: Any,
    audio: Any,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
    output_channels: int,
) -> Any:
    converted = audio

    if converted.ndim == 1:
        converted = converted.reshape(-1, 1)

    if input_sample_rate != output_sample_rate:
        converted = resample_audio(np, converted, input_sample_rate, output_sample_rate)

    if converted.shape[1] != output_channels:
        converted = adapt_channels(np, converted, output_channels)

    return converted.astype("int16", copy=False)


def resample_audio(np: Any, audio: Any, input_sample_rate: int, output_sample_rate: int) -> Any:
    if audio.shape[0] <= 1:
        return audio

    duration = audio.shape[0] / input_sample_rate
    output_frames = max(1, int(round(duration * output_sample_rate)))
    old_positions = np.linspace(0, audio.shape[0] - 1, audio.shape[0], dtype=np.float32)
    new_positions = np.linspace(0, audio.shape[0] - 1, output_frames, dtype=np.float32)

    channels = [
        np.interp(new_positions, old_positions, audio[:, channel]).astype(np.float32)
        for channel in range(audio.shape[1])
    ]
    resampled = np.stack(channels, axis=1)
    return np.clip(np.rint(resampled), -32768, 32767).astype("int16")


def adapt_channels(np: Any, audio: Any, output_channels: int) -> Any:
    input_channels = audio.shape[1]
    if input_channels == output_channels:
        return audio
    if output_channels == 1:
        return np.mean(audio.astype(np.float32), axis=1, keepdims=True).astype("int16")
    if input_channels == 1:
        return np.repeat(audio, output_channels, axis=1)

    if input_channels > output_channels:
        return audio[:, :output_channels]

    repeats = output_channels - input_channels
    extra = np.repeat(audio[:, -1:], repeats, axis=1)
    return np.concatenate([audio, extra], axis=1)


def run_loopback(
    np: Any,
    sd: Any,
    *,
    input_device: int | None,
    output_device: int | None,
    input_sample_rate: int,
    output_sample_rate: int,
    input_channels: int,
    output_channels: int,
    frame_ms: int,
    seconds: float,
) -> None:
    print("\nLoopback test requested.")
    print("Lower the EMEET volume now. This can create feedback.")
    for remaining in range(3, 0, -1):
        print(f"Starting loopback in {remaining}...")
        time.sleep(1)

    input_blocksize = max(1, int(input_sample_rate * frame_ms / 1000))
    output_blocksize = max(1, int(output_sample_rate * frame_ms / 1000))
    audio_queue: queue.Queue[Any] = queue.Queue(maxsize=20)

    def input_callback(indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if status:
            print(f"Input status: {status}", file=sys.stderr)
        output_audio = convert_audio(
            np,
            indata,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_channels=output_channels,
        )
        try:
            audio_queue.put_nowait(output_audio.copy())
        except queue.Full:
            pass

    def output_callback(outdata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if status:
            print(f"Output status: {status}", file=sys.stderr)
        try:
            outdata[:] = audio_queue.get_nowait()
        except queue.Empty:
            outdata.fill(0)

    print(f"Running loopback for {seconds:.1f}s...")
    with sd.InputStream(
        samplerate=input_sample_rate,
        blocksize=input_blocksize,
        device=input_device,
        channels=input_channels,
        dtype="int16",
        callback=input_callback,
    ), sd.OutputStream(
        samplerate=output_sample_rate,
        blocksize=output_blocksize,
        device=output_device,
        channels=output_channels,
        dtype="int16",
        callback=output_callback,
    ):
        time.sleep(seconds)
    print("Loopback finished.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basic Phase 1 audio tests for the EMEET speakerphone."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "list", "tone", "record"),
        default="all",
        help="Test mode. Default: all.",
    )
    parser.add_argument(
        "--device-name",
        default=DEFAULT_DEVICE_NAME,
        help="Fallback partial name for both input and output devices. Default: EMEET.",
    )
    parser.add_argument(
        "--input-name",
        default=None,
        help="Partial input device name. Defaults to --device-name.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Partial output device name. Defaults to --device-name.",
    )
    parser.add_argument(
        "--use-default-devices",
        action="store_true",
        help="Use the system default input/output devices instead of matching by name.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate for tests. Default: VOICE_SAMPLE_RATE or 16000.",
    )
    parser.add_argument(
        "--use-device-default-rate",
        action="store_true",
        help="Use the selected device default sample rate instead of --sample-rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help="Audio channels. Default: VOICE_CHANNELS or 1.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=3.0,
        help="Seconds to record in all/record mode. Default: 3.",
    )
    parser.add_argument(
        "--tone-seconds",
        type=float,
        default=1.0,
        help="Seconds to play tone in all/tone mode. Default: 1.",
    )
    parser.add_argument(
        "--tone-frequency",
        type=float,
        default=440.0,
        help="Tone frequency in Hz. Default: 440.",
    )
    parser.add_argument(
        "--tone-volume",
        type=float,
        default=0.15,
        help="Tone volume from 0.0 to 1.0. Default: 0.15.",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Record but do not play the recording back.",
    )
    parser.add_argument(
        "--save-wav",
        type=Path,
        default=DEFAULT_RECORDING_PATH,
        help=f"Path for captured WAV. Default: {DEFAULT_RECORDING_PATH}.",
    )
    parser.add_argument(
        "--loopback-seconds",
        type=float,
        default=0.0,
        help="Optional live mic-to-speaker loopback duration. Default: disabled.",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=DEFAULT_FRAME_MS,
        help="Loopback frame size in milliseconds. Default: VOICE_FRAME_MS or 20.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    np, sd = load_audio_deps()

    list_audio_devices(sd)
    if args.mode == "list" and args.loopback_seconds <= 0:
        return 0

    if args.use_default_devices:
        input_device = None
        output_device = None
    else:
        input_name = args.input_name or DEFAULT_INPUT_NAME or args.device_name
        output_name = args.output_name or DEFAULT_OUTPUT_NAME or args.device_name
        input_device = find_device(sd, input_name, "input")
        output_device = find_device(sd, output_name, "output")

    describe_selected_devices(sd, input_device, output_device)

    print("\nResolving usable audio formats...")
    input_sample_rate, input_channels = resolve_audio_settings(
        sd,
        device_index=input_device,
        kind="input",
        preferred_rate=args.sample_rate,
        preferred_channels=args.channels,
        use_device_default_rate=args.use_device_default_rate,
    )
    output_sample_rate, output_channels = resolve_audio_settings(
        sd,
        device_index=output_device,
        kind="output",
        preferred_rate=args.sample_rate,
        preferred_channels=args.channels,
        use_device_default_rate=args.use_device_default_rate,
    )
    print(f"  input:  {input_sample_rate} Hz, {input_channels} channel(s), int16")
    print(f"  output: {output_sample_rate} Hz, {output_channels} channel(s), int16/float32")

    if args.mode in ("all", "tone"):
        play_test_tone(
            np,
            sd,
            output_device=output_device,
            sample_rate=output_sample_rate,
            channels=output_channels,
            seconds=args.tone_seconds,
            frequency=args.tone_frequency,
            volume=args.tone_volume,
        )

    if args.mode in ("all", "record"):
        recording = record_audio(
            np,
            sd,
            input_device=input_device,
            sample_rate=input_sample_rate,
            channels=input_channels,
            seconds=args.record_seconds,
        )
        save_wav(args.save_wav, recording, sample_rate=input_sample_rate, channels=input_channels)
        if not args.no_playback:
            play_audio(
                np,
                sd,
                recording,
                output_device=output_device,
                input_sample_rate=input_sample_rate,
                output_sample_rate=output_sample_rate,
                output_channels=output_channels,
            )

    if args.loopback_seconds > 0:
        run_loopback(
            np,
            sd,
            input_device=input_device,
            output_device=output_device,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            input_channels=input_channels,
            output_channels=output_channels,
            frame_ms=args.frame_ms,
            seconds=args.loopback_seconds,
        )

    print("\nBasic audio test complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nAudio test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
