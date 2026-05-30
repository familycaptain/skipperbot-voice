from __future__ import annotations

import argparse
import asyncio
import os
import sys
import wave
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basic_audio_test import (  # noqa: E402
    DEFAULT_DEVICE_NAME,
    DEFAULT_INPUT_NAME,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_SAMPLE_RATE,
    describe_selected_devices,
    find_device,
    list_audio_devices,
    load_audio_deps,
    play_audio,
    record_audio,
    resolve_audio_settings,
    save_wav,
)
from chat import process_chat  # noqa: E402
from config import logger, openai_client  # noqa: E402
from voice_prompting import build_base_voice_payload  # noqa: E402


DEFAULT_INPUT_PATH = Path(__file__).with_name("one_shot_input.wav")
DEFAULT_RESPONSE_PATH = Path(__file__).with_name("one_shot_response.wav")
DEFAULT_STT_MODEL = os.getenv("VOICE_STT_MODEL", "gpt-4o-mini-transcribe")
DEFAULT_TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "ash")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record one EMEET utterance, transcribe it, send it through Skipper's "
            "normal text brain with voice context, synthesize the response, and "
            "play it back through the EMEET speaker."
        )
    )
    parser.add_argument("--user-id", default=os.getenv("VOICE_USER_ID", "user1"))
    parser.add_argument("--device-id", default=os.getenv("VOICE_DEVICE_ID", "windows-server-local-emeet"))
    parser.add_argument("--room", default=os.getenv("VOICE_ROOM", "office"))
    parser.add_argument("--friendly-name", default=os.getenv("VOICE_FRIENDLY_NAME", "Office Speaker"))
    parser.add_argument("--device-name", default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--record-seconds", type=float, default=5.0)
    parser.add_argument("--input-wav", type=Path, default=None, help="Use an existing WAV instead of recording.")
    parser.add_argument("--save-input-wav", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--save-response-wav", type=Path, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE)
    parser.add_argument("--no-playback", action="store_true", help="Create response WAV but do not play it.")
    parser.add_argument(
        "--skip-tool-init",
        action="store_true",
        help="Skip MCP/direct tool initialization. Useful for a pure chat smoke test.",
    )
    return parser


async def main_async() -> int:
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
    payload = build_base_voice_payload(user_id=args.user_id, device_info=device_info)
    print("Shared home voice profile:")
    print(f"  user_id:            {args.user_id}")
    print(f"  room:               {args.room}")
    print(f"  active_app:         {payload.get('app')}")
    print(f"  default_categories: {', '.join(payload.get('default_categories', []))}")
    print(f"  tools:              {len(payload.get('tools', []))}")

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
        preferred_rate=args.sample_rate,
        preferred_channels=args.channels,
        use_device_default_rate=False,
    )
    output_sample_rate, output_channels = resolve_audio_settings(
        sd,
        device_index=output_device,
        kind="output",
        preferred_rate=args.sample_rate,
        preferred_channels=args.channels,
        use_device_default_rate=False,
    )

    input_wav = args.input_wav
    if input_wav is None:
        recording = record_audio(
            np,
            sd,
            input_device=input_device,
            sample_rate=input_sample_rate,
            channels=input_channels,
            seconds=args.record_seconds,
        )
        save_wav(args.save_input_wav, recording, sample_rate=input_sample_rate, channels=input_channels)
        input_wav = args.save_input_wav

    if not args.skip_tool_init:
        await initialize_skipper_tools()

    transcript = await transcribe_audio(input_wav, args.stt_model)
    print(f"\nYou said: {transcript}")
    if not transcript.strip():
        print("No transcript returned; stopping before Skipper/TTS.")
        return 1

    response_text = await process_chat(
        args.user_id,
        transcript,
        channel="voice",
        app_context=device_info,
    )
    response_text = (response_text or "").strip()
    print(f"\nSkipper: {response_text}")
    if not response_text:
        print("Skipper returned an empty response; stopping before TTS.")
        return 1

    await synthesize_response(
        response_text,
        path=args.save_response_wav,
        model=args.tts_model,
        voice=args.tts_voice,
    )
    print(f"\nSaved response audio: {args.save_response_wav}")

    if not args.no_playback:
        response_audio, response_rate, response_channels = read_wav_int16(np, args.save_response_wav)
        print("Playing Skipper response through EMEET...")
        play_audio(
            np,
            sd,
            response_audio,
            output_device=output_device,
            input_sample_rate=response_rate,
            output_sample_rate=output_sample_rate,
            output_channels=output_channels,
        )

    print("\nOne-shot Skipper response test complete.")
    return 0


async def initialize_skipper_tools() -> None:
    """Initialize MCP and direct tool dispatch for standalone voice tests."""
    print("\nInitializing Skipper tool routing...")
    import mcp_client
    import tool_dispatch

    tools = await mcp_client.connect_to_mcp()
    await asyncio.to_thread(tool_dispatch.init)
    tool_dispatch.verify_against_mcp([tool.name for tool in tools])
    print(f"Tool routing ready: {len(tools)} MCP tools discovered.")


async def transcribe_audio(path: Path, model: str) -> str:
    print(f"\nTranscribing {path} with {model}...")

    def _run() -> str:
        with open(path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )
        return getattr(transcription, "text", str(transcription))

    return await asyncio.to_thread(_run)


async def synthesize_response(text: str, *, path: Path, model: str, voice: str) -> None:
    print(f"\nSynthesizing response with {model}/{voice}...")
    path.parent.mkdir(parents=True, exist_ok=True)

    def _run() -> None:
        with openai_client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav",
            instructions="Speak naturally and concisely, like a helpful home assistant.",
        ) as response:
            response.stream_to_file(path)

    await asyncio.to_thread(_run)


def read_wav_int16(np: Any, path: Path) -> tuple[Any, int, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit WAV from TTS, got sample width {sample_width}")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    else:
        audio = audio.reshape(-1, 1)
    return audio, sample_rate, channels


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)
    except Exception as exc:
        logger.error("HOME_VOICE: one-shot response test failed: %s", exc, exc_info=True)
        print(f"\nOne-shot response test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
