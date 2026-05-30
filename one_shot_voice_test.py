from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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
from voice_prompting import build_base_voice_payload  # noqa: E402


DEFAULT_ONE_SHOT_PATH = Path(__file__).with_name("one_shot_input.wav")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Home voice one-shot starter test. This records from EMEET and builds "
            "the same voice prompt/tool config used by the Android voice path."
        )
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("VOICE_USER_ID", "user1"),
        help="Skipper user id/name for prompt and tool context. Default: VOICE_USER_ID or user1.",
    )
    parser.add_argument(
        "--device-id",
        default=os.getenv("VOICE_DEVICE_ID", "windows-server-local-emeet"),
        help="Home voice device id.",
    )
    parser.add_argument(
        "--room",
        default=os.getenv("VOICE_ROOM", "office"),
        help="Room context for ambiguous home commands.",
    )
    parser.add_argument(
        "--friendly-name",
        default=os.getenv("VOICE_FRIENDLY_NAME", "Office Speaker"),
        help="Human-friendly device name.",
    )
    parser.add_argument(
        "--device-name",
        default=DEFAULT_DEVICE_NAME,
        help="Fallback partial name for both input and output devices. Default: EMEET.",
    )
    parser.add_argument(
        "--input-name",
        default=None,
        help="Partial input device name. Defaults to VOICE_DEVICE_INPUT_NAME or --device-name.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Partial output device name. Defaults to VOICE_DEVICE_OUTPUT_NAME or --device-name.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Preferred mic sample rate. The script falls back if needed.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Preferred mic channels. Default: 1.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=5.0,
        help="Seconds to record. Default: 5.",
    )
    parser.add_argument(
        "--save-wav",
        type=Path,
        default=DEFAULT_ONE_SHOT_PATH,
        help=f"Path for captured WAV. Default: {DEFAULT_ONE_SHOT_PATH}.",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Play the captured utterance back through the EMEET speaker.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only build and print the shared prompt/tool config summary.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device_info = {
        "platform": "home_voice",
        "device_type": "home_voice",
        "device_id": args.device_id,
        "room": args.room,
        "friendly_name": args.friendly_name,
        "audio_device_name": args.device_name,
    }

    payload = build_base_voice_payload(user_id=args.user_id, device_info=device_info)
    print("Shared voice config:")
    print(f"  user_id:            {args.user_id}")
    print(f"  platform:           {device_info['platform']}")
    print(f"  room:               {args.room}")
    print(f"  active_app:         {payload.get('app')}")
    print(f"  active_category:    {payload.get('category')}")
    print(f"  default_categories: {', '.join(payload.get('default_categories', []))}")
    print(f"  tools:              {len(payload.get('tools', []))}")
    print(f"  instructions chars: {len(payload.get('instructions', ''))}")

    if args.config_only:
        return 0

    np, sd = load_audio_deps()
    list_audio_devices(sd)

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
    print(f"  input:  {input_sample_rate} Hz, {input_channels} channel(s)")
    print(f"  output: {output_sample_rate} Hz, {output_channels} channel(s)")

    recording = record_audio(
        np,
        sd,
        input_device=input_device,
        sample_rate=input_sample_rate,
        channels=input_channels,
        seconds=args.record_seconds,
    )
    save_wav(args.save_wav, recording, sample_rate=input_sample_rate, channels=input_channels)

    if args.playback:
        play_audio(
            np,
            sd,
            recording,
            output_device=output_device,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_channels=output_channels,
        )

    print("\nOne-shot starter complete.")
    print(
        "Next wiring step: send this WAV into the realtime voice transport using "
        "the shared instructions/tools printed above."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)
