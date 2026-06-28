# skipperbot-voice

Skipperbot **companion service** — the wake-word voice assistant ("Hey Skipper").
Runs as its own process on local audio hardware (mic + speaker, e.g. an EMEET
speakerphone or a Raspberry Pi satellite). The platform does not manage this
service — if it isn't running, the platform is unaffected.

**Thin client.** This service does only audio capture/playback, wake-word
detection, and OpenAI Realtime streaming. It talks to the platform over HTTP/WS
(`POST /api/voice/session` to mint an ephemeral Realtime token + session config,
`WS /ws/voice/{id}` to relay tool calls and transcripts, `POST /api/voice/end`).
The brain, tools (MCP), and database all live on the platform — so a satellite
Pi needs **no database connection, no OpenAI key, and no platform code**, only
`SKIPPER_API_BASE` pointing at the platform. This is what makes it safe to run
on a separate Pi from the one hosting the Skipper docker stack + Postgres.

> **Status:** extracted from the monolith's `home_voice/` and refactored into a
> thin client (`skipper_voice_client.py`). Built but not yet hardware-tested
> end-to-end. Copy `.env.example` → `.env` to configure.
>
> Some Phase-1 setup notes below were written when these files lived in the
> monolith's `home_voice/` subdirectory — drop the `home_voice/` prefix from
> those commands (e.g. `python wake_voice_service.py`).

---

## Run with Docker (recommended)

```bash
# 1. Config
cp .env.example .env       # set SKIPPER_API_BASE + SKIPPERBOT_TOKEN (+ devices). No OpenAI key — the platform mints the realtime token.

# 2. Mint a service token ON THE PLATFORM HOST (the platform enforces auth, so
#    /api/voice/* returns 401 without it). Native:
#      python scripts/service_token.py create voice
#    Docker platform:
#      docker compose exec agent python scripts/service_token.py create voice
#    Copy the printed st_... value into .env as SKIPPERBOT_TOKEN.

# 3. Start
docker compose up --build
```

> **Auth is required.** If the wake word is detected and a tone plays but the log
> shows `POST /api/voice/session -> HTTP 401`, the `SKIPPERBOT_TOKEN` is missing
> or wrong — mint one as in step 2 and set it in `.env`.

The image pins **Python 3.12**, installs the deps in `requirements.txt`, then
installs **openWakeWord ≥ 0.6.0 with `--no-deps`** (its full dependency set
pulls in tflite-runtime and conflicting pins; we run the ONNX framework
instead), and finally **pre-downloads the pretrained models** at build time
(`from openwakeword.utils import download_models; download_models()`) so the
container starts offline with no first-run fetch.

Audio uses ALSA device passthrough (`/dev/snd`) — works on Linux / Raspberry
Pi; the container joins the host `audio` group. For a PulseAudio host, see the
commented block in `docker-compose.yml`. `network_mode: host` lets the service
reach the platform API/DB; point `SKIPPER_API_BASE` at the platform if it runs
on another host.

To smoke-test with a bundled label before training "Hey Skipper":

```bash
docker compose run --rm voice python wake_voice_service.py \
  --wake-backend openwakeword --openwakeword-label hey_jarvis \
  --openwakeword-inference-framework onnx --self-test-wake
```

## Hardware / setup notes (Phase 1)

Early hardware-bringup tests. They were written against an **EMEET Conference
Speakerphone M0 Plus** — a good, inexpensive reference device if you're buying
something — but any USB mic + speaker works. Point the commands at your hardware
with `--device-name` / `VOICE_DEVICE_NAME` (the default match is `EMEET`).

Install the test dependencies:

```powershell
pip install -r home_voice/requirements.txt
pip install --no-deps openwakeword>=0.6.0
```

### OpenWakeWord model download (required, one-time)

The OpenWakeWord pretrained / preprocessing models are **not** bundled with the
pip package and **must** be downloaded before `wake_voice_service.py` can
start. Without them the service will fail to load `melspectrogram`,
`embedding_model`, and `silero_vad`.

From a venv where `openwakeword` and the deps in `requirements.txt` are
installed, run:

```bash
python -c "from openwakeword.utils import download_models; download_models()"
```

This downloads `melspectrogram`, `embedding_model`, `silero_vad`, and the
bundled wake-word labels (`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`,
`timer`, `weather`) into the `openwakeword` package directory in both
`.tflite` and `.onnx` formats. You only need to do this once per venv.

The download step requires `onnxruntime`, `scipy`, and `scikit-learn` — these
are listed in `requirements.txt`, so `pip install -r home_voice/requirements.txt`
followed by the `--no-deps openwakeword` install above is enough.

On Linux, some USB speakerphones (the EMEET among them) do not expose a usable
PortAudio *output* device even when the microphone enumerates fine. The output
defaults to the system default (`"default"`); set `VOICE_DEVICE_OUTPUT_NAME` or
pass `--output-name=<your device>` (e.g. `--output-name=EMEET`) to force a
specific speaker (typically needed on Windows).

List all audio devices:

```powershell
python home_voice/basic_audio_test.py --mode list
```

Run the basic audio test:

```powershell
python home_voice/basic_audio_test.py
```

That default test will:

- list audio devices
- find input and output devices matching the configured device name (default
  `EMEET`; set `--device-name` / `VOICE_DEVICE_NAME` for yours)
- play a short tone through the speaker
- record a short sample from the microphone
- play that sample back through the speaker
- save the sample to `home_voice/last_recording.wav`

Optional guarded loopback test:

```powershell
python home_voice/basic_audio_test.py --mode list --loopback-seconds 10
```

Start with low speaker volume before loopback testing to avoid feedback.

Build the shared home voice config and record a one-shot utterance:

```powershell
python home_voice/one_shot_voice_test.py
```

That script uses the same backend voice prompt/tool builder as the Android
voice path, with `platform=home_voice` and room/device context added.

Run the one-shot Skipper response test:

```powershell
python home_voice/one_shot_response_test.py
```

That records one utterance, transcribes it, sends the text through Skipper's
normal chat/tool brain with home voice context, synthesizes Skipper's response,
and plays the response through the speaker.

This chained STT -> text -> TTS test is only a Phase 1 stepping stone. The final
home voice path should use the same two-way OpenAI Realtime streaming pattern as
the Android app, with the backend providing compact voice instructions and
switchable app/tool guides.

Run the realtime prototype:

```powershell
python home_voice/realtime_voice_test.py
```

This opens a realtime session, streams mic PCM audio to OpenAI, plays assistant
audio deltas through the speaker, and routes voice tool calls through Skipper's
shared voice runtime. Press `Ctrl+C` to stop.

Run the always-on wake-word service with OpenWakeWord:

```powershell
python home_voice/wake_voice_service.py --wake-backend openwakeword
```

If `VOICE_OPENWAKEWORD_MODEL_PATHS` is not set, the service auto-selects a
stable `hey-skipper.onnx` / `hey_skipper.onnx` file or the newest timestamped
`Hey_Skipper*.onnx` export from `home_voice\wake_words`.

OpenWakeWord avoids Picovoice per-device activation limits and is the preferred
wake backend for the voice satellite (e.g. an EMEET speakerphone or a Raspberry
Pi). For quick testing before training "Hey Skipper", download OpenWakeWord's
pretrained models, initialize the wake listener, and try a bundled label such as
`hey_jarvis`:

```powershell
python home_voice/wake_voice_service.py --wake-backend openwakeword --openwakeword-download-models --openwakeword-label hey_jarvis --openwakeword-inference-framework onnx --self-test-wake
```

Then run the full wake -> realtime conversation loop with the bundled model:

```powershell
python home_voice/wake_voice_service.py --wake-backend openwakeword --openwakeword-label hey_jarvis --openwakeword-inference-framework onnx
```

False-wake safety timeouts are enabled by default:

- `VOICE_INITIAL_SPEECH_TIMEOUT_SECONDS=20` returns to wake mode if no user transcript arrives after a wake.
- `VOICE_IDLE_TIMEOUT_SECONDS=45` returns to wake mode after a real conversation goes quiet.
- `VOICE_MAX_SESSION_SECONDS=300` is a hard cap so a session cannot stream all day.

Set any value to `0` to disable that specific timeout, or pass the matching
CLI option, e.g.:

```powershell
python home_voice/wake_voice_service.py --initial-speech-timeout-seconds 15 --idle-timeout-seconds 60 --max-session-seconds 300
```

Experimental session speaker lock:

```powershell
pip install -r home_voice/requirements-speaker-isolation.txt
python home_voice/wake_voice_service.py --wake-backend openwakeword --speaker-isolation optional
```

With speaker isolation enabled, the first usable speech segment after the wake
word enrolls the active session speaker. Later speech segments are buffered
locally and forwarded to Realtime only if their speaker embedding is similar
enough to that first speaker. This is intended to ignore loud background
speakers during a conversation. It is experimental and tunable:

```powershell
python home_voice/wake_voice_service.py --speaker-isolation optional --speaker-similarity 0.68 --speaker-rms 0.015
```

Use `--speaker-isolation required` only when you want startup to fail if the
speaker-lock dependency is unavailable. The default is `off`.

Porcupine is still supported for comparison or for a future multi-device
Picovoice plan:

```powershell
python home_voice/wake_voice_service.py --wake-backend porcupine --keyword-path C:\path\to\hey-skipper_windows.ppn
```
