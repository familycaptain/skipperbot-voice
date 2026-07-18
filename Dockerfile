# =============================================================================
# skipperbot-voice — Dockerfile
# =============================================================================
# Wake-word voice companion service ("Hey Skipper"). Runs on local audio
# hardware (mic + speaker) and talks to the platform over its REST API / DB.
#
# openWakeWord notes (these are load-bearing):
#   * Python 3.12 specifically.
#   * openwakeword must be >= 0.6.0 and installed with --no-deps — its full
#     dependency set pulls in tflite-runtime and conflicting pins. We provide
#     the deps we actually use (onnxruntime + scipy + scikit-learn + numpy)
#     via requirements.txt and run the ONNX inference framework.
#   * The pretrained/preprocessing models (melspectrogram, embedding_model,
#     silero_vad) are NOT shipped in the pip package and must be downloaded
#     once. We do that at build time so the image is self-contained and starts
#     offline — no first-run network fetch.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Run the ONNX framework (we deliberately don't install tflite-runtime).
    VOICE_OPENWAKEWORD_INFERENCE_FRAMEWORK=onnx

# Runtime libraries:
#   - libportaudio2        sounddevice (PortAudio) mic/speaker I/O
#   - libsndfile1          soundfile read/write (WAV captures)
#   - libgomp1             OpenMP runtime required by onnxruntime
#   - alsa-utils/libasound2 ALSA backend for PortAudio (pulled by libportaudio2)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 \
        libgomp1 \
        alsa-utils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Project deps (numpy, sounddevice, onnxruntime, scipy, scikit-learn, ...).
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 1b) OPTIONAL: in-app acoustic echo cancellation, enabled at runtime with
#     VOICE_AEC=on. Lets the eMeet run at FULL volume without feedback while keeping
#     full-duplex barge-in (aec.py). BEST-EFFORT: every line is non-fatal, so a
#     backend that won't build on this arch is simply skipped and the AEC gracefully
#     no-ops at runtime (voice unchanged). webrtc = best quality; speexdsp = reliable
#     fallback. If a line errors, send the build output and we'll adjust the package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libspeexdsp-dev libwebrtc-audio-processing-dev \
        ; rm -rf /var/lib/apt/lists/* ; true
RUN pip install speexdsp || echo "AEC: speexdsp binding not installed (speex echo cancellation off)"
RUN pip install webrtc-audio-processing || echo "AEC: webrtc binding not installed (falls back to speexdsp)"

# 2) openWakeWord WITHOUT its dependencies (see notes above).
RUN pip install --no-deps "openwakeword>=0.6.0"

# 3) Pre-download the pretrained/preprocessing models into the package dir so
#    the service never has to fetch them at start (and works offline).
RUN python -c "from openwakeword.utils import download_models; download_models()"

# 4) Application source (incl. the custom wake_words/Hey_Skipper*.onnx).
COPY . .

# Default to the always-on wake service with the openWakeWord backend (the
# code defaults to openwakeword and auto-selects the Hey_Skipper model from
# wake_words/). Override the command / env (.env) to change behavior.
CMD ["python", "wake_voice_service.py", "--wake-backend", "openwakeword", "--openwakeword-inference-framework", "onnx"]
