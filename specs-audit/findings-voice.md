# Findings — voice (the speaker device)

Survey only; nothing fixed. Corpus 0 → 64 records (written from scratch — no `specs/`, no manifest, no
`CLAUDE.md`). No test suite exists, so every spec carries `tests: []`. Items marked **VERIFIED** were
confirmed independently by the PM.

Deliberately not duplicated: `platform/voice`, `platform/speech`, `platform/speaking`. This corpus states
what the *device* does and what it may assert, and defers attribution and surface routing to those.

## Listening when it should not, and what is kept

**F1. VERIFIED — the "nobody spoke after the wake" timeout can be postponed indefinitely by room noise, in
the default mode.** `host_relay_client.py:59-60::waiting_for_first_user_transcript_seconds` returns
`time.monotonic() - self._last_activity`, and `_last_activity` is bumped by **every inbound frame** — host
`status` events fired by voice-activity detection, and every audio frame. The direct client
(`realtime_voice_test.py:328`) does it correctly: it checks whether a transcript arrived and otherwise
measures from `started_at`. The comment above the relay version claims "Interface mirrored from the direct
client"; the semantics differ.

Consequence: **a false wake in a room with a television streams the room to the platform up to the 300 s
hard cap rather than stopping at the 20 s no-speech cap.** The platform's 30 s substantive-turn watchdog
(`app_platform/voice/autoend.py`) is what actually saves this today; in `--mode direct` there is no such
backstop. Expected: measure from session start.

**F2. VERIFIED — verbatim transcripts of every voice exchange are printed to stdout and retained
indefinitely by Docker.** `host_relay_client.py::_on_message` prints `You: …` / `Skipper: …` and tool calls
with parameters; `realtime_voice_test.py::_on_message` does the same; `wake_voice_service.py::_on_announce`
prints announcement text. `docker-compose.yml` sets `restart: unless-stopped` and **no `logging:` driver or
limits at all**, so the default json-file driver accumulates them without bound on the device.

This is the one durable, on-disk record of speech at a speaker the operator deliberately excluded from the
per-person console record (`platform.speaking.voice-is-nobodys-personal-surface`). Expected: no transcript
text at default verbosity, or an explicit rotation cap in compose.

**F3. VERIFIED — the device sends a hard-coded default identity, and the platform uses it for tools, data
and permissions.** `wake_voice_service.py:395` defaults `--user-id` / `VOICE_USER_ID` to the literal
`"user1"`. `skipper_voice_client.py::create_session` sends it to `POST /api/voice/session`, where
`mint_ephemeral_token` stores it as `session["user_id"]` and `relay.py` **runs the tools as that user**
until speaker-ID resolves a name. Speaker-ID is opt-in (`platform.voice.speaker-id-optin`), so on a default
install it never resolves — meaning everything said at the shared speaker runs with whatever household
member `user1` happens to be.

**This directly contradicts the operator's rule that the shared speaker is nobody's personal surface.**
Specced as intent in `voice.privacy.the-device-never-decides-who-is-speaking`; the code contradicts the
spec. Expected: no default identity, and personal data gated on actual attribution. (Compare the platform
finding that `POST /api/voice/session` trusts the body's `user_id` — the two compound.)

**F4. `basic_audio_test.py::save_wav` leaves a recording of the room on the device.** `--save-wav` defaults
to `last_recording.wav` beside the source with no expiry. Only reachable by hand-running the test tool, and
`.gitignore`/`.dockerignore` exclude `*.wav` so it cannot be committed or baked into an image — but it is
the only path that puts room audio on disk. Low severity.

## A device that appears off and is not, or gives no sign at all

**F5. An HTTP failure minting a session kills the whole service instead of returning to wake listening.**
`skipper_voice_client.py::create_session` returns `None` only for a 200 carrying an `error` key.
`urllib.request.urlopen` raises `HTTPError`/`URLError` for a 401 (missing or wrong `SKIPPERBOT_TOKEN`), a
refused connection or a timeout, and nothing catches it — it escapes `main()` and the process exits 1. The
graceful branch ("Could not create realtime voice session; returning to wake listening.") is therefore
**unreachable for the most likely failures**, and under `restart: unless-stopped` the container
crash-loops silently. `README.md` §Auth describes the opposite. Expected: catch transport and HTTP errors.

**F6. No audible difference between "did not hear you" and "cannot reach Skipper".** The wake chime
deliberately fires only once the audio bridge is live (correct, and specced), so a platform outage produces
**no sound at all** from the room. Recorded as `voice.platform-link.an-outage-is-silent-from-the-room`;
expected behaviour would be a distinct failure earcon, since this is the one failure a person could act on.

**F7. Unplugging the speakerphone turns the device into a silent crash-loop.**
`basic_audio_test.py::find_device` raises when the configured name substring matches nothing, killing the
service; `restart: unless-stopped` then restarts it forever. Failing rather than listening on the wrong
microphone is right; the unbounded restart with no local signal is not.

**F8. Microphone frames are dropped silently when the wake queue backs up.**
`OpenWakeWordWakeListener._input_callback` does `except queue.Full: pass` on a `maxsize=50` queue (≈4 s at
the 80 ms default frame). A stalled `model.predict` on a loaded Pi can miss the wake word with nothing
logged above debug.

**F9. The wake model is re-loaded from disk on every wake cycle.** `wake_voice_service.py::main` calls
`create_wake_listener(...)` *inside* the `while` loop, so the openWakeWord `Model` (or the whole Porcupine
engine) is reconstructed after every conversation — latency on each wake, and repeated model file reads.

## Barge-in and echo cancellation

**F10. `VOICE_SUPPRESS_MIC_DURING_PLAYBACK` only mutes the mic for 700 ms, not for the reply.**
`realtime_voice_test.py::RealtimeAudioBridge._input_callback` gates on
`is_playback_barge_in_grace_active()` — the first `VOICE_PLAYBACK_BARGE_IN_GRACE_MS` (700 ms) of a
response. `.env.example` documents the flag as "you'll lose barge-in while Skipper talks", i.e. muted for
the duration. `is_output_active()` — the method that implements the documented semantics, with
`_playback_tail_seconds` — is **dead code**, referenced nowhere. So hardware with no echo cancellation
still hears itself for all but the first 700 ms, and **the documented remedy does not work.**

**F11. Relay mode never clears the local output buffer on barge-in.**
`RealtimeHomeVoiceClient._on_message` calls `audio_bridge.clear_output_audio()` on `speech_started`;
`host_relay_client.py::RelayClient._on_message` handles the host's `status: speech_started` by printing
"Listening..." and nothing else. So in the default mode, interrupting Skipper lets whatever PCM is already
buffered on the device finish playing. `voice.hearing.interrupting-stops-the-answer` is written to the
host-side truncation (which does hold — no *further* audio arrives), but the interruption is measurably
less clean than in the legacy mode.

**F12. The speexdsp echo canceller can lose near/far alignment permanently.** `aec.py::_SpeexAEC` pairs one
far-end frame with each captured frame from a FIFO list and, on overflow past 50 frames, discards from the
**front** (`self._ref_frames.pop(0)`) — shifting alignment rather than resynchronising. speexdsp has no
adaptive delay estimation (only webrtc does, and `aec.py` notes there is no working py3.12 binding), so one
overrun can leave echo cancellation misaligned for the rest of the session, presenting as echo returning
mid-conversation at high volume. *Uncertain how often it fires.*

**F13. `aec.py::_install_imp_shim` installs a fake `imp` module into `sys.modules` process-wide** to satisfy
speexdsp's SWIG loader on Python 3.12. Deliberate and minimal, but it changes global import behaviour for
everything else in the process.

## Announcements

**F14. An announcement arriving mid-conversation is thrown away, and the platform records it as spoken.**
`wake_voice_service.py::_on_announce` prints "Announcement received during a conversation — skipping." and
returns; `_on_announce_end` drops the buffered PCM. But
`app_platform/voice/announce.py::announce_to_device` returns `True` as soon as the envelope and frames are
*sent*, and `apps/notifications/delivery.py` treats `True` as spoken — so it does **not** fall back to the
push channels and writes a "spoke" receipt. **Net: the message is lost silently, with a receipt saying it
was delivered.** Expected: the device acknowledges or refuses, or the host does not count a send as speech.

**F15. No interlock in the other direction: a wake during announcement playback fights for the speaker.**
`in_session` is set only when a conversation starts, so nothing stops `wait_for_detection` firing and
`RealtimeAudioBridge.start()` opening the ALSA output device while `sd.play(...)`/`sd.wait()` on the
device-link thread still holds it — the exact conflict `in_session` exists to prevent, approached from the
other side. Relatedly, the `_ann` dict is mutated from both the socket thread and the main thread with no
lock.

**F16. Announcement playback blocks the device-link socket thread and can drop the link.**
`_on_announce_end` calls `sd.play(...); sd.wait()` on the callback thread, against `DeviceLink`'s own
docstring ("keep them quick"). The link runs `run_forever(ping_interval=20, ping_timeout=10)`, so an
announcement longer than ~30 s can starve the keepalive and cause a disconnect/reconnect mid-announcement.

**F17. `priority` and `listen_after` from the announce envelope are ignored entirely.** The platform sends
both; nothing on the device reads either. `listen_after` is documented platform-side as "Group 2 opens the
mic for a reply". One-way-ness is specced as intent
(`voice.announcements.announcements-are-one-way`) — correctly, for a device that must not start recording
because the house decided to talk — but **the platform field is a promise nothing keeps.**

**F18. (Cross-repo.) A named room can never be reached.** `apps/notifications/delivery.py` reads
`notif.get("device_id")` for the voice channel and nothing ever writes it, so
`app_platform/voice/devices.py::resolve("")` always takes the unnamed branch: the single online device, or
**nothing at all when two or more are online.** The device side is already in place — it registers a stable
`VOICE_DEVICE_ID` and `room` on the standing link — so the gap is entirely on the platform.

## Protocol drift between the device and the platform

**F19. The device silently discards two frame types the host sends it.**
`host_relay_client.py::RelayClient._on_message` handles `transcript`, `status`, `tool_call`, `host_info`,
`session_ended`, `error` — and ignores `speaker` (sent by `relay.py::_set_locked_identity`: who the
platform decided is talking) and `transcript_partial` (sent expressly so the satellite can show words as
they are heard). So the live partial-transcript feature is inert at its only consumer, and the resolved
speaker is never shown on the device.

**F20. The satellite half of the unified voice debug stream does not exist.**
`app_platform/voice/debug_log.py`'s docstring states "the satellite POSTs its own events to
/api/voice/debug", and the platform serves that endpoint for exactly this. Nothing in this repo ever calls
it, so the endpoint's device-side producer is missing and the stream only ever shows the platform's half.

**F21. `RelayClient.request_end` is dead code.** Nothing calls it, so the relay protocol's
`{"type": "end"}` control frame is never sent. Conversations end by closing the socket plus
`POST /api/voice/end`, and `relay.py::pump_satellite_to_openai` handles an `end` it will never receive.

**F22. Relay mode receives an unused third-party credential on every wake.**
`POST /api/voice/session` always mints an ephemeral OpenAI Realtime token and returns it in the session
config; in relay mode `RelayClient` connects only to the platform and never uses it. **So the default
configuration ships a live outside credential to the speaker box on every single wake for no reason** —
against this repo's stated "no OpenAI key on the device". Expected: mint it only for `--mode direct`.

**F23. Two dead client functions, one documented as a working path.**
`skipper_voice_client.py::get_picovoice_config` and `::switch_app` are never called anywhere.
`.env.example` tells the operator the Porcupine key can be served by "GET /api/config/picovoice"; the wake
service only reads `PICOVOICE_API_KEY` from the environment, so that documented path does not work.

**F24. The audio relay socket has no keepalive.** `RelayClient.run` calls `run_forever()` with no
`ping_interval`/`ping_timeout`, unlike `DeviceLink` and `RealtimeHomeVoiceClient` (both 20/10). A
wedged-but-open socket would not be detected by keepalive; only the timeout loop would end the session.

## Dead code, stale configuration, stale docs

**F25. An event-loop thread is started and stopped for nothing.** `wake_voice_service.py` imports
`run_async` and calls `initialize_async_loop()` / `shutdown_async_loop()`, but nothing schedules work on
that loop after the thin-client refactor — `run_async` is never called anywhere.

**F26. Both entry points prepend the *parent of the repo* to `sys.path`**
(`wake_voice_service.py:17-19`, `realtime_voice_test.py:18-20`) — a leftover from when these lived in the
platform's `home_voice/` subdirectory. In the Docker image (`WORKDIR /app`) that inserts `/` at the head of
the import path, so any stray `.py` at the filesystem root would shadow a real module.

**F27. `VOICE_WAKE_BACKEND` has no effect under Docker.** The Dockerfile `CMD` hard-codes
`--wake-backend openwakeword --openwakeword-inference-framework onnx`, and CLI arguments beat env defaults
— so setting `VOICE_WAKE_BACKEND=porcupine` in `.env`, the documented way to configure the service, **does
nothing.** The Porcupine path is only reachable by overriding the whole command.

**F28. `--openwakeword-threshold` help text says "Default: … or 0.85"** while
`DEFAULT_OPENWAKEWORD_THRESHOLD` is `0.90` and `.env.example` says 0.90.

**F29. `docker-compose.yml`'s header still instructs the operator to set `SKIPPERBOT_DB_DSN` and an "OpenAI
voice key" in `.env`.** The thin client needs neither and `.env.example` correctly says so — actively
misleading in the one repo whose selling point is that no keys and no database live on the device.

**F30. `README.md` documents two scripts that do not exist** — a whole Phase-1 section covers
`home_voice/one_shot_voice_test.py` and `home_voice/one_shot_response_test.py`, neither of which is in this
repo. Its "Status" block also still says "Built but not yet hardware-tested end-to-end", contradicted by
the AEC-tuning, volume-pinning and `redeploy.sh` commit history.

**F31. `requirements.txt` pins `pvporcupine>=3.0.0` for every install** even though openWakeWord is the
default and, per F27, the only backend the image can reach.

**F32. Two speaker-lock knobs are source-only.** `SpeakerIsolationConfig.prefix_ms` and `.max_segment_ms`
have no CLI flag or environment variable, unlike every other field — so exactly the two settings that
decide how much speech is buffered and how long a segment may grow can only be changed by editing the file.

**F33. A speaker lock that is on can silently never engage.** In
`SpeakerIsolationGate._finish_segment`, when `not self.enrolled` and `segment_ms < enroll_min_ms`, the
branch falls through to `return [segment]` with `enrolled` still `False`. A speaker whose first utterances
are all short therefore never enrols, every segment is forwarded, and the gate behaves exactly as `off`
**while the service prints "speaker lock: on".**

**F34. `wake_words/README.md` is stale** — it gives the preferred model path as
`home_voice/wake_words/hey-skipper.onnx` (a directory level that no longer exists) and describes the host
as "this Windows server".
