"""In-app acoustic echo cancellation (AEC) for full-duplex voice.

The eMeet's hardware AEC leaks Skipper's own voice back into the mic at high
volume, which the server VAD then treats as phantom user turns. This cancels the
playback out of the mic IN SOFTWARE, on the Pi, using the LOCAL played audio as
the far-end reference — so the speaker can run at FULL volume while the mic stays
clean AND barge-in still works (we never mute the mic).

Why network latency is irrelevant: both signals are local to the Pi — the mic
(near-end) and the exact frames going to the speaker (far-end). The response
audio travelled over the network to *arrive*, but by the time it's played it's a
local signal, and the mic hears its echo a few ms later. The only delay to align
is the local acoustic + audio-buffer path, which the canceller handles (webrtc
does adaptive delay estimation; speexdsp absorbs it in the filter tail).

Backends, best first:
  1. webrtc APM  (`webrtc_audio_processing`) — adaptive delay estimation, best quality.
  2. speexdsp    (`speexdsp`)                — reliable, needs a longer filter tail.
If neither imports, create() returns None and voice runs UNCHANGED (AEC off). The
whole feature is opt-in via VOICE_AEC=on, so it can never break a working setup.

Everything here runs at ONE rate (the mic's), in 10 ms frames, mono int16.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger("skipperbot_voice.aec")

_FRAME_MS = 10  # webrtc requires 10ms frames; speexdsp uses the same for parity


def enabled() -> bool:
    # On by default now that it's proven; harmless where unneeded (graceful no-op if
    # no backend is installed). Set VOICE_AEC=off to disable.
    return os.getenv("VOICE_AEC", "on").strip().lower() in ("1", "true", "on", "yes")


class _Base:
    """Common interface. Feed the far-end (played) frames via add_reference() as
    they go to the speaker, and run each mic block through process(). Both take
    arbitrary-length mono int16 bytes at `rate`; framing to 10ms is internal."""

    name = "none"

    def __init__(self, rate: int):
        self.rate = rate
        self.frame_bytes = int(rate * _FRAME_MS / 1000) * 2  # int16 mono
        self._ref = bytearray()
        self._ref_lock = threading.Lock()   # add_reference (output thread) vs process (input thread)
        self._near = bytearray()
        self._max_ref = self.frame_bytes * (1000 // _FRAME_MS)  # cap the ref backlog at ~1s

    # subclasses implement these on exact 10ms frames — ALWAYS called from the
    # input-callback thread (process), so the underlying APM is single-threaded.
    def _reverse_frame(self, frame: bytes) -> None: ...
    def _capture_frame(self, frame: bytes) -> bytes: ...

    def add_reference(self, pcm16: bytes) -> None:
        """Called from the OUTPUT callback thread — buffer only, no APM work here."""
        with self._ref_lock:
            self._ref.extend(pcm16)
            if len(self._ref) > self._max_ref:
                del self._ref[:len(self._ref) - self._max_ref]

    def process(self, pcm16: bytes) -> bytes:
        """Called from the INPUT callback thread — does ALL APM work (reverse then
        capture), so the backend is only ever touched by one thread."""
        # 1) feed the far-end (played) frames buffered since last call
        while True:
            with self._ref_lock:
                if len(self._ref) < self.frame_bytes:
                    break
                frame = bytes(self._ref[:self.frame_bytes])
                del self._ref[:self.frame_bytes]
            try:
                self._reverse_frame(frame)
            except Exception as exc:  # noqa: BLE001 — never break audio on an AEC hiccup
                logger.debug("AEC reverse frame failed: %s", exc)
        # 2) cancel echo from the near-end (mic) frames
        self._near.extend(pcm16)
        out = bytearray()
        while len(self._near) >= self.frame_bytes:
            frame = bytes(self._near[:self.frame_bytes])
            del self._near[:self.frame_bytes]
            try:
                cleaned = self._capture_frame(frame)
            except Exception as exc:  # noqa: BLE001
                logger.debug("AEC capture frame failed: %s", exc)
                cleaned = frame
            out.extend(cleaned or frame)
        return bytes(out)


class _WebrtcAEC(_Base):
    name = "webrtc"

    def __init__(self, rate: int):
        super().__init__(rate)
        from webrtc_audio_processing import AudioProcessingModule as AP  # type: ignore
        ap = AP(enable_ns=True, enable_agc=False, enable_aec=True)
        ap.set_stream_format(rate, 1)
        ap.set_reverse_stream_format(rate, 1)
        try:
            ap.set_ns_level(1)
        except Exception:
            pass
        # A rough starting delay hint; webrtc's estimator refines it adaptively.
        try:
            ap.set_stream_delay_ms(int(os.getenv("VOICE_AEC_DELAY_MS", "120")))
        except Exception:
            pass
        self._ap = ap

    def _reverse_frame(self, frame: bytes) -> None:
        self._ap.process_reverse_stream(frame)

    def _capture_frame(self, frame: bytes) -> bytes:
        return self._ap.process_stream(frame)


def _install_imp_shim() -> None:
    """Python 3.12 removed the `imp` module, but speexdsp's SWIG loader still does
    `import imp`. Provide a minimal shim (find_module/load_module over importlib) so
    the compiled extension loads. No-op if `imp` already exists."""
    import sys
    if "imp" in sys.modules:
        return
    import types
    import importlib.util
    import importlib.machinery
    shim = types.ModuleType("imp")

    def find_module(name, path=None):
        spec = importlib.machinery.PathFinder().find_spec(name, path)
        if spec is None or not spec.origin:
            raise ImportError(name)
        return (open(spec.origin, "rb"), spec.origin, ("", "rb", 3))

    def load_module(name, file, pathname, desc):
        spec = importlib.util.spec_from_file_location(name, pathname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod

    shim.find_module = find_module
    shim.load_module = load_module
    shim.C_EXTENSION = 3
    sys.modules["imp"] = shim


class _SpeexAEC(_Base):
    name = "speexdsp"

    def __init__(self, rate: int):
        super().__init__(rate)
        _install_imp_shim()
        from speexdsp import EchoCanceller  # type: ignore
        frame_size = int(rate * _FRAME_MS / 1000)          # samples per 10ms frame
        # Filter tail must cover the acoustic + buffer delay. 400ms is what cleared the
        # residual echo on the reference eMeet setup at full volume (the delay is
        # dominated by the playback/resample buffering, ~100-150ms); 200ms was too short.
        tail_ms = int(os.getenv("VOICE_AEC_TAIL_MS", "400"))
        filter_length = max(frame_size * 2, int(rate * tail_ms / 1000))
        self._ec = EchoCanceller.create(frame_size, filter_length, rate)
        self._ref_frames: list[bytes] = []

    def _reverse_frame(self, frame: bytes) -> None:
        # speexdsp cancels near+far TOGETHER, so hold the far frames and pair them
        # in _capture_frame. Bound the backlog so a stall can't grow unbounded.
        self._ref_frames.append(frame)
        if len(self._ref_frames) > 50:
            self._ref_frames.pop(0)

    def _capture_frame(self, frame: bytes) -> bytes:
        ref = self._ref_frames.pop(0) if self._ref_frames else (b"\x00" * self.frame_bytes)
        return self._ec.process(frame, ref)


def create(rate: int) -> _Base | None:
    """Build the best available echo canceller for `rate`, or None if AEC is off or
    no backend is installed (voice then runs unchanged)."""
    if not enabled():
        return None
    prefer = os.getenv("VOICE_AEC_BACKEND", "").strip().lower()  # "", "webrtc", "speex"
    order = ([_WebrtcAEC, _SpeexAEC] if prefer in ("", "webrtc")
             else [_SpeexAEC, _WebrtcAEC])
    for cls in order:
        try:
            inst = cls(rate)
            # print() (not just logger.info) so it shows in the console alongside the
            # other satellite status lines, confirming AEC is actually active.
            print(f"AEC: {inst.name} echo cancellation ACTIVE @ {rate}Hz "
                  f"(tail {os.getenv('VOICE_AEC_TAIL_MS', '400')}ms)")
            logger.info("AEC: using %s backend @ %dHz", inst.name, rate)
            return inst
        except Exception as exc:  # noqa: BLE001 — EXPECTED during fallback; not an error.
            # Debug-level so a normal "webrtc not installed, using speexdsp" fallback
            # doesn't look like a failure in the console. Only a total miss is surfaced.
            logger.debug("AEC: %s backend not available (%s); trying next", cls.name, exc)
    print("AEC: no echo-cancellation backend installed — running WITHOUT AEC "
          "(this is fine unless you need full-volume full-duplex; see aec.py)")
    return None
