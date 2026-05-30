from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger("skipperbot_voice")


@dataclass
class SpeakerIsolationConfig:
    mode: str = "off"
    similarity_threshold: float = 0.68
    speech_rms_threshold: float = 0.015
    prefix_ms: int = 250
    silence_ms: int = 700
    enroll_min_ms: int = 900
    verify_min_ms: int = 450
    max_segment_ms: int = 12000


class SpeakerIsolationGate:
    """Local VAD + speaker verification gate for home voice audio.

    The first usable post-wake speech segment enrolls the session speaker.
    Later speech segments are forwarded only when their embedding is similar
    enough to the enrolled speaker. If disabled, or if optional dependencies are
    unavailable in optional mode, audio passes through unchanged.
    """

    def __init__(
        self,
        np: Any,
        *,
        sample_rate: int,
        frame_ms: int,
        config: SpeakerIsolationConfig,
    ) -> None:
        self.np = np
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.config = config
        self.mode = (config.mode or "off").strip().lower()
        self.active = self.mode in {"optional", "required", "on", "true", "1"}
        self.required = self.mode == "required"
        self.encoder = None
        self.preprocess_wav = None
        self.enrolled_embedding = None
        self.enrolled = False
        self.speaking = False
        self.silence_ms = 0
        self.segment: list[bytes] = []
        self.prefix: deque[bytes] = deque(maxlen=max(1, config.prefix_ms // max(1, frame_ms)))
        self.accepted_segments = 0
        self.rejected_segments = 0

        if self.active:
            self._load_backend()

    def _load_backend(self) -> None:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav

            self.encoder = VoiceEncoder()
            self.preprocess_wav = preprocess_wav
            logger.info("HOME_VOICE: speaker isolation enabled with resemblyzer")
        except Exception as exc:
            if self.required:
                raise RuntimeError(
                    "Speaker isolation requires resemblyzer. Install optional home voice "
                    "speaker-isolation dependencies or use --speaker-isolation off/optional."
                ) from exc
            logger.warning(
                "HOME_VOICE: speaker isolation disabled; resemblyzer unavailable: %s",
                exc,
            )
            self.active = False

    def process(self, pcm16_mono: bytes) -> list[bytes]:
        if not self.active:
            return [pcm16_mono]

        is_speech = self._is_speech(pcm16_mono)
        accepted: list[bytes] = []

        if not self.speaking:
            self.prefix.append(pcm16_mono)
            if is_speech:
                self.speaking = True
                self.silence_ms = 0
                self.segment = list(self.prefix)
            return accepted

        self.segment.append(pcm16_mono)
        if is_speech:
            self.silence_ms = 0
        else:
            self.silence_ms += self.frame_ms

        if self._segment_ms() >= self.config.max_segment_ms:
            accepted.extend(self._finish_segment(reason="max_segment"))
        elif self.silence_ms >= self.config.silence_ms:
            accepted.extend(self._finish_segment(reason="silence"))

        return accepted

    def flush(self) -> list[bytes]:
        if self.active and self.speaking and self.segment:
            return self._finish_segment(reason="flush")
        return []

    def _is_speech(self, pcm16_mono: bytes) -> bool:
        audio = self.np.frombuffer(pcm16_mono, dtype=self.np.int16)
        if audio.size == 0:
            return False
        normalized = audio.astype(self.np.float32) / 32768.0
        rms = float(self.np.sqrt(self.np.mean(normalized * normalized)))
        return rms >= self.config.speech_rms_threshold

    def _segment_ms(self) -> float:
        samples = sum(len(frame) for frame in self.segment) / 2
        return samples / self.sample_rate * 1000.0

    def _finish_segment(self, *, reason: str) -> list[bytes]:
        segment = b"".join(self.segment)
        segment_ms = self._segment_ms()
        self.segment = []
        self.speaking = False
        self.silence_ms = 0
        self.prefix.clear()

        if not segment:
            return []

        if not self.enrolled:
            if segment_ms >= self.config.enroll_min_ms:
                embedding = self._embed(segment)
                if embedding is not None:
                    self.enrolled_embedding = embedding
                    self.enrolled = True
                    self.accepted_segments += 1
                    logger.info(
                        "HOME_VOICE: speaker isolation enrolled session speaker "
                        "(%.0f ms, reason=%s)",
                        segment_ms,
                        reason,
                    )
                else:
                    logger.warning("HOME_VOICE: could not enroll session speaker")
            return [segment]

        if segment_ms < self.config.verify_min_ms:
            self.rejected_segments += 1
            logger.info(
                "HOME_VOICE: speaker isolation dropped short segment (%.0f ms)",
                segment_ms,
            )
            return []

        embedding = self._embed(segment)
        if embedding is None:
            self.rejected_segments += 1
            logger.info("HOME_VOICE: speaker isolation dropped segment without embedding")
            return []

        similarity = float(self.np.dot(self.enrolled_embedding, embedding))
        if similarity >= self.config.similarity_threshold:
            self.accepted_segments += 1
            logger.info(
                "HOME_VOICE: speaker isolation accepted segment similarity=%.3f",
                similarity,
            )
            return [segment]

        self.rejected_segments += 1
        logger.info(
            "HOME_VOICE: speaker isolation rejected segment similarity=%.3f threshold=%.3f",
            similarity,
            self.config.similarity_threshold,
        )
        return []

    def _embed(self, segment: bytes):
        if not self.encoder or not self.preprocess_wav:
            return None
        try:
            audio = self.np.frombuffer(segment, dtype=self.np.int16).astype(self.np.float32) / 32768.0
            wav = self.preprocess_wav(audio, source_sr=self.sample_rate)
            embedding = self.encoder.embed_utterance(wav)
            norm = self.np.linalg.norm(embedding)
            if not norm:
                return None
            return embedding / norm
        except Exception as exc:
            logger.warning("HOME_VOICE: speaker embedding failed: %s", exc)
            return None
