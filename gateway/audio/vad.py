"""Local VAD gate backed by TEN VAD.

Gateway audio is 24 kHz mono PCM16. TEN VAD expects 16 kHz frames, so this
module downsamples only for inference while forwarding the original 24 kHz
chunks to the realtime backend.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
from typing import Protocol

import av
import numpy as np

try:
    import ten_vad
except (ImportError, OSError):
    ten_vad = None  # type: ignore[assignment]

logger = logging.getLogger("gateway.vad")

_INPUT_SAMPLE_RATE = 24000


@dataclass
class VADConfig:
    """Tuning knobs for the endpointing gate.

    ``hangover_frames`` counts ~20 ms *input chunks*, not VAD frames: it is how
    much trailing silence must accumulate before an utterance is committed. The
    default (15 x 20 ms = 300 ms) is therefore a hard floor on response latency
    — the single biggest latency/accuracy trade-off in the pipeline. Lower it
    and the agent responds faster but clips slow talkers mid-sentence; raise it
    (or let background noise hold the gate open) and the agent stalls.
    ``pre_roll_ms`` likewise assumes ~20 ms input chunks.
    """

    enabled: bool = True
    threshold: float = 0.6
    vad_sample_rate: int = 16000
    vad_frame_ms: int = 32
    pre_roll_ms: int = 100
    hangover_frames: int = 15

    @classmethod
    def from_env(cls) -> VADConfig:
        return cls(
            enabled=_env_bool("VAD__ENABLED", True),
            threshold=float(os.getenv("VAD__THRESHOLD", "0.6")),
            vad_sample_rate=int(os.getenv("VAD__VAD_SAMPLE_RATE", "16000")),
            vad_frame_ms=int(os.getenv("VAD__VAD_FRAME_MS", "32")),
            pre_roll_ms=int(os.getenv("VAD__PRE_ROLL_MS", "100")),
            hangover_frames=int(os.getenv("VAD__HANGOVER_FRAMES", "15")),
        )


@dataclass
class VADResult:
    is_speech: bool
    speech_probability: float
    frames_to_flush: list[bytes]
    speech_started: bool = False
    speech_ended: bool = False


class LocalVAD(Protocol):
    def process(self, pcm16: bytes) -> VADResult:
        """Process one 24 kHz PCM16 chunk."""


class _State(Enum):
    SILENCE = auto()
    SPEECH = auto()
    HANGOVER = auto()


class TenVADGate:
    """TEN VAD plus a small onset/hangover state machine."""

    def __init__(self, config: VADConfig | None = None) -> None:
        if ten_vad is None:
            raise NotImplementedError("ten_vad is not available on this platform")

        self._config = config or VADConfig()
        self._hop_size = int(
            self._config.vad_sample_rate * self._config.vad_frame_ms / 1000)
        try:
            self._vad_model = ten_vad.TenVad(hop_size=self._hop_size)
        except Exception as exc:
            # Some ten_vad builds call __del__ on a half-constructed object.
            ten_vad.TenVad.__del__ = lambda self: None
            raise NotImplementedError(str(exc)) from exc

        self._resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self._config.vad_sample_rate,
        )
        self._pre_roll: deque[bytes] = deque(
            maxlen=math.ceil(self._config.pre_roll_ms / 20))
        self._vad_samples: list[int] = []
        self._state = _State.SILENCE
        self._hangover_count = 0
        self._last_probability = 0.0

    def process(self, pcm16: bytes) -> VADResult:
        samples_16k = self._downsample(pcm16)
        if samples_16k.size:
            self._vad_samples.extend(samples_16k.tolist())

        probability = self._last_probability
        while len(self._vad_samples) >= self._hop_size:
            frame = np.array(self._vad_samples[: self._hop_size], dtype=np.int16)
            self._vad_samples = self._vad_samples[self._hop_size:]
            probability, _ = self._vad_model.process(frame)
        self._last_probability = probability

        is_speech_frame = probability >= self._config.threshold
        frames_to_flush: list[bytes] = []
        speech_started = False
        speech_ended = False

        if self._state is _State.SILENCE:
            if is_speech_frame:
                frames_to_flush = list(self._pre_roll) + [pcm16]
                self._pre_roll.clear()
                self._state = _State.SPEECH
                speech_started = True
            else:
                self._pre_roll.append(pcm16)
        elif self._state is _State.SPEECH:
            frames_to_flush = [pcm16]
            if not is_speech_frame:
                self._state = _State.HANGOVER
                self._hangover_count = self._config.hangover_frames
        elif self._state is _State.HANGOVER:
            if is_speech_frame:
                frames_to_flush = [pcm16]
                self._state = _State.SPEECH
            elif self._hangover_count > 0:
                frames_to_flush = [pcm16]
                self._hangover_count -= 1
            else:
                self._state = _State.SILENCE
                self._pre_roll.append(pcm16)
                speech_ended = True

        return VADResult(
            is_speech=self._state in (_State.SPEECH, _State.HANGOVER),
            speech_probability=probability,
            frames_to_flush=frames_to_flush,
            speech_started=speech_started,
            speech_ended=speech_ended,
        )

    def reset(self) -> None:
        self._pre_roll.clear()
        self._vad_samples.clear()
        self._state = _State.SILENCE
        self._hangover_count = 0
        self._last_probability = 0.0

    def _downsample(self, pcm16: bytes) -> np.ndarray:
        audio = np.frombuffer(pcm16, dtype=np.int16)
        if audio.size == 0:
            return np.array([], dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(
            audio.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = _INPUT_SAMPLE_RATE
        frame.time_base = Fraction(1, _INPUT_SAMPLE_RATE)
        resampled = self._resampler.resample(frame)
        if not resampled:
            return np.array([], dtype=np.int16)
        chunks = [f.to_ndarray().flatten().astype(np.int16) for f in resampled]
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


class PassthroughVAD:
    """Fallback when local VAD is disabled or TEN VAD is unavailable."""

    def process(self, pcm16: bytes) -> VADResult:
        return VADResult(
            is_speech=False,
            speech_probability=0.0,
            frames_to_flush=[pcm16] if pcm16 else [],
        )


def build_local_vad(config: VADConfig | None = None) -> LocalVAD:
    cfg = config or VADConfig.from_env()
    if not cfg.enabled:
        logger.info("local TEN VAD disabled; forwarding audio without local gating")
        return PassthroughVAD()
    try:
        vad = TenVADGate(cfg)
    except NotImplementedError as exc:
        logger.warning(
            "local TEN VAD unavailable; forwarding audio without local gating: %s",
            exc,
        )
        return PassthroughVAD()
    logger.info(
        "local TEN VAD enabled (threshold=%s frame_ms=%s)",
        cfg.threshold,
        cfg.vad_frame_ms,
    )
    return vad


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


__all__ = [
    "LocalVAD",
    "PassthroughVAD",
    "TenVADGate",
    "VADConfig",
    "VADResult",
    "build_local_vad",
]
