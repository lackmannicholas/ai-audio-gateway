"""A dead-simple energy-based VAD. Good enough to demonstrate barge-in; not
production-grade. Real systems use a proper model (Silero, WebRTC VAD, etc.) —
swap this out behind the same tiny interface.

It tracks short-term audio energy over 16-bit PCM frames and reports speech
start/stop with a little hysteresis so a single quiet frame doesn't end speech.

RMS is computed in pure Python (via ``array``) rather than ``audioop``, which
was removed in Python 3.13. One fewer stdlib dependency to outlive us.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass


@dataclass
class VADConfig:
    energy_threshold: int = 500        # RMS above this = speech
    start_frames: int = 2              # consecutive speech frames to start
    stop_frames: int = 10              # consecutive silence frames to stop (~200ms @20ms)


def _rms16(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    samples = array.array("h")
    # Tolerate odd-length buffers by truncating the trailing byte.
    samples.frombytes(pcm16[: len(pcm16) - (len(pcm16) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class EnergyVAD:
    def __init__(self, config: VADConfig | None = None) -> None:
        self._cfg = config or VADConfig()
        self._speech = False
        self._speech_run = 0
        self._silence_run = 0

    def is_speaking(self) -> bool:
        return self._speech

    def process(self, pcm16: bytes) -> str | None:
        """Feed one frame. Returns 'start', 'stop', or None."""
        rms = _rms16(pcm16)
        loud = rms >= self._cfg.energy_threshold

        if loud:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if not self._speech and self._speech_run >= self._cfg.start_frames:
            self._speech = True
            return "start"
        if self._speech and self._silence_run >= self._cfg.stop_frames:
            self._speech = False
            return "stop"
        return None


__all__ = ["EnergyVAD", "VADConfig"]
