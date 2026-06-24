"""Audio conversion between browser WebRTC and Realtime PCM.

aiortc delivers browser microphone audio as ``av.AudioFrame`` objects, usually
48 kHz. The Realtime websocket path uses 24 kHz mono PCM16. Keep the resamplers
per-session: they are stateful and sharing them across calls can corrupt audio.
"""

from __future__ import annotations

import base64
from fractions import Fraction

import av
import numpy as np

REALTIME_SAMPLE_RATE = 24000
WEBRTC_SAMPLE_RATE = 48000
WEBRTC_FRAME_SAMPLES = 960


class AudioConverter:
    def __init__(self) -> None:
        self._downsampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=REALTIME_SAMPLE_RATE,
        )
        self._upsampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=WEBRTC_SAMPLE_RATE,
        )

    def aiortc_frame_to_pcm16(self, frame: av.AudioFrame) -> bytes:
        """Convert browser WebRTC audio to 24 kHz mono PCM16 bytes."""
        resampled = self._downsampler.resample(frame)
        if not resampled:
            return b""
        chunks = [f.to_ndarray().flatten().astype(np.int16) for f in resampled]
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return audio.tobytes()

    def realtime_b64_to_aiortc_frame(self, audio_b64: str) -> av.AudioFrame:
        """Convert 24 kHz mono PCM16 base64 from Realtime to a WebRTC frame."""
        raw = base64.b64decode(audio_b64)
        audio = np.frombuffer(raw, dtype=np.int16)
        if len(audio) == 0:
            audio = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)

        frame_24k = av.AudioFrame.from_ndarray(
            audio.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame_24k.sample_rate = REALTIME_SAMPLE_RATE
        frame_24k.time_base = Fraction(1, REALTIME_SAMPLE_RATE)

        resampled = self._upsampler.resample(frame_24k)
        if not resampled:
            silence = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(
                silence.reshape(1, -1),
                format="s16",
                layout="mono",
            )
            frame.sample_rate = WEBRTC_SAMPLE_RATE
            frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
            return frame

        frame = resampled[0]
        frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        return frame


def pcm16_bytes_to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("utf-8")


__all__ = [
    "AudioConverter",
    "REALTIME_SAMPLE_RATE",
    "WEBRTC_SAMPLE_RATE",
    "WEBRTC_FRAME_SAMPLES",
    "pcm16_bytes_to_b64",
]
