from __future__ import annotations

import numpy as np

import gateway.audio.vad as vad_mod
from gateway.audio.vad import PassthroughVAD, TenVADGate, VADConfig


class FakeTenVad:
    def __init__(self, hop_size: int) -> None:
        self.hop_size = hop_size
        self.probabilities = iter([0.0, 0.9, 0.0, 0.0, 0.0])

    def process(self, frame):
        assert len(frame) == self.hop_size
        return next(self.probabilities), None


class FakeTenVadModule:
    TenVad = FakeTenVad


def test_ten_vad_gate_flushes_preroll_and_marks_boundaries(monkeypatch):
    monkeypatch.setattr(vad_mod, "ten_vad", FakeTenVadModule)
    gate = TenVADGate(VADConfig(threshold=0.6, pre_roll_ms=40, hangover_frames=1))
    monkeypatch.setattr(
        gate,
        "_downsample",
        lambda _pcm: np.zeros(gate._hop_size, dtype=np.int16),  # noqa: SLF001
    )

    first_silence = gate.process(b"pre")
    assert first_silence.frames_to_flush == []
    assert not first_silence.speech_started

    speech_start = gate.process(b"start")
    assert speech_start.speech_started
    assert speech_start.frames_to_flush == [b"pre", b"start"]

    first_quiet = gate.process(b"hangover-1")
    assert not first_quiet.speech_ended
    assert first_quiet.frames_to_flush == [b"hangover-1"]

    second_quiet = gate.process(b"hangover-2")
    assert not second_quiet.speech_ended
    assert second_quiet.frames_to_flush == [b"hangover-2"]

    speech_end = gate.process(b"silence")
    assert speech_end.speech_ended
    assert speech_end.frames_to_flush == []


def test_passthrough_vad_forwards_audio_without_boundaries():
    result = PassthroughVAD().process(b"audio")

    assert result.frames_to_flush == [b"audio"]
    assert not result.speech_started
    assert not result.speech_ended
