"""Tests for the partial/early-final handler."""

import asyncio
import sys
import time
import types

import numpy as np
import pytest

# onnx_asr is optional for these tests: provide a stub so handler.py imports.
if "onnx_asr" not in sys.modules:
    onnx_asr = types.ModuleType("onnx_asr")
    adapters = types.ModuleType("onnx_asr.adapters")

    class AsrAdapter:  # pragma: no cover - stub
        pass

    adapters.AsrAdapter = AsrAdapter
    onnx_asr.adapters = adapters
    sys.modules["onnx_asr"] = onnx_asr
    sys.modules["onnx_asr.adapters"] = adapters

from wyoming.asr import Transcribe, Transcript, TranscriptChunk
from wyoming.audio import AudioChunk, AudioStop
from wyoming.info import Info

from wyoming_onnx_asr.handler_fast import ModelEntry, NemoAsrEventHandler

RATE = 16_000


class FakeAdapter:
    """Decode cost: 80 ms fixed + 0.1 x audio duration (RTF 0.1)."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def recognize(self, waveform, **kwargs):
        seconds = len(waveform) / RATE
        self.calls.append(seconds)
        time.sleep(0.08 + 0.1 * seconds)
        return f"texto hasta {seconds:.1f}s"


def _pcm(seconds: float, amplitude: float, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    samples = rng.uniform(-1.0, 1.0, n) * amplitude
    return (samples * 32767).astype("<i2").tobytes()


async def _handler(adapter: FakeAdapter, **kwargs):
    handler = NemoAsrEventHandler(
        Info(),
        {"multi": ModelEntry("parakeet", adapter)},
        asyncio.Lock(),
        asyncio.StreamReader(),
        object(),
        default_language="es",
        partial_interval=0.5,
        partial_min_audio=0.5,
        **kwargs,
    )
    written = []

    async def write_event(event):
        written.append((time.monotonic(), event))

    handler.write_event = write_event
    return handler, written


async def _stream(handler, pcm: bytes, chunk_seconds: float = 0.1, pace: float = 0.02):
    chunk_bytes = int(RATE * chunk_seconds) * 2
    for offset in range(0, len(pcm), chunk_bytes):
        await handler.handle_event(
            AudioChunk(
                rate=RATE, width=2, channels=1, audio=pcm[offset : offset + chunk_bytes]
            ).event()
        )
        await asyncio.sleep(pace)
    t_stop = time.monotonic()
    await handler.handle_event(AudioStop().event())
    return t_stop


@pytest.mark.asyncio
async def test_early_final_when_tail_is_silence() -> None:
    adapter = FakeAdapter()
    handler, written = await _handler(adapter)
    await handler.handle_event(Transcribe(language="es").event())

    audio = _pcm(3.0, 0.3) + _pcm(1.0, 0.002, seed=1)  # speech then silence
    t_stop = await _stream(handler, audio, pace=0.1)  # real-time pacing

    finals = [e for _, e in written if Transcript.is_type(e.type)]
    assert len(finals) == 1
    text = Transcript.from_event(finals[0]).text
    # Final came from a partial that already covered the speech (>= 3.0 s)
    assert text.startswith("texto hasta") and float(text.split()[2][:-1]) >= 3.0
    assert float(text.split()[2][:-1]) <= 4.0
    # And it was emitted right after stop, without a decode on the critical path
    assert written[-1][0] - t_stop < 0.06, (
        f"final took {written[-1][0] - t_stop:.3f}s after stop"
    )
    assert len(adapter.calls) >= 3  # partials happened


@pytest.mark.asyncio
async def test_full_decode_when_speech_until_the_end() -> None:
    adapter = FakeAdapter()
    handler, written = await _handler(adapter)

    audio = _pcm(2.0, 0.3)  # no silent tail
    t_stop = await _stream(handler, audio)

    finals = [e for _, e in written if Transcript.is_type(e.type)]
    assert Transcript.from_event(finals[0]).text == "texto hasta 2.0s"
    assert written[-1][0] - t_stop >= 0.08  # a real decode happened after stop


@pytest.mark.asyncio
async def test_event_loop_not_blocked_during_decode() -> None:
    adapter = FakeAdapter()
    handler, _ = await _handler(adapter, partials=False)

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    await _stream(handler, _pcm(3.0, 0.3), pace=0.0)  # decode ~0.38 s in a thread
    task.cancel()
    assert ticks >= 20, f"loop was blocked (only {ticks} ticks)"


@pytest.mark.asyncio
async def test_transcript_chunks_when_enabled() -> None:
    adapter = FakeAdapter()
    handler, written = await _handler(adapter, transcript_streaming=True)
    await _stream(handler, _pcm(2.0, 0.3) + _pcm(0.8, 0.002, seed=2))
    chunks = [e for _, e in written if TranscriptChunk.is_type(e.type)]
    assert chunks, "no transcript-chunk emitted"
    assert [e.type for _, e in written][-1] == "transcript"


@pytest.mark.asyncio
async def test_error_when_no_model_for_request() -> None:
    adapter = FakeAdapter()
    handler, written = await _handler(adapter)
    await handler.handle_event(Transcribe(name="does-not-exist").event())
    result = await handler.handle_event(
        AudioChunk(rate=RATE, width=2, channels=1, audio=b"\0\0" * 160).event()
    )
    assert result is False
    assert Transcript.from_event(written[0][1]).text.startswith("ERROR:")
