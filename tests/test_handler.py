"""Tests for Wyoming event handler model selection."""

import asyncio

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.info import Info

from wyoming_onnx_asr.handler import ModelEntry, NemoAsrEventHandler, _pcm_to_float32


class FakeAsrAdapter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def recognize(self, waveform, **kwargs):
        self.calls.append({"waveform": waveform, **kwargs})
        return self.text


def _audio_chunk() -> AudioChunk:
    return AudioChunk(rate=16_000, width=2, channels=1, audio=b"\0\0" * 160)


async def _create_handler(
    en_adapter: FakeAsrAdapter,
    multi_adapter: FakeAsrAdapter,
    default_language: str = "en",
) -> tuple[NemoAsrEventHandler, list]:
    handler = NemoAsrEventHandler(
        Info(),
        {
            "en": ModelEntry("english-model", en_adapter),
            "multi": ModelEntry("multilingual-model", multi_adapter),
        },
        asyncio.Lock(),
        asyncio.StreamReader(),
        object(),
        default_language=default_language,
    )
    written_events = []

    async def write_event(event):
        written_events.append(event)

    handler.write_event = write_event
    return handler, written_events


@pytest.mark.asyncio
async def test_transcribe_name_selects_requested_model() -> None:
    en_adapter = FakeAsrAdapter("english")
    multi_adapter = FakeAsrAdapter("multilingual")
    handler, written_events = await _create_handler(en_adapter, multi_adapter)

    assert await handler.handle_event(
        Transcribe(name="multilingual-model", language="nl").event()
    )
    assert await handler.handle_event(_audio_chunk().event())
    assert not await handler.handle_event(AudioStop().event())

    assert not en_adapter.calls
    assert len(multi_adapter.calls) == 1
    assert multi_adapter.calls[0]["language"] == "nl"
    assert Transcript.from_event(written_events[-1]).text == "multilingual"


@pytest.mark.asyncio
async def test_default_language_selects_english_model() -> None:
    en_adapter = FakeAsrAdapter("english")
    multi_adapter = FakeAsrAdapter("multilingual")
    handler, written_events = await _create_handler(en_adapter, multi_adapter)

    assert await handler.handle_event(_audio_chunk().event())
    assert not await handler.handle_event(AudioStop().event())

    assert len(en_adapter.calls) == 1
    assert not multi_adapter.calls
    assert en_adapter.calls[0]["language"] == "en"
    assert Transcript.from_event(written_events[-1]).text == "english"


@pytest.mark.asyncio
async def test_unknown_requested_model_returns_error() -> None:
    en_adapter = FakeAsrAdapter("english")
    multi_adapter = FakeAsrAdapter("multilingual")
    handler, written_events = await _create_handler(en_adapter, multi_adapter)

    assert await handler.handle_event(Transcribe(name="missing-model").event())
    assert await handler.handle_event(_audio_chunk().event())
    assert not await handler.handle_event(AudioStop().event())

    assert not en_adapter.calls
    assert not multi_adapter.calls
    assert (
        "ERROR: Model 'missing-model' is not available"
        in Transcript.from_event(written_events[-1]).text
    )
    assert handler.request_model_name is None
    assert handler.request_language is None


@pytest.mark.asyncio
async def test_default_language_selects_multilingual_model() -> None:
    en_adapter = FakeAsrAdapter("english")
    multi_adapter = FakeAsrAdapter("multilingual")
    handler, written_events = await _create_handler(
        en_adapter, multi_adapter, default_language="es"
    )

    assert await handler.handle_event(_audio_chunk().event())
    assert not await handler.handle_event(AudioStop().event())

    assert not en_adapter.calls
    assert len(multi_adapter.calls) == 1
    assert multi_adapter.calls[0]["language"] == "es"
    assert Transcript.from_event(written_events[-1]).text == "multilingual"


def test_pcm_to_float32_averages_channels() -> None:
    waveform = _pcm_to_float32(
        b"\x00\x40\x00\x80\x00\x20\x00\x60", sample_width=2, channels=2
    )

    assert waveform.tolist() == [-0.25, 0.5]
