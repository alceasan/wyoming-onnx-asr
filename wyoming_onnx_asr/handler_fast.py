"""Wyoming ASR event handler with background partial decoding and early final.

Improvements over ``handler.py``:

* ``recognize`` runs in a worker thread, so the asyncio loop (and other
  clients) are never blocked by a decode.
* While audio is still arriving, the accumulated buffer is decoded every
  ``partial_interval`` seconds ("partials"). Optionally each partial is sent
  as a Wyoming ``transcript-chunk`` (whole transcript so far, as required by
  wyoming >= 1.11) for clients that support transcript streaming.
* On ``audio-stop``, if everything after the last partial (minus a small
  guard) is silence, the last partial is returned immediately as the final
  ``Transcript``. Satellites send ``audio-stop`` only after a silence
  timeout, so this is the common case and it removes the post-speech decode
  from the critical path. Otherwise a full decode is performed as before.

Configuration (environment variables, all optional):

* ``WYOMING_ONNX_ASR_PARTIALS``           default ``true``
* ``WYOMING_ONNX_ASR_PARTIAL_INTERVAL``   seconds of new audio between partials, default ``1.0``
* ``WYOMING_ONNX_ASR_PARTIAL_MIN_AUDIO``  seconds before the first partial, default ``1.0``
* ``WYOMING_ONNX_ASR_SILENCE_GUARD``      seconds before the partial end that must be silent, default ``0.15``
* ``WYOMING_ONNX_ASR_MAX_TAIL``           longest silent tail accepted for an early final, default ``2.5``
* ``WYOMING_ONNX_ASR_SILENCE_RMS``        absolute RMS floor for silence (float PCM), default ``0.004``
* ``WYOMING_ONNX_ASR_SILENCE_RATIO``      silence = frame RMS below ratio * loudest frame, default ``0.12``
* ``WYOMING_ONNX_ASR_TRANSCRIPT_STREAMING`` emit transcript-chunk events, default ``false``
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
from wyoming.asr import Transcribe, Transcript, TranscriptChunk
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .handler import ModelEntry, _pcm_to_float32

_LOGGER = logging.getLogger(__name__)

__all__ = ["ModelEntry", "NemoAsrEventHandler", "_pcm_to_float32"]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _Partial:
    end_bytes: int
    text: str


class NemoAsrEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        models: dict[str, ModelEntry],
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: str | None = None,
        default_language: str = "en",
        partials: bool | None = None,
        partial_interval: float | None = None,
        partial_min_audio: float | None = None,
        silence_guard: float | None = None,
        max_tail: float | None = None,
        silence_rms: float | None = None,
        silence_ratio: float | None = None,
        transcript_streaming: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.models = models
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self.default_language = default_language

        self.partials = (
            _env_flag("WYOMING_ONNX_ASR_PARTIALS", True)
            if partials is None
            else partials
        )
        self.partial_interval = (
            partial_interval
            if partial_interval is not None
            else _env_float("WYOMING_ONNX_ASR_PARTIAL_INTERVAL", 1.0)
        )
        self.partial_min_audio = (
            partial_min_audio
            if partial_min_audio is not None
            else _env_float("WYOMING_ONNX_ASR_PARTIAL_MIN_AUDIO", 1.0)
        )
        self.silence_guard = (
            silence_guard
            if silence_guard is not None
            else _env_float("WYOMING_ONNX_ASR_SILENCE_GUARD", 0.15)
        )
        self.max_tail = (
            max_tail
            if max_tail is not None
            else _env_float("WYOMING_ONNX_ASR_MAX_TAIL", 2.5)
        )
        self.silence_rms = (
            silence_rms
            if silence_rms is not None
            else _env_float("WYOMING_ONNX_ASR_SILENCE_RMS", 0.004)
        )
        self.silence_ratio = (
            silence_ratio
            if silence_ratio is not None
            else _env_float("WYOMING_ONNX_ASR_SILENCE_RATIO", 0.12)
        )
        self.transcript_streaming = (
            _env_flag("WYOMING_ONNX_ASR_TRANSCRIPT_STREAMING", False)
            if transcript_streaming is None
            else transcript_streaming
        )

        self.request_model_name: str | None = None
        self.request_language: str | None = None
        self._audio = bytearray()
        self._audio_rate: int | None = None
        self._audio_width: int | None = None
        self._audio_channels: int | None = None

        self._model: ModelEntry | None = None
        self._lang: str = default_language
        self._partial_task: asyncio.Task[None] | None = None
        self._last_partial: _Partial | None = None
        self._last_partial_end: int = 0  # bytes covered by the last *started* partial
        self._peak_frame_rms: float = 0.0
        self._request_t0: float = 0.0

    # ----------------------------------------------------------------- events

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            return await self._handle_audio_chunk(AudioChunk.from_event(event))

        if AudioStop.is_type(event.type):
            return await self._handle_audio_stop()

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.request_model_name = transcribe.name
            self.request_language = transcribe.language
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> bool:
        if self._audio_rate is None:
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._request_t0 = time.monotonic()
            # Resolve model once per request so partials and final agree.
            self._lang = self.request_language or self.default_language
            self._model = self._select_model(self._lang)
            if self._model is None:
                return await self._fail(self._no_model_message())
        elif (
            chunk.rate != self._audio_rate
            or chunk.width != self._audio_width
            or chunk.channels != self._audio_channels
        ):
            return await self._fail("Audio format changed during stream")

        self._audio.extend(chunk.audio)
        self._track_peak(chunk.audio)

        if self.partials:
            self._maybe_start_partial()
        return True

    async def _handle_audio_stop(self) -> bool:
        assert self._audio_rate is not None
        assert self._audio_width is not None
        assert self._audio_channels is not None
        if self._model is None:
            return await self._fail(self._no_model_message())

        t_stop = time.monotonic()
        total_bytes = len(self._audio)

        text: str | None = None
        partial = self._usable_partial(total_bytes)
        if (
            partial is None
            and self._partial_task is not None
            and not self._partial_task.done()
        ):
            # No completed partial covers the end of speech yet; the one in
            # flight probably does, and waiting for it beats a full decode.
            try:
                await self._partial_task
            except Exception as err:  # noqa: BLE001 - logged inside task
                _LOGGER.debug("Partial failed before stop: %s", err)
            partial = self._usable_partial(total_bytes)
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        self._partial_task = None

        if partial is not None:
            text = partial.text
            _LOGGER.info(
                "Early final from partial (tail %.2fs silent, %.0f ms after stop): %s",
                self._seconds(total_bytes - partial.end_bytes),
                (time.monotonic() - t_stop) * 1000,
                text,
            )

        if text is None:
            try:
                text = await self._recognize(bytes(self._audio))
            except Exception as err:  # noqa: BLE001
                _LOGGER.error("Model recognition failed: %s", err)
                return await self._fail(f"Transcription failed: {err}")
            _LOGGER.info(
                "Full decode (%.0f ms after stop): %s",
                (time.monotonic() - t_stop) * 1000,
                text,
            )

        _LOGGER.info("%s:%s", self._lang, text)
        await self.write_event(Transcript(text=text).event())
        self._reset_request()
        return False

    # --------------------------------------------------------------- partials

    def _usable_partial(self, total_bytes: int) -> _Partial | None:
        """The last completed partial, if only silence followed it."""
        partial = self._last_partial
        if partial is not None and self._tail_is_silence(
            partial.end_bytes, total_bytes
        ):
            return partial
        return None

    def _maybe_start_partial(self) -> None:
        if self._partial_task is not None and not self._partial_task.done():
            return
        self._partial_task = None

        total = len(self._audio)
        if self._seconds(total) < self.partial_min_audio:
            return
        if self._seconds(total - self._last_partial_end) < self.partial_interval:
            return
        if self._usable_partial(total) is not None:
            return  # nothing new to transcribe: only silence since the last partial

        self._last_partial_end = total
        self._partial_task = asyncio.create_task(self._run_partial(total))

    async def _run_partial(self, end_bytes: int) -> None:
        try:
            t0 = time.monotonic()
            text = await self._recognize(bytes(self._audio[:end_bytes]))
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Partial decode failed: %s", err)
            return
        # A later-started partial may have finished first (should not happen
        # because partials are serialized, but keep the newest only).
        if self._last_partial is None or end_bytes >= self._last_partial.end_bytes:
            self._last_partial = _Partial(end_bytes=end_bytes, text=text)
        _LOGGER.debug(
            "Partial @%.2fs took %.0f ms: %s",
            self._seconds(end_bytes),
            (time.monotonic() - t0) * 1000,
            text,
        )
        if self.transcript_streaming and text:
            await self.write_event(TranscriptChunk(text=text).event())

    async def _recognize(self, pcm: bytes) -> str:
        assert self._model is not None
        assert self._audio_width is not None and self._audio_channels is not None
        waveform = _pcm_to_float32(pcm, self._audio_width, self._audio_channels)
        model, lang, rate = self._model, self._lang, self._audio_rate
        async with self.model_lock:
            text = await asyncio.to_thread(
                model.adapter.recognize, waveform, language=lang, sample_rate=rate
            )
        return text if isinstance(text, str) else str(text)

    # ---------------------------------------------------------------- silence

    def _frame_bytes(self) -> int:
        assert self._audio_rate and self._audio_width and self._audio_channels
        return int(self._audio_rate * 0.1) * self._audio_width * self._audio_channels

    def _track_peak(self, pcm: bytes) -> None:
        frame = self._frame_bytes()
        if not frame:
            return
        for offset in range(0, len(pcm) - frame + 1, frame):
            rms = self._rms(pcm[offset : offset + frame])
            self._peak_frame_rms = max(self._peak_frame_rms, rms)

    def _rms(self, pcm: bytes) -> float:
        assert self._audio_width is not None and self._audio_channels is not None
        samples = _pcm_to_float32(pcm, self._audio_width, self._audio_channels)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))))

    def _tail_is_silence(self, partial_end: int, total: int) -> bool:
        """True if audio from (partial_end - guard) to the end is silence."""
        tail_seconds = self._seconds(total - partial_end)
        if tail_seconds > self.max_tail:
            return False
        frame = self._frame_bytes()
        if not frame:
            return False
        guard_bytes = (
            int(self.silence_guard * self._audio_rate)
            * self._audio_width
            * self._audio_channels
        )  # type: ignore[operator]
        start = max(0, partial_end - guard_bytes)
        start -= start % frame
        threshold = max(self.silence_rms, self.silence_ratio * self._peak_frame_rms)
        tail = bytes(self._audio[start:total])
        for offset in range(0, len(tail), frame):
            if self._rms(tail[offset : offset + frame]) > threshold:
                return False
        return True

    def _seconds(self, nbytes: int) -> float:
        if not (self._audio_rate and self._audio_width and self._audio_channels):
            return 0.0
        return nbytes / (self._audio_rate * self._audio_width * self._audio_channels)

    # ------------------------------------------------------------ model choice

    def _select_model(self, lang: str) -> ModelEntry | None:
        if self.request_model_name is not None:
            for key, entry in self.models.items():
                if self.request_model_name in (key, entry.name):
                    _LOGGER.info("Selected requested model '%s'", entry.name)
                    return entry
            return None
        if lang == "en" and "en" in self.models:
            return self.models["en"]
        if "multi" in self.models:
            return self.models["multi"]
        if "en" in self.models:
            _LOGGER.warning("Fallback to English model for language '%s'", lang)
            return self.models["en"]
        return None

    def _no_model_message(self) -> str:
        available = [entry.name for entry in self.models.values()]
        if self.request_model_name:
            return f"Model '{self.request_model_name}' is not available. Available models: {available}"
        if self.request_language:
            return f"Language '{self.request_language}' is not supported. Available models: {available}"
        return "No ASR model is available for transcription"

    # ------------------------------------------------------------------- misc

    async def _fail(self, message: str) -> bool:
        _LOGGER.error(message)
        await self.write_event(Transcript(text=f"ERROR: {message}").event())
        self._reset_request()
        return False

    def _reset_request(self) -> None:
        self.request_model_name = None
        self.request_language = None
        self._audio.clear()
        self._audio_rate = None
        self._audio_width = None
        self._audio_channels = None
        self._model = None
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        self._partial_task = None
        self._last_partial = None
        self._last_partial_end = 0
        self._peak_frame_rms = 0.0
