"""Event handler for clients of the server."""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
from onnx_asr.adapters import AsrAdapter
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


def _pcm_to_float32(audio: bytes, sample_width: int, channels: int) -> np.ndarray:
    """Convert little-endian PCM audio bytes to mono float32 samples."""
    if sample_width == 1:
        waveform = np.frombuffer(audio, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif sample_width == 2:
        waveform = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_array = np.frombuffer(audio, dtype=np.uint8).reshape(-1, 3)
        int32_array = (
            bytes_array[:, 0].astype(np.int32)
            | (bytes_array[:, 1].astype(np.int32) << 8)
            | (bytes_array[:, 2].astype(np.int32) << 16)
        )
        int32_array = np.where(
            int32_array & 0x800000, int32_array - 0x1000000, int32_array
        )
        waveform = int32_array.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        waveform = np.frombuffer(audio, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)

    return waveform


@dataclass(frozen=True)
class ModelEntry:
    """Loaded ASR model with Wyoming metadata."""

    name: str
    adapter: AsrAdapter


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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.models = models
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self.default_language = default_language
        self.request_model_name: str | None = None
        self.request_language: str | None = None
        self._audio = bytearray()
        self._audio_rate: int | None = None
        self._audio_width: int | None = None
        self._audio_channels: int | None = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._audio_rate is None:
                self._audio_rate = chunk.rate
                self._audio_width = chunk.width
                self._audio_channels = chunk.channels
            elif (
                chunk.rate != self._audio_rate
                or chunk.width != self._audio_width
                or chunk.channels != self._audio_channels
            ):
                error_msg = "Audio format changed during stream"
                _LOGGER.error(error_msg)
                await self.write_event(Transcript(text=f"ERROR: {error_msg}").event())
                self._reset_request()
                return False

            self._audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._audio_rate is not None
            assert self._audio_width is not None
            assert self._audio_channels is not None

            sample_rate = self._audio_rate
            try:
                waveform = _pcm_to_float32(
                    bytes(self._audio), self._audio_width, self._audio_channels
                )
            except Exception as e:  # noqa: BLE001
                _LOGGER.error("Audio conversion failed: %s", str(e))
                error_msg = f"Audio conversion failed: {e!s}"
                await self.write_event(Transcript(text=f"ERROR: {error_msg}").event())
                self._reset_request()
                return False

            # Decide on language and model
            lang = self.request_language or self.default_language
            model = None

            _LOGGER.info(f"Model requested: {self.request_model_name}")
            _LOGGER.info(f"Language requested: {lang}")
            _LOGGER.info(
                f"Available models: {[entry.name for entry in self.models.values()]}"
            )

            if self.request_model_name is not None:
                for model_key, model_entry in self.models.items():
                    if self.request_model_name in (model_key, model_entry.name):
                        model = model_entry
                        _LOGGER.info(f"Selected requested model '{model_entry.name}'")
                        break
            elif lang == "en" and "en" in self.models:
                model = self.models["en"]
                _LOGGER.info(
                    f"Selected English model '{model.name}' for language '{lang}'"
                )
            elif "multi" in self.models:
                model = self.models["multi"]
                _LOGGER.info(
                    f"Selected multilingual model '{model.name}' for language '{lang}'"
                )
            elif "en" in self.models:
                model = self.models["en"]
                _LOGGER.info(
                    f"Fallback to English model '{model.name}' for language '{lang}'"
                )

            if model is None:
                available_models = [entry.name for entry in self.models.values()]
                if self.request_model_name:
                    _LOGGER.error(
                        "Model '%s' requested but it is not available",
                        self.request_model_name,
                    )
                    _LOGGER.error("Available models: %s", available_models)
                    error_msg = (
                        f"Model '{self.request_model_name}' is not available. "
                        f"Available models: {available_models}"
                    )
                elif self.request_language:
                    _LOGGER.error(
                        "Language '%s' requested but no suitable model is available",
                        self.request_language,
                    )
                    _LOGGER.error("Available models: %s", available_models)
                    error_msg = (
                        f"Language '{self.request_language}' is not supported. "
                        f"Available models: {available_models}"
                    )
                else:
                    _LOGGER.error("No ASR model loaded - server misconfiguration")
                    error_msg = "No ASR model is available for transcription"

                # Send error response instead of raising exception
                await self.write_event(Transcript(text=f"ERROR: {error_msg}").event())
                self._reset_request()
                return False

            async with self.model_lock:
                try:
                    _LOGGER.info(
                        f"Starting transcription with model for language '{lang}'"
                    )
                    text = model.adapter.recognize(
                        waveform, language=lang, sample_rate=sample_rate
                    )
                    _LOGGER.info(
                        f"Transcription completed successfully for language '{lang}'"
                    )
                except Exception as e:  # noqa: BLE001
                    _LOGGER.error("Model recognition failed: %s", str(e))
                    error_msg = f"Transcription failed: {e!s}"
                    await self.write_event(
                        Transcript(text=f"ERROR: {error_msg}").event()
                    )
                    self._reset_request()
                    return False

            _LOGGER.info(f"{lang}:{text}")

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            self._reset_request()

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            # Extract model and language for the next audio stream.
            self.request_model_name = transcribe.name
            self.request_language = transcribe.language
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    def _reset_request(self) -> None:
        self.request_model_name = None
        self.request_language = None
        self._audio.clear()
        self._audio_rate = None
        self._audio_width = None
        self._audio_channels = None
