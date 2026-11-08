# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-11-08

First release maintained in the `alceasan` fork (upstream `tboby/wyoming-onnx-asr` is inactive).

### Added

- Background partial decoding with *early final*: while audio streams in, the buffer is decoded every
  `WYOMING_ONNX_ASR_PARTIAL_INTERVAL` seconds; on `audio-stop`, if only silence followed the last partial,
  it is returned immediately instead of running a full decode. Tunable via `WYOMING_ONNX_ASR_SILENCE_GUARD`,
  `WYOMING_ONNX_ASR_MAX_TAIL`, `WYOMING_ONNX_ASR_SILENCE_RMS`, `WYOMING_ONNX_ASR_SILENCE_RATIO`;
  disable with `WYOMING_ONNX_ASR_PARTIALS=false`.
- Optional Wyoming transcript streaming (`WYOMING_ONNX_ASR_TRANSCRIPT_STREAMING=true`): partials are sent as
  `transcript-chunk` events (whole transcript so far) and `supports_transcript_streaming` is advertised.
- `WYOMING_ONNX_ASR_HF_SUBDIR` to load Hugging Face repos that keep ONNX files in precision subfolders
  (e.g. `Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx` with `int8/`).
- Warm-up decode after model loading (`WYOMING_ONNX_ASR_WARMUP`, default on).
- `tests/test_handler_fast.py` covering partials, early final, non-blocking decode, transcript chunks and errors.

### Changed

- Recognition now runs in a worker thread; the asyncio loop is no longer blocked during a decode.
- New handler module `handler_fast.py` is the default; the previous `handler.py` is kept untouched.
- Per-request log lines state which path produced the result (`Early final …` / `Full decode …`) and its latency.
- README rewritten for this fork.

## [0.6.0]

### Changed

- Updated onnx-asr to 0.11.0, including the new CPU NumPy preprocessor path.
- Respect Wyoming `Transcribe.name` when selecting between loaded ASR models.
- Added default language and ONNX Runtime thread-count options.
- Convert Wyoming PCM audio in memory instead of writing and rereading a temporary WAV file.

## [0.5.0]

### Changed

- Breaking change: `ONNX_ASR_MODEL_DIR` is now used for the model cache directory
- Updated onnx-asr to 0.10.1 with support for canary v2

###

## [0.4.1]

### Fixed

- Fixed version number

## [0.4.0]

### Changed

- Updated onnx-asr to 0.7.0 add support for parakeet-v3

## [0.3.5] - 2025-08-02

### Fixed

- Fix default dockerfile ports

## [0.3.4] - 2025-08-02

### Fixed

- Corrected default docker image arguments (Thanks @RyanMorash)

## [0.3.3] - 2025-07-17

### Fixed
- Remove "en" from the multilingual model

## [0.3.2] - 2025-07-17

### Fixed
- Properly present all language codes for the multilingual model

## [0.3.1] - 2025-07-15

### Changed
- Update defaults to be easier for the addon

## [0.3.0] - 2025-07-13

### Added
- Simple quantization support (for models which support it in onnx-asr)
- Multilingual model setting
- Startup logging

### Changed
- Tests to test quantization

## [0.2.1] - 2025-07-07

### Added
- arm64 support for CPU (pipeline only change)

### Changed
- Re-organised dockerfiles to cause packages to actually be cached in a separate layer

## [0.2.0] - 2025-07-06

### Added
- TensorRT support for GPU optimization - it's not very good though
- Separate Docker image for TensorRT (gpu-tensorrt)
- GitHub workflow for building and publishing TensorRT Docker image

### Changed
- Restructured Docker setup with separate Dockerfiles for GPU and TensorRT
- Renamed Dockerfile.gpu to gpu.Dockerfile and created gpu-tensorrt.Dockerfile
- Split GPU dependencies in pyproject.toml to separate regular GPU and TensorRT options

## [0.1.0] - 2025-07-04

### Changed
- Fork from wyoming-faster-whisper to create wyoming-onnx-asr
- Switch from Faster Whisper to ONNX ASR backend
- Update Python requirement to >=3.10.0
- Replace faster-whisper dependency with onnx-asr[hub]>=0.6
- Changed project name and metadata in pyproject.toml
- Update repository URL to point to new GitHub location
- Update Docker configurations to use ONNX ASR
- Simplified formatting setup by:
  - Moving all formatting configurations to pyproject.toml
  - Adding format script command

### Added
- New benchmark tool (WyomingASRBenchmark.py) for ASR performance testing
- Simple ASR client tool (asr_client.py) for testing
- GPU support with ONNX runtime
- Optional dependencies for CPU and GPU configurations
- Docker Compose configurations for both CPU and GPU setups
- Sound file handling through soundfile>=0.12.1
- Mise tasks

### Removed
- Home Assistant Add-on integration
- Transformers support
- Whisper-specific configurations and dependencies
- Standalone formatting configuration file
- Script files -> use mise instead

### Fixed
- Updated test suite for ONNX ASR compatibility
