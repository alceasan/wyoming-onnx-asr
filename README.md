# Wyoming ONNX ASR

A [Wyoming protocol](https://github.com/OHF-Voice/wyoming) speech-to-text server built on
[onnx-asr](https://github.com/istupakov/onnx-asr). It runs NVIDIA Parakeet / Canary and other
ONNX speech models on CPU (or NVIDIA GPU) and plugs straight into
[Home Assistant Assist](https://www.home-assistant.io/voice_control/) as a Wyoming STT provider.

This is an actively maintained fork of [tboby/wyoming-onnx-asr](https://github.com/tboby/wyoming-onnx-asr)
(which in turn forked `wyoming-faster-whisper`). The upstream project has been inactive for a while;
this fork keeps the same CLI and Docker layout and adds latency and robustness improvements aimed at
voice assistants.

## What this fork adds

| Feature | What it does |
|---|---|
| **Early final transcript** | While audio is still streaming in, the accumulated buffer is decoded in the background every ~1 s. When `audio-stop` arrives and only silence followed the last partial decode (always the case when a satellite stops after its silence timeout), that partial is returned immediately. The post-speech decode (typically 300–500 ms of Parakeet on a desktop CPU) disappears from the critical path. |
| **Non-blocking decoding** | Recognition runs in a worker thread. The asyncio loop keeps serving other satellites while a decode is in progress (upstream blocked the loop for the whole decode). |
| **Hugging Face repos with precision subfolders** | Repos such as `Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx` store their weights under `int8/`, `fp16/`, … which onnx-asr cannot find. `WYOMING_ONNX_ASR_HF_SUBDIR` fetches the right subfolder and exposes the files where onnx-asr expects them. |
| **Warm-up at start-up** | A dummy decode runs after loading, so the first real request does not pay ONNX Runtime's lazy initialisation. |
| **Wyoming transcript streaming (opt-in)** | Partials can be sent as `transcript-chunk` events (whole transcript so far, as required by Wyoming ≥ 1.11) and the server advertises `supports_transcript_streaming`. Home Assistant's Wyoming integration does not consume these yet, so it is off by default. |
| **No silent English fallback** | If only a multilingual model is loaded, every language goes to it. If a requested model/language cannot be served, the client gets an explicit error instead of an English transcription of non-English audio. |

Everything else (model selection, quantization, GPU images, tools) works as upstream.

## Quick start (Docker)

```shell
docker run -d --name wyoming-onnx-asr \
  -p 10300:10300 \
  -v /path/to/models:/data \
  ghcr.io/alceasan/wyoming-onnx-asr:latest \
  --model-multilingual nemo-parakeet-tdt-0.6b-v3 \
  --default-language es \
  --uri tcp://0.0.0.0:10300
```

Then in Home Assistant: **Settings → Devices & services → Add integration → Wyoming Protocol**,
host = your Docker host, port = `10300`, and select the new STT engine in your Assist pipeline.

### Docker Compose example

```yaml
services:
  wyoming-onnx-asr:
    image: ghcr.io/alceasan/wyoming-onnx-asr:latest
    container_name: wyoming-onnx-asr
    command:
      - "--model-multilingual"
      - "Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx"   # or nemo-parakeet-tdt-0.6b-v3
      - "--default-language"
      - "es"
      - "-q"
      - "int8"
      - "--intra-op-num-threads"
      - "4"
      - "--inter-op-num-threads"
      - "1"
      - "--uri"
      - "tcp://0.0.0.0:10300"
    volumes:
      - /data/models/wyoming-onnx:/data
    ports:
      - "10300:10300"
    restart: unless-stopped
    environment:
      - ORT_DISABLE_CPU_AFFINITY=1
      - OMP_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
      - WYOMING_ONNX_ASR_HF_SUBDIR=auto          # needed for Olicorne/* repos; omit for istupakov/* names
      - WYOMING_ONNX_ASR_PARTIAL_INTERVAL=1.0
      - WYOMING_ONNX_ASR_SILENCE_GUARD=0.15
      - WYOMING_ONNX_ASR_MAX_TAIL=2.5
      - WYOMING_ONNX_ASR_TRANSCRIPT_STREAMING=false
```

GPU images (`-gpu`, `-gpu-tensorrt`) are still built from the same workflows as upstream; run them with
`--gpus all` and `--device gpu`. The maintainer runs the CPU image; GPU images are provided as-is.

## Models

Any name accepted by [onnx-asr](https://github.com/istupakov/onnx-asr?tab=readme-ov-file#supported-model-names)
works, including Hugging Face repo ids. Models are downloaded on first start into
`ONNX_ASR_MODEL_DIR` (default `/data`), one subdirectory per model.

| Model | Languages | Notes |
|---|---|---|
| `nemo-parakeet-tdt-0.6b-v3` | 25 European languages | Default recommendation. Punctuation and capitalisation, ~0.3–0.5 s per command on a modern desktop CPU. |
| `Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx` | 25 European languages | ONNX export of moondream's post-trained v3: lower WER than the stock export at the same speed. Weights live in `int8/`, `fp16/`, `fp32/` subfolders, so set `WYOMING_ONNX_ASR_HF_SUBDIR=auto` (or `int8`). |
| `nemo-canary-1b-v2` | 25 European languages | Higher accuracy, 2–3× slower, encoder–decoder. |
| `nemo-parakeet-tdt-0.6b-v2` | English only | Use with `--model-en`. |
| `onnx-community/whisper-large-v3-turbo` | 99 languages | Much slower on CPU; only if you need a language Parakeet lacks. |

Quantization (`-q int8`) is supported where the model ships int8 files. For Parakeet it roughly halves
the download and memory without a measurable WER change; speed gains depend on the CPU, so compare.

### Model selection rules

- A Wyoming `Transcribe.name` that matches a loaded model name wins.
- Otherwise `en` requests go to `--model-en` if loaded, everything else to `--model-multilingual`.
- If neither can serve the request, the server replies with an `ERROR: …` transcript and logs why.
- `--default-language` (or `WYOMING_ONNX_ASR_DEFAULT_LANGUAGE`) applies when the client sends no language.

## How the early final works

```
satellite:  |---- speech ----|-- silence (VAD timeout) --| audio-stop
server:     p@1s   p@2s   p@3s(ends in silence)          ↑ return p@3s immediately
```

1. Every `WYOMING_ONNX_ASR_PARTIAL_INTERVAL` seconds of new audio (default 1.0) the full buffer is decoded
   in the background. A new partial is not started when only silence arrived since the previous one.
2. On `audio-stop` the server looks for the last completed partial. If the audio from
   `WYOMING_ONNX_ASR_SILENCE_GUARD` seconds before its end (default 0.15) up to the end of the stream is
   silence, that partial is already a complete transcription and is returned at once.
3. Silence is judged per 100 ms frame: RMS below `max(WYOMING_ONNX_ASR_SILENCE_RMS, WYOMING_ONNX_ASR_SILENCE_RATIO × loudest frame of the request)`.
4. If no partial qualifies (speech ran until the very end, or the tail is longer than
   `WYOMING_ONNX_ASR_MAX_TAIL`), the server waits for the in-flight partial if it covers the end, and
   otherwise performs a normal full decode.

The log tells you which path was taken on every request:

```
Early final from partial (tail 0.72s silent, 1 ms after stop): enciende la luz del salón.
Full decode (412 ms after stop): …
```

If you see too many `Full decode` lines with a silent tail, the silence threshold does not match your
microphone: raise `WYOMING_ONNX_ASR_SILENCE_RATIO` (e.g. `0.2`) or `WYOMING_ONNX_ASR_SILENCE_RMS` (e.g. `0.008`).
Set `WYOMING_ONNX_ASR_PARTIALS=false` to get the classic one-shot behaviour.

Cost: two to four extra decodes per utterance while the user is speaking, when the CPU is otherwise idle.

## Configuration

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--uri` | *(required)* | `tcp://0.0.0.0:10300` or `unix://…` |
| `--model-en` | – | English model name |
| `--model-multilingual` | – | Multilingual model name (at least one of the two is required) |
| `-q`, `--quantization` | – | e.g. `int8` |
| `--model-dir` | `$ONNX_ASR_MODEL_DIR` or `/data` | Model cache directory |
| `--device` | `cpu` | `cpu`, `gpu`, `gpu-trt` |
| `--default-language` | `$WYOMING_ONNX_ASR_DEFAULT_LANGUAGE` or `en` | Language when the client sends none |
| `--intra-op-num-threads` / `--inter-op-num-threads` | ORT default | ONNX Runtime threading (4 / 1 is a good start on 8-core CPUs) |
| `--debug` | off | Logs every partial with its timing |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ONNX_ASR_MODEL_DIR` | `/data` | Model cache directory |
| `WYOMING_ONNX_ASR_DEFAULT_LANGUAGE` | `en` | See `--default-language` |
| `WYOMING_ONNX_ASR_HF_SUBDIR` | *(unset)* | For HF repos that keep ONNX files in a subfolder: `int8`, `fp16`, … or `auto` (= the `-q` value, `fp32` if unset). Ignored for plain model names. |
| `WYOMING_ONNX_ASR_WARMUP` | `true` | Run a dummy decode after loading |
| `WYOMING_ONNX_ASR_PARTIALS` | `true` | Enable background partials / early final |
| `WYOMING_ONNX_ASR_PARTIAL_INTERVAL` | `1.0` | Seconds of new audio between partials |
| `WYOMING_ONNX_ASR_PARTIAL_MIN_AUDIO` | `1.0` | Seconds of audio before the first partial |
| `WYOMING_ONNX_ASR_SILENCE_GUARD` | `0.15` | Seconds before a partial's end that must already be silent |
| `WYOMING_ONNX_ASR_MAX_TAIL` | `2.5` | Longest silent tail accepted for an early final |
| `WYOMING_ONNX_ASR_SILENCE_RMS` | `0.004` | Absolute RMS floor for silence (float PCM, 1.0 = full scale) |
| `WYOMING_ONNX_ASR_SILENCE_RATIO` | `0.12` | Silence = frame RMS below this fraction of the loudest frame |
| `WYOMING_ONNX_ASR_TRANSCRIPT_STREAMING` | `false` | Emit `transcript-chunk` events and advertise `supports_transcript_streaming` |
| `ORT_DISABLE_CPU_AFFINITY`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | – | Standard ONNX Runtime / NumPy threading knobs; `1` for the latter two avoids oversubscription inside containers |

## Local install and development

Install [uv](https://docs.astral.sh/uv/) and optionally [mise](https://mise.jdx.dev/):

```sh
git clone https://github.com/alceasan/wyoming-onnx-asr.git
cd wyoming-onnx-asr
uv sync --extra cpu
uv run --extra cpu -m wyoming_onnx_asr --uri tcp://0.0.0.0:10300 --model-multilingual nemo-parakeet-tdt-0.6b-v3
```

`mise run test`, `mise run lint`, `mise run format` and `mise run docker` wrap the usual tasks.
`tests/test_handler_fast.py` exercises the partial/early-final logic with a fake model and needs no ONNX download.

### Tools

- `tools/WyomingASRBenchmark.py` – sends WAV files to a running server and reports latency statistics per model/language.
- `tools/asr_client.py` – minimal Wyoming client to transcribe one WAV file.

## Credits and license

- Original Wyoming server: [Michael Hansen](https://github.com/synesthesiam) (`wyoming-faster-whisper`).
- ONNX ASR port: [Thomas Boby](https://github.com/tboby) (`wyoming-onnx-asr`).
- ONNX models and runtime: [Ilya Stupakov](https://github.com/istupakov) (`onnx-asr`), NVIDIA NeMo, and the community exports referenced above.

MIT License – see [LICENSE.md](LICENSE.md).
