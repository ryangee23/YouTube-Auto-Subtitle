# ASR - YouTube Bilingual Subtitle Generator

[中文文档](README_CN.md)

Automated workflow: Download YouTube video → Extract audio → Whisper speech recognition → LLM translation → Generate ASS bilingual subtitles → Burn subtitles into video.

## Features

- 🎙️ **Multiple ASR Engines** - Whisper Large V3 Turbo / FunASR Nano / FunASR MLT
- 🌍 **Multi-language Translation** - Supports Ollama local models
- ⏱️ **Precise Timestamps** - Silero VAD voice detection + stable-ts timestamp optimization
- 🎬 **Bilingual Subtitles** - ASS format, original + translated text displayed simultaneously
- 🚀 **Apple Silicon Support** - Automatically uses MPS acceleration
- 💾 **Resume Support** - Recognition and translation results are cached, supports resuming after interruption

## Installation

### Requirements

- Python >= 3.14
- [uv](https://github.com/astral-sh/uv) package manager
- [FFmpeg](https://ffmpeg.org/)
- [Ollama](https://ollama.com/) (required for local model translation)

### Install Dependencies

```bash
uv sync
```

## Usage

### Basic Usage

```bash
# Download video, recognize speech, translate to Chinese, add bilingual subtitles
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify target language for translation
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -t ja

# Specify source language (skip auto-detection)
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -s en -t zh-CN

# Choose ASR model
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -a funasr-nano  # FunASR Nano (Chinese/English/Japanese)
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -a funasr-mlt   # FunASR MLT (31 languages)

# Specify Ollama translation model
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -m gemini-3-flash-preview:cloud

# High-precision timestamps (uses stable-ts optimization, slower but more accurate)
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --refine

# Disable Silero VAD, use native Whisper chunking
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --no-vad

# Custom directories
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --download-dir ./videos --result-dir ./subtitled
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `url` | YouTube video URL | (required) |
| `-t, --target-lang` | Target language code for translation | `zh-CN` |
| `-s, --source-lang` | Source language code | Auto-detect |
| `-a, --asr` | ASR model: `whisper` / `funasr-nano` / `funasr-mlt` | `whisper` |
| `--download-dir` | Video download directory | `data/video` |
| `-o, --result-dir` | Result output directory | `data/result` |
| `-m, --model` | Ollama translation model | `glm-4.7:cloud` |
| `-b, --batch-size` | Whisper batch size, increase for speed but requires more memory | 1 |
| `--refine` | Use stable-ts for timestamp precision (more accurate but slower, Whisper only) | False |
| `--no-vad` | Disable Silero VAD, use native Whisper chunking | False |
| `--funasr-batch-size` | FunASR batch processing duration (seconds) | 60 |
| `--funasr-merge-length` | FunASR max segment merge length (seconds) | 15 |
| `--funasr-vad` | Enable Silero VAD pre-segmentation for FunASR | False |
| `--funasr-refine` | Use stable-ts for FunASR timestamp refinement | False |
| `--max-chars` | Max characters per subtitle line | 40 |
| `--max-lines` | Max lines per subtitle screen | 2 |
| `--subtitle-format` | Subtitle output format: srt,vtt,ass,json (comma-separated) | `ass` |

### ASR Models

| Model | Description | Language Support |
|-------|-------------|------------------|
| `whisper` | OpenAI Whisper Large V3 Turbo (default) | 99 languages |
| `funasr-nano` | Alibaba FunASR Nano | Chinese, English, Japanese |
| `funasr-mlt` | Alibaba FunASR MLT | 31 languages |

### Translation Models

**Ollama Cloud:**

| Model | Description |
|-------|-------------|
| `glm-4.7:cloud` | Zhipu GLM-4 (default) |
| `gemini-3-flash-preview:cloud` | Google Gemini 3 Flash |
| `minimax-m2.1:cloud` | MiniMax M2.1 |

### Output Files

- `data/chunks/{video_title}_{asr_model}.json` - Recognition and translation result cache
- `data/oss/{video_title}_{asr_model}.ass` - ASS format bilingual subtitle file
- `data/result/{video_title}_subtitled.mp4` - Video with burned-in subtitles

## Model Download

Models are automatically downloaded from HuggingFace to `./model/` directory on first run:

| Model File | Source |
|------------|--------|
| `whisper-large-v3-turbo/` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| `Fun-ASR-Nano-2512/` | [FunAudioLLM/FunASR-Nano-2512](https://huggingface.co/FunAudioLLM/FunASR-Nano-2512) |
| `Fun-ASR-MLT-Nano-2512/` | [FunAudioLLM/FunASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/FunASR-MLT-Nano-2512) |

## Hardware Requirements

| Hardware | Support |
|----------|---------|
| Apple Silicon (MPS) | ✅ Recommended, auto-detected and used |
| CPU | ⚠️ Supported, slower performance |

## Project Structure

```
ASR/
├── youtube_subtitle.py          # YouTube video bilingual subtitle generation
├── pyproject.toml               # Project configuration and dependencies
├── model/                       # Model storage directory (auto-download)
│   ├── whisper-large-v3-turbo/  # Whisper model
│   ├── Fun-ASR-Nano-2512/       # FunASR Nano model (Chinese/English/Japanese)
│   └── Fun-ASR-MLT-Nano-2512/   # FunASR MLT model (31 languages)
├── audio/                       # Audio file directory
└── data/                        # Data directory
    ├── video/                   # YouTube video download directory
    ├── audio/                   # Extracted audio file directory
    ├── chunks/                  # Recognition and translation cache (.json)
    ├── oss/                     # Subtitle file output directory (.ass)
    └── result/                  # Final video output directory (_subtitled.mp4)
```

## Known Limitations

- **Whisper**: Processes max 30 seconds of audio at a time, uses chunked processing for longer audio; weakly supervised training may cause hallucinations
- **YouTube Subtitles**: Ollama requires local service running; translation quality depends on model choice

## Dependencies

Main dependencies:
- `torch` >= 2.9.1
- `transformers` == 4.51.3
- `funasr` >= 1.2.0
- `yt-dlp` >= 2024.1.0
- `httpx` >= 0.27.0
- `stable-ts` >= 2.0 (timestamp optimization)

Runtime dependencies (auto-downloaded):
- Silero VAD - Voice activity detection (via torch.hub)

## FAQ

**Q: Translation results are not good?**

A: Try different Ollama models. Recommended: `glm-4.7:cloud`.

**Q: Timestamps are inaccurate?**

A: Use the `--refine` option to enable stable-ts timestamp optimization. If background music interference is severe, Silero VAD will automatically filter it.

## License

MIT
