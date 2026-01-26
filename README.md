# ASR - YouTube Bilingual Subtitle Generator

[中文文档](README_CN.md)

Automated workflow: Download YouTube video → Extract audio → Whisper/FunASR speech recognition → LLM translation → Generate bilingual subtitles → Burn subtitles into video.

## Features

- 🎙️ **Multiple ASR Engines** - Whisper Large V3 Turbo / FunASR Nano / FunASR MLT
- 🌍 **Multi-language Translation** - Supports Ollama local models
- ⏱️ **Precise Timestamps** - Silero VAD voice detection + stable-ts timestamp optimization
- 🎬 **Bilingual Subtitles** - ASS format, original + translated text displayed simultaneously
- 🚀 **Apple Silicon Support** - Automatically uses MPS acceleration
- 💾 **Resume Support** - Recognition and translation results are cached, supports resuming after interruption
- 📄 **Multiple Subtitle Formats** - Supports ASS, SRT, VTT, and JSON formats
- 🎯 **Smart Segmentation** - Intelligent text splitting with semantic boundaries
- 📺 **YouTube Subtitle Integration** - Option to use existing YouTube subtitles instead of ASR

## Architecture

The project follows a modular design with the following components:

- **youtube_subtitle.py**: Main entry point orchestrating the entire workflow
- **asr/**: ASR engine abstraction layer supporting multiple models
  - **whisper.py**: OpenAI Whisper integration with VAD and stable-ts optimization
  - **funasr.py**: Alibaba FunASR integration with multiple language support
  - **vad.py**: Silero VAD voice activity detection
  - **utils.py**: Utility functions for timestamp refinement
  - **base.py**: Common types and configurations
- **Data Pipeline**: Video → Audio extraction → Speech recognition → Translation → Subtitle generation → Video processing

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
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -a whisper      # Whisper Large V3 Turbo (99 languages)
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

# Use existing YouTube subtitles instead of ASR
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --use-youtube-sub

# Smart segmentation options
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --max-chars 40 --max-lines 2 --max-duration 4.0

# Multiple subtitle formats
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle-format srt,vtt,ass
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
| `--max-duration` | Max duration per subtitle segment (seconds) | 4.0 |
| `--subtitle-format` | Subtitle output format: srt,vtt,ass,json (comma-separated) | `ass` |
| `--use-youtube-sub` | Use existing YouTube subtitles instead of ASR | False |
| `--no-youtube-sub` | Force ASR even if YouTube subtitles exist | False |

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
- `data/oss/{video_title}_{asr_model}.srt` - SRT format bilingual subtitle file
- `data/oss/{video_title}_{asr_model}.vtt` - VTT format bilingual subtitle file
- `data/oss/{video_title}_{asr_model}.json` - JSON format subtitle file
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
| CUDA GPU | ✅ Supported, automatic detection |
| CPU | ⚠️ Supported, slower performance |

## Project Structure

```
ASR/
├── youtube_subtitle.py          # Main entry point orchestrating the workflow
├── pyproject.toml               # Project configuration and dependencies
├── model/                       # Model storage directory (auto-download)
│   ├── whisper-large-v3-turbo/  # Whisper model
│   ├── Fun-ASR-Nano-2512/       # FunASR Nano model (Chinese/English/Japanese)
│   └── Fun-ASR-MLT-Nano-2512/   # FunASR MLT model (31 languages)
├── asr/                         # ASR engine abstraction layer
│   ├── __init__.py              # Module exports
│   ├── base.py                  # Common types and configurations
│   ├── whisper.py               # Whisper integration
│   ├── funasr.py                # FunASR integration
│   ├── vad.py                   # Voice activity detection
│   └── utils.py                 # Utility functions
├── data/                        # Data directory
│   ├── video/                   # YouTube video download directory
│   ├── audio/                   # Extracted audio file directory
│   ├── chunks/                  # Recognition and translation cache (.json)
│   ├── oss/                     # Subtitle file output directory
│   └── result/                  # Final video output directory (_subtitled.mp4)
└── README.md                    # Documentation
```

## Smart Segmentation

The project implements intelligent text segmentation with multiple strategies:

1. **Strong punctuation**: Always break at periods, question marks, exclamation points
2. **Weak punctuation**: Break at commas, colons when exceeding character limits
3. **Semantic boundaries**: Break at connectors like "but", "then", "so" when appropriate
4. **Character limits**: Enforce max characters per line and max lines per screen
5. **Duration limits**: Split segments that exceed maximum duration (default 4.0 seconds)

## Known Limitations

- **Whisper**: Processes max 30 seconds of audio at a time, uses chunked processing for longer audio; weakly supervised training may cause hallucinations
- **Translation Quality**: Depends heavily on selected Ollama model and language pair
- **Timestamp Accuracy**: May vary depending on audio quality and speaker characteristics
- **Memory Usage**: Large models require significant RAM, especially with higher batch sizes

## Dependencies

Main dependencies:
- `torch` >= 2.9.1
- `transformers` == 4.51.3
- `funasr` >= 1.2.0
- `yt-dlp` >= 2024.1.0
- `httpx` >= 0.27.0
- `stable-ts` >= 2.0 (timestamp optimization)
- `librosa` >= 0.10.0 (audio processing)
- `soundfile` >= 0.13.1 (audio I/O)

Runtime dependencies (auto-downloaded):
- Silero VAD - Voice activity detection (via torch.hub)

## FAQ

**Q: Translation results are not good?**

A: Try different Ollama models. Recommended: `glm-4.7:cloud`.

**Q: Timestamps are inaccurate?**

A: Use the `--refine` option to enable stable-ts timestamp optimization. If background music interference is severe, Silero VAD will automatically filter it.

**Q: Processing is too slow?**

A: Consider using FunASR models which are faster than Whisper for supported languages. You can also increase batch size with `--batch-size` parameter.

**Q: How do I use existing YouTube subtitles instead of ASR?**

A: Use the `--use-youtube-sub` flag to download and use existing YouTube subtitles instead of performing ASR.

**Q: Can I generate multiple subtitle formats?**

A: Yes, use the `--subtitle-format` parameter with comma-separated values (e.g., `srt,vtt,ass`).

## License

MIT
