# ASR Project Agent Guide

## Commands

```bash
# Install dependencies
uv sync

# Run Whisper Large V3 Turbo
uv run whisper_turbo.py <audio_file> [-l LANG] [-t transcribe|translate] [--no-timestamps]

# Run FunASR Nano (中英日)
uv run funasr_nano.py <audio_file> [-l 中文|英文|日文] [-ts]

# Run FunASR MLT (31 languages)
uv run funasr_mlt.py <audio_file> [-l LANG] [-ts]

# YouTube subtitle generation
uv run youtube_subtitle.py <url> [-t TARGET_LANG] [-s SOURCE_LANG] [-m MODEL] [-a ASR_MODEL]

# Video OCR subtitle extraction (hardcoded subtitles)
uv run video_ocr_subtitle.py <video_file> [-t TARGET_LANG] [-l OCR_LANG] [--fps FPS] [--detect-only]
# OCR languages: ch (default), en, japan, korean, german, french
# Options:
#   --detect-only      Only detect if video has hardcoded subtitles
#   --fps N            Frame extraction rate (default: 2.0)
#   --crop-ratio N     Bottom crop ratio for subtitle region (default: 0.25)
#   --no-translate     Skip translation, extract subtitles only
#   --no-burn          Don't burn subtitles into video
# ASR models: whisper (default), funasr-nano, funasr-mlt
# Whisper options:
#   --refine    Use stable-ts for better timestamp accuracy (slower)
#   --no-vad    Disable Silero VAD, use native Whisper chunking
# FunASR options:
#   --funasr-batch-size N    Batch processing duration in seconds (default: 60)
#   --funasr-merge-length N  Max segment merge length in seconds (default: 15)
#   --funasr-vad             Enable Silero VAD pre-segmentation
#   --funasr-refine          Use stable-ts for timestamp refinement
# YouTube subtitle options:
#   --use-youtube-sub        Use existing YouTube subtitles if available (skip ASR)
#   --no-youtube-sub         Force ASR even if YouTube has subtitles
# Subtitle format options:
#   --max-chars N            Max characters per line (default: 40)
#   --max-lines N            Max lines per screen (default: 2)
#   --max-duration N         Max duration per subtitle in seconds (default: 4.0)
#   --subtitle-format FMT    Output format: srt,vtt,ass,json (default: ass)
```

## Project Structure

```
├── whisper_turbo.py      # Whisper Large V3 Turbo ASR
├── funasr_nano.py        # FunASR Nano (Chinese/English/Japanese)
├── funasr_mlt.py         # FunASR MLT (31 languages)
├── youtube_subtitle.py   # YouTube video bilingual subtitle generator
├── model/                # Model storage (auto-downloaded)
├── audio/                # Audio files
└── data/                 # Data directory
    ├── download/         # YouTube video downloads
    ├── audio/            # Extracted audio files
    └── result/           # Subtitle output
```

## Code Style

- **Imports**: stdlib → third-party, group related imports
- **Type hints**: Required on all function signatures and returns
- **Naming**:
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Error handling**: Validate inputs early, descriptive messages in exceptions
- **File paths**: Use `pathlib.Path`, prefer absolute paths
- **Strings**: Prefer f-strings over `.format()`
- **Torch**:
  - Device-agnostic code with `.to(device)`
  - Auto-detect: MPS → CPU
- **Comments**: Minimal unless explaining complex logic

## Key Patterns

```python
# Device detection pattern
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float32

# Model path pattern
MODEL_DIR = Path("./model/model-name")

# Input validation pattern
if not os.path.exists(args.audio_file):
    print(f"错误: 音频文件不存在: {args.audio_file}")
    exit(1)
```

## Dependencies

- `torch` >= 2.9.1
- `transformers` == 4.51.3
- `funasr` >= 1.2.0
- `yt-dlp` >= 2024.1.0
- `httpx` >= 0.27.0
- `stable-ts` >= 2.0

## Testing

No test suite exists. Add pytest setup before testing.
