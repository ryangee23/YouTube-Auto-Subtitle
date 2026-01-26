# ASR - YouTube 双语字幕生成工具

[English Documentation](README.md)

自动化工作流：下载 YouTube 视频 → 提取音频 → Whisper/FunASR 语音识别 → LLM 翻译 → 生成双语字幕 → 烧录字幕到视频。

## 功能特性

- 🎙️ **多 ASR 引擎** - Whisper Large V3 Turbo / FunASR Nano / FunASR MLT
- 🌍 **多语言翻译** - 支持 Ollama 本地模型
- ⏱️ **精准时间戳** - Silero VAD 语音检测 + stable-ts 时间戳优化
- 🎬 **双语字幕** - ASS 格式，原文+译文同时显示
- 🚀 **Apple Silicon 支持** - 自动使用 MPS 加速
- 💾 **断点续传** - 识别和翻译结果自动缓存，支持中断后继续
- 📄 **多种字幕格式** - 支持 ASS、SRT、VTT 和 JSON 格式
- 🎯 **智能分段** - 基于语义边界的智能文本分割
- 📺 **YouTube 字幕集成** - 可选择使用现有的 YouTube 字幕而非 ASR

## 架构设计

项目采用模块化设计，包含以下组件：

- **youtube_subtitle.py**: 主入口点，协调整个工作流程
- **asr/**: 支持多种模型的 ASR 引擎抽象层
  - **whisper.py**: OpenAI Whisper 集成，支持 VAD 和 stable-ts 优化
  - **funasr.py**: 阿里 FunASR 集成，支持多种语言
  - **vad.py**: Silero VAD 语音活动检测
  - **utils.py**: 时间戳优化的实用函数
  - **base.py**: 通用类型和配置
- **数据管道**: 视频 → 音频提取 → 语音识别 → 翻译 → 字幕生成 → 视频处理

## 安装

### 环境要求

- Python >= 3.14
- [uv](https://github.com/astral-sh/uv) 包管理器
- [FFmpeg](https://ffmpeg.org/)
- [Ollama](https://ollama.com/) (使用本地模型翻译时需要)

### 安装依赖

```bash
uv sync
```

## 使用方法

### 基本使用

```bash
# 下载视频，识别语音，翻译成中文，添加双语字幕
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID"

# 指定翻译目标语言
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -t ja

# 指定源语言 (跳过自动检测)
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -s en -t zh-CN

# 选择 ASR 模型
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -a whisper      # Whisper Large V3 Turbo (99 种语言)
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -a funasr-nano  # FunASR Nano (中英日)
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -a funasr-mlt   # FunASR MLT (31 种语言)

# 指定 Ollama 翻译模型
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" -m gemini-3-flash-preview:cloud

# 高精度时间戳 (使用 stable-ts 优化，更慢但更准确)
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --refine

# 禁用 Silero VAD，使用原生 Whisper 分段
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --no-vad

# 自定义目录
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --download-dir ./videos --result-dir ./subtitled

# 使用现有的 YouTube 字幕而非 ASR
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --use-youtube-sub

# 智能分段选项
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --max-chars 40 --max-lines 2 --max-duration 4.0

# 多种字幕格式
uv run youtube_subtitle.py "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle-format srt,vtt,ass
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `url` | YouTube 视频 URL | (必需) |
| `-t, --target-lang` | 翻译目标语言代码 | `zh-CN` |
| `-s, --source-lang` | 源语言代码 | 自动检测 |
| `-a, --asr` | ASR 模型：`whisper` / `funasr-nano` / `funasr-mlt` | `whisper` |
| `--download-dir` | 视频下载目录 | `data/video` |
| `-o, --result-dir` | 结果输出目录 | `data/result` |
| `-m, --model` | Ollama 翻译模型 | `glm-4.7:cloud` |
| `-b, --batch-size` | Whisper 批处理大小，增大可加速但需更多内存 | 1 |
| `--refine` | 使用 stable-ts 优化时间戳精度 (更准确但更慢，仅 Whisper) | False |
| `--no-vad` | 禁用 Silero VAD，使用原生 Whisper 分段 | False |
| `--funasr-batch-size` | FunASR 批处理时长（秒） | 60 |
| `--funasr-merge-length` | FunASR 合并片段最大时长（秒） | 15 |
| `--funasr-vad` | FunASR 启用 Silero VAD 预分片 | False |
| `--funasr-refine` | FunASR 使用 stable-ts 精调时间戳 | False |
| `--max-chars` | 字幕每行最大字符数 | 40 |
| `--max-lines` | 字幕每屏最大行数 | 2 |
| `--max-duration` | 字幕片段最大持续时间（秒） | 4.0 |
| `--subtitle-format` | 字幕输出格式：srt,vtt,ass,json（逗号分隔） | `ass` |
| `--use-youtube-sub` | 使用现有的 YouTube 字幕而非 ASR | False |
| `--no-youtube-sub` | 即使 YouTube 有字幕也强制使用 ASR | False |

### ASR 模型

| 模型 | 说明 | 语言支持 |
|------|------|----------|
| `whisper` | OpenAI Whisper Large V3 Turbo (默认) | 99 种语言 |
| `funasr-nano` | 阿里 FunASR Nano | 中文、英文、日文 |
| `funasr-mlt` | 阿里 FunASR MLT | 31 种语言 |

### 翻译模型

**Ollama Cloud:**

| 模型 | 说明 |
|------|------|
| `glm-4.7:cloud` | 智谱 GLM-4 (默认) |
| `gemini-3-flash-preview:cloud` | Google Gemini 3 Flash |
| `minimax-m2.1:cloud` | MiniMax M2.1 |

### 输出文件

- `data/chunks/{视频标题}_{asr模型}.json` - 识别和翻译结果缓存
- `data/oss/{视频标题}_{asr模型}.ass` - ASS 格式双语字幕文件
- `data/oss/{视频标题}_{asr模型}.srt` - SRT 格式双语字幕文件
- `data/oss/{视频标题}_{asr模型}.vtt` - VTT 格式双语字幕文件
- `data/oss/{视频标题}_{asr模型}.json` - JSON 格式字幕文件
- `data/result/{视频标题}_subtitled.mp4` - 烧录字幕后的视频

## 模型下载

首次运行时会自动从 HuggingFace 下载模型到 `./model/` 目录：

| 模型文件 | 来源 |
|----------|------|
| `whisper-large-v3-turbo/` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| `Fun-ASR-Nano-2512/` | [FunAudioLLM/FunASR-Nano-2512](https://huggingface.co/FunAudioLLM/FunASR-Nano-2512) |
| `Fun-ASR-MLT-Nano-2512/` | [FunAudioLLM/FunASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/FunASR-MLT-Nano-2512) |

## 硬件要求

| 硬件 | 支持情况 |
|------|----------|
| Apple Silicon (MPS) | ✅ 推荐，自动检测并使用 |
| CUDA GPU | ✅ 支持，自动检测 |
| CPU | ⚠️ 支持，速度较慢 |

## 项目结构

```
ASR/
├── youtube_subtitle.py          # 主入口点，协调整个工作流程
├── pyproject.toml               # 项目配置和依赖
├── model/                       # 模型存储目录 (自动下载)
│   ├── whisper-large-v3-turbo/  # Whisper 模型
│   ├── Fun-ASR-Nano-2512/       # FunASR Nano 模型 (中英日)
│   └── Fun-ASR-MLT-Nano-2512/   # FunASR MLT 模型 (31 种语言)
├── asr/                         # ASR 引擎抽象层
│   ├── __init__.py              # 模块导出
│   ├── base.py                  # 通用类型和配置
│   ├── whisper.py               # Whisper 集成
│   ├── funasr.py                # FunASR 集成
│   ├── vad.py                   # 语音活动检测
│   └── utils.py                 # 实用函数
├── data/                        # 数据目录
│   ├── video/                   # YouTube 视频下载目录
│   ├── audio/                   # 提取的音频文件目录
│   ├── chunks/                  # 识别和翻译结果缓存 (.json)
│   ├── oss/                     # 字幕文件输出目录
│   └── result/                  # 最终视频输出目录 (_subtitled.mp4)
└── README.md                    # 文档
```

## 智能分段

项目实现了基于多种策略的智能文本分段：

1. **强标点符号**: 在句号、问号、感叹号处始终断开
2. **弱标点符号**: 在超出字符限制时在逗号、冒号处断开
3. **语义边界**: 在"但是"、"然后"、"所以"等连接词处适当断开
4. **字符限制**: 强制每行最大字符数和每屏最大行数
5. **时长限制**: 分割超过最大持续时间的片段（默认 4.0 秒）

## 已知限制

- **Whisper**: 单次处理最多 30 秒音频，长音频使用分块处理；弱监督训练可能导致幻觉
- **翻译质量**: 严重依赖所选的 Ollama 模型和语言对
- **时间戳准确性**: 可能因音频质量和说话人特征而有所不同
- **内存使用**: 大模型需要大量 RAM，特别是使用更高批处理大小时

## 依赖项

主要依赖：
- `torch` >= 2.9.1
- `transformers` == 4.51.3
- `funasr` >= 1.2.0
- `yt-dlp` >= 2024.1.0
- `httpx` >= 0.27.0
- `stable-ts` >= 2.0 (时间戳优化)
- `librosa` >= 0.10.0 (音频处理)
- `soundfile` >= 0.13.1 (音频 I/O)

运行时依赖 (自动下载)：
- Silero VAD - 语音活动检测 (via torch.hub)

## 常见问题

**Q: 翻译结果不理想？**

A: 尝试不同的 Ollama 模型。推荐：`glm-4.7:cloud`。

**Q: 时间戳不准确？**

A: 使用 `--refine` 选项启用 stable-ts 时间戳优化。如果背景音乐干扰严重，Silero VAD 会自动过滤。

**Q: 处理速度太慢？**

A: 考虑使用 FunASR 模型，对于支持的语言比 Whisper 更快。您也可以使用 `--batch-size` 参数增加批处理大小。

**Q: 如何使用现有的 YouTube 字幕而非 ASR？**

A: 使用 `--use-youtube-sub` 标志来下载和使用现有的 YouTube 字幕，而不是执行 ASR。

**Q: 我可以生成多种字幕格式吗？**

A: 是的，使用 `--subtitle-format` 参数配合逗号分隔的值（例如 `srt,vtt,ass`）。

## License

MIT
