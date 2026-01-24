#!/usr/bin/env python3
"""YouTube视频下载、语音识别、翻译并添加双语字幕

本模块提供完整的YouTube视频处理流水线：
1. 下载YouTube视频（最高画质）
2. 提取音频轨道
3. 使用ASR模型（Whisper/FunASR）进行语音识别
4. 使用LLM（Ollama）进行翻译
5. 生成ASS双语字幕
6. 将字幕烧录到视频
"""

# ============================================================
# 标准库导入
# ============================================================
import argparse  # 命令行参数解析
import asyncio  # 异步IO支持，用于并发翻译
import json  # JSON序列化/反序列化
import os  # 操作系统接口，读取环境变量
import subprocess  # 子进程调用，用于执行FFmpeg命令
from pathlib import Path  # 面向对象的文件路径处理

# ============================================================
# 第三方库导入
# ============================================================
import httpx  # 异步HTTP客户端，用于API调用
import numpy as np  # 数值计算，用于音频数组处理
# tempfile不再需要 - 直接处理音频数组
import torch  # PyTorch深度学习框架
import yt_dlp  # YouTube视频下载库
from transformers import (  # HuggingFace Transformers
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

# ============================================================
# Ollama 本地模型配置
# ============================================================
OLLAMA_MODELS = [
    "glm-4.7:cloud",                # 智谱GLM-4.7云端模型
    "gemini-3-flash-preview:cloud",  # Google Gemini 3 Flash预览版
    "minimax-m2.1:cloud",            # MiniMax M2.1云端模型
]
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama本地API端点

# ============================================================
# 计算设备配置
# ============================================================
# 优先使用MPS (Apple Silicon)，否则回退到CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float32

# ============================================================
# ASR模型路径配置
# ============================================================
ASR_MODELS = {
    "whisper": Path("./model/whisper-large-v3-turbo"),      # OpenAI Whisper Large V3 Turbo
    "funasr-nano": Path("./model/Fun-ASR-Nano-2512"),       # FunASR Nano（中英日）
    "funasr-mlt": Path("./model/Fun-ASR-MLT-Nano-2512"),    # FunASR MLT（31种语言）
}
DEFAULT_ASR_MODEL = "whisper"  # 默认使用Whisper模型

# ============================================================
# 目录路径配置
# ============================================================
DOWNLOAD_DIR = Path("./data/video")   # 视频下载目录
AUDIO_DIR = Path("./data/audio")      # 提取的音频存储目录
OSS_DIR = Path("./data/oss")          # 生成的字幕文件目录
RESULT_DIR = Path("./data/result")    # 最终输出（带字幕视频）目录
CHUNKS_DIR = Path("./data/chunks")    # 识别结果缓存目录


def sanitize_filename(title: str) -> str:
    """将视频标题转换为安全的纯英文文件名

    Args:
        title: 原始视频标题，可能包含中文、特殊字符等

    Returns:
        处理后的安全文件名，只包含英文字母、数字和下划线

    Example:
        >>> sanitize_filename("【中文】Hello World! 2024")
        'Hello_World_2024'
    """
    import re  # 正则表达式模块
    # 第一步：移除所有非英文字母、数字、空格的字符
    sanitized = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    # 第二步：将连续空格替换为单个下划线，并去除首尾空格
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    # 第三步：限制文件名长度为100字符，如果为空则使用默认值"video"
    return sanitized[:100] if sanitized else "video"


def download_youtube_subtitles(url: str, output_dir: Path, lang: str | None = None) -> Path | None:
    """尝试下载YouTube视频的现有字幕

    优先下载人工字幕，如果不存在则下载自动生成的字幕。

    Args:
        url: YouTube视频URL
        output_dir: 字幕保存目录
        lang: 首选语言代码（如 'en', 'zh'），None表示自动选择

    Returns:
        下载的字幕文件路径，如果没有可用字幕则返回None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with yt_dlp.YoutubeDL({"skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        safe_title = sanitize_filename(info.get("title", "video"))
        subtitles = info.get("subtitles", {})
        auto_captions = info.get("automatic_captions", {})

    available_subs = subtitles if subtitles else auto_captions
    if not available_subs:
        return None

    if lang and lang in available_subs:
        target_lang = lang
    elif "en" in available_subs:
        target_lang = "en"
    elif "zh" in available_subs or "zh-Hans" in available_subs:
        target_lang = "zh" if "zh" in available_subs else "zh-Hans"
    else:
        target_lang = list(available_subs.keys())[0]

    is_auto = not subtitles or target_lang not in subtitles
    sub_type = "自动生成" if is_auto else "人工"
    print(f"发现{sub_type}字幕 ({target_lang})，正在下载...")

    output_template = str(output_dir / f"{safe_title}.%(ext)s")
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": not is_auto,
        "writeautomaticsub": is_auto,
        "subtitleslangs": [target_lang],
        "subtitlesformat": "vtt/srt/best",
        "outtmpl": output_template,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for ext in ["vtt", "srt"]:
        sub_path = output_dir / f"{safe_title}.{target_lang}.{ext}"
        if sub_path.exists():
            print(f"字幕已下载: {sub_path}")
            return sub_path

    return None


def parse_subtitle_to_chunks(subtitle_path: Path) -> list[dict]:
    """将VTT/SRT字幕文件解析为chunks格式

    Args:
        subtitle_path: 字幕文件路径

    Returns:
        chunks列表，每个chunk包含text、start、end字段
    """
    import re

    content = subtitle_path.read_text(encoding="utf-8")
    chunks = []

    if subtitle_path.suffix == ".vtt":
        content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>", "", content)
        time_pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"
    else:
        time_pattern = r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"

    def time_to_seconds(time_str: str) -> float:
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        time_match = None
        text_lines = []

        for line in lines:
            match = re.match(time_pattern, line)
            if match:
                time_match = match
            elif time_match and line.strip() and not re.match(r"^\d+$", line.strip()):
                text_lines.append(line.strip())

        if time_match and text_lines:
            text = " ".join(text_lines)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append({
                    "text": text,
                    "start": time_to_seconds(time_match.group(1)),
                    "end": time_to_seconds(time_match.group(2)),
                })

    merged = []
    for chunk in chunks:
        if merged and chunk["text"] == merged[-1]["text"]:
            merged[-1]["end"] = chunk["end"]
        else:
            merged.append(chunk)

    return merged


def download_youtube_video(url: str, output_dir: Path) -> Path:
    """下载YouTube视频（最高画质），返回视频文件路径

    使用yt-dlp下载器，自动选择最佳视频和音频轨道并合并为MP4格式。
    如果目标文件已存在，则跳过下载以节省时间和带宽。

    Args:
        url: YouTube视频URL
        output_dir: 视频保存目录

    Returns:
        下载完成的视频文件路径
    """
    # 确保输出目录存在，parents=True表示递归创建父目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 第一阶段：获取视频元信息（不下载）
    with yt_dlp.YoutubeDL({"skip_download": True}) as ydl:
        # extract_info获取视频信息，download=False表示只获取信息不下载
        info = ydl.extract_info(url, download=False)
        # 将视频标题转换为安全的文件名
        safe_title = sanitize_filename(info.get("title", "video"))

    # 构建目标视频文件路径
    video_path = output_dir / f"{safe_title}.mp4"

    # 检查视频是否已存在，存在则跳过下载
    if video_path.exists():
        print(f"视频已存在，跳过下载: {video_path}")
        return video_path

    # 第二阶段：实际下载视频
    # 输出模板：使用安全标题，%(ext)s会被yt-dlp替换为实际扩展名
    output_template = str(output_dir / f"{safe_title}.%(ext)s")
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",  # 选择最佳视频+最佳音频，或单个最佳文件
        "outtmpl": output_template,             # 输出文件名模板
        "merge_output_format": "mp4",           # 合并后的输出格式
    }

    # 使用配置好的选项创建下载器并下载
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"视频已下载: {video_path}")
    return video_path


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """从视频文件中提取音频轨道

    使用FFmpeg提取音频并转换为ASR模型所需的格式：
    - 格式：WAV（无损）
    - 采样率：16kHz（语音识别标准）
    - 声道：单声道
    - 编码：PCM 16位有符号整数

    Args:
        video_path: 输入视频文件路径
        output_dir: 音频输出目录

    Returns:
        提取的音频文件路径
    """
    # 构建输出音频文件路径，使用视频同名但.wav扩展名
    audio_path = output_dir / f"{video_path.stem}.wav"

    # 检查音频是否已存在，存在则跳过提取
    if audio_path.exists():
        print(f"音频已存在，跳过提取: {audio_path}")
        return audio_path

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建FFmpeg命令
    cmd = [
        "ffmpeg",           # FFmpeg可执行文件
        "-y",               # 覆盖输出文件（不询问）
        "-i", str(video_path),  # 输入文件
        "-vn",              # 禁用视频流（只处理音频）
        "-acodec", "pcm_s16le",  # 音频编码：PCM 16位小端
        "-ar", "16000",     # 采样率：16kHz
        "-ac", "1",         # 声道数：1（单声道）
        str(audio_path)     # 输出文件
    ]
    # 执行命令，check=True表示命令失败时抛出异常，capture_output隐藏输出
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"音频已提取: {audio_path}")
    return audio_path


def load_whisper_pipeline(batch_size: int = 1, model_key: str = "whisper"):
    """加载Whisper语音识别模型并创建推理管道

    加载OpenAI Whisper模型，配置数据类型和设备，
    并构建HuggingFace pipeline用于推理。
    支持标准模型和LoRA微调模型。

    Args:
        batch_size: 批处理大小，增大可加速但需要更多显存
        model_key: 模型键名，支持 'whisper'

    Returns:
        配置好的HuggingFace ASR pipeline对象
    """
    # 获取Whisper模型路径
    model_path = str(ASR_MODELS["whisper"])

    # 加载预训练的Whisper模型
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,                    # 模型路径
        torch_dtype=torch_dtype,       # 数据类型
        low_cpu_mem_usage=True,        # 低CPU内存模式，加快加载速度
        use_safetensors=True,          # 使用safetensors格式（更安全）
    )
    # 清除强制解码器ID，允许自动语言检测
    model.config.forced_decoder_ids = None
    # 将模型移动到指定设备（MPS或CPU）
    model.to(device)

    # 加载处理器（包含tokenizer和feature_extractor）
    processor = AutoProcessor.from_pretrained(model_path)

    # 创建并返回ASR pipeline
    return pipeline(
        "automatic-speech-recognition",   # 任务类型：自动语音识别
        model=model,                       # 加载的模型
        tokenizer=processor.tokenizer,     # 分词器
        feature_extractor=processor.feature_extractor,  # 特征提取器
        torch_dtype=torch_dtype,           # 数据类型
        device=device,                     # 计算设备
        batch_size=batch_size,             # 批处理大小
    )


def load_funasr_model(model_name: str):
    """加载FunASR语音识别模型

    FunASR是阿里达摩院开源的ASR工具包，支持中英日等多种语言。
    需要特殊处理以解决模块导入问题。

    Args:
        model_name: 模型名称，'funasr-nano'或'funasr-mlt'

    Returns:
        加载好的FunASR模型对象
    """
    import sys  # 系统模块，用于操作模块路径

    from funasr import AutoModel  # FunASR自动模型加载器
    from funasr.models.ctc import ctc  # CTC解码模块

    # 获取模型路径
    model_path = ASR_MODELS[model_name]
    # 将ctc模块注册到sys.modules，解决导入问题
    sys.modules["ctc"] = ctc
    # 将模型路径添加到Python搜索路径
    sys.path.insert(0, str(model_path))
    try:
        # 导入模型定义（仅为触发注册，不实际使用）
        import model as _  # noqa: F401 忽略"导入但未使用"警告
    finally:
        # 恢复原始搜索路径
        sys.path.pop(0)

    # 创建并返回FunASR模型
    model = AutoModel(
        model=str(model_path),      # 模型路径
        trust_remote_code=True,     # 信任模型中的自定义代码
        device=device,              # 计算设备
        disable_update=True,        # 禁用自动更新检查
    )
    return model


def transcribe_audio_funasr(
    model,
    audio_path: Path,
    language: str | None = None,
    batch_size_s: int = 60,
    merge_length_s: int = 15,
    use_vad: bool = False,
    refine_timestamps: bool = False,
) -> list[dict]:
    """使用FunASR模型识别音频

    调用FunASR模型进行语音识别，返回带时间戳的文本片段列表。
    支持VAD预分片和stable-ts时间戳精调。

    Args:
        model: 已加载的FunASR模型
        audio_path: 音频文件路径
        language: 指定语言代码（zh/en/ja/ko/yue），None表示自动检测
        batch_size_s: 批处理时长（秒）
        merge_length_s: 合并片段的最大时长（秒）
        use_vad: 是否使用Silero VAD预分片
        refine_timestamps: 是否使用stable-ts精调时间戳

    Returns:
        识别结果列表，每个元素包含start/end/text字段
    """
    import tempfile

    import torchaudio

    # 语言代码映射：将ISO语言代码转换为FunASR使用的中文名称
    lang_map = {
        "zh": "中文",
        "en": "英文",
        "ja": "日文",
        "ko": "韩文",
        "yue": "粤语",
    }
    lang = lang_map.get(language, language) if language else None

    chunks = []

    if use_vad:
        # 使用VAD预分片
        print("使用 Silero VAD 预分片...")
        vad_segments = detect_speech_segments_silero(audio_path)

        if not vad_segments:
            print("VAD 未检测到语音片段，使用整段音频")
            vad_segments = [(0, None)]  # None表示到音频结尾

        # 加载完整音频
        waveform, sample_rate = torchaudio.load(str(audio_path))

        for seg_idx, (seg_start, seg_end) in enumerate(vad_segments):
            print(f"处理片段 {seg_idx + 1}/{len(vad_segments)}: {seg_start:.2f}s - {seg_end:.2f}s" if seg_end else f"处理片段 {seg_idx + 1}/{len(vad_segments)}: {seg_start:.2f}s - 结尾")

            # 截取音频片段
            start_sample = int(seg_start * sample_rate)
            end_sample = int(seg_end * sample_rate) if seg_end else waveform.shape[1]
            segment_waveform = waveform[:, start_sample:end_sample]

            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            torchaudio.save(tmp_path, segment_waveform, sample_rate)

            try:
                # 对片段进行识别
                result = model.generate(
                    input=tmp_path,
                    language=lang,
                    use_itn=True,
                    batch_size_s=batch_size_s,
                    merge_length_s=merge_length_s,
                )

                # 解析并调整时间戳
                if result and len(result) > 0:
                    res = result[0]
                    if "sentence_info" in res:
                        for sent in res["sentence_info"]:
                            start_ms = sent.get("start", 0)
                            end_ms = sent.get("end", start_ms + 3000)
                            text = sent.get("text", "").strip()
                            if text:
                                chunks.append({
                                    "start": seg_start + start_ms / 1000.0,
                                    "end": seg_start + end_ms / 1000.0,
                                    "text": text,
                                })
                    elif "text" in res:
                        text = res["text"].strip()
                        if text:
                            chunks.append({
                                "start": seg_start,
                                "end": seg_end if seg_end else seg_start + 10.0,
                                "text": text,
                            })
            finally:
                # 清理临时文件
                Path(tmp_path).unlink(missing_ok=True)
    else:
        # 不使用VAD，直接识别整段音频
        result = model.generate(
            input=str(audio_path),
            language=lang,
            use_itn=True,
            batch_size_s=batch_size_s,
            merge_length_s=merge_length_s,
        )

        if result and len(result) > 0:
            res = result[0]
            if "sentence_info" in res:
                for sent in res["sentence_info"]:
                    start_ms = sent.get("start", 0)
                    end_ms = sent.get("end", start_ms + 3000)
                    text = sent.get("text", "").strip()
                    if text:
                        chunks.append({
                            "start": start_ms / 1000.0,
                            "end": end_ms / 1000.0,
                            "text": text,
                        })
            elif "text" in res:
                chunks.append({
                    "start": 0.0,
                    "end": 10.0,
                    "text": res["text"].strip(),
                })

    print(f"FunASR 识别完成，共 {len(chunks)} 个片段")

    # stable-ts 时间戳精调
    if refine_timestamps and chunks:
        chunks = refine_timestamps_with_stable_ts(audio_path, chunks)

    return chunks


def refine_timestamps_with_stable_ts(
    audio_path: Path,
    chunks: list[dict],
) -> list[dict]:
    """使用stable-ts精调时间戳

    利用stable-ts的对齐功能，根据音频波形优化时间戳精度。

    Args:
        audio_path: 音频文件路径
        chunks: 原始识别结果

    Returns:
        时间戳精调后的识别结果
    """
    import warnings

    import stable_whisper

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("使用 stable-ts 精调时间戳...")

    try:
        # 加载Whisper模型用于对齐
        model = stable_whisper.load_model("base", device=device)

        # 合并所有文本
        full_text = " ".join(c["text"] for c in chunks)

        # 使用stable-ts进行对齐
        result = model.align(str(audio_path), full_text)

        # 提取精调后的时间戳
        refined_chunks = []
        for segment in result.segments:
            refined_chunks.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })

        if refined_chunks:
            print(f"stable-ts 精调完成，{len(chunks)} -> {len(refined_chunks)} 个片段")
            return refined_chunks
        else:
            print("stable-ts 精调未产生结果，保留原始时间戳")
            return chunks

    except Exception as e:
        print(f"stable-ts 精调失败: {e}，保留原始时间戳")
        return chunks


# ============================================================
# Silero VAD 全局缓存（单例模式）
# ============================================================
_silero_vad_model = None   # 缓存的VAD模型
_silero_vad_utils = None   # 缓存的VAD工具函数


def _load_silero_vad():
    """懒加载Silero VAD（语音活动检测）模型

    使用torch.hub从GitHub加载Silero VAD模型。
    采用全局缓存避免重复加载，提高效率。

    Returns:
        tuple: (VAD模型, 工具函数集合)
    """
    global _silero_vad_model, _silero_vad_utils
    # 检查是否已加载，避免重复加载
    if _silero_vad_model is None:
        # 从torch.hub加载Silero VAD
        _silero_vad_model, _silero_vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",  # GitHub仓库
            model="silero_vad",                   # 模型名称
            force_reload=False,                   # 不强制重新下载
            trust_repo=True,                      # 信任仓库代码
        )
    return _silero_vad_model, _silero_vad_utils


def detect_speech_segments_silero(
    audio_path: Path,
    min_speech_duration: float = 0.25,
    min_silence_duration: float = 0.3,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
) -> list[tuple[float, float]]:
    """使用Silero VAD检测音频中的语音片段

    VAD（Voice Activity Detection）用于检测音频中哪些部分包含语音。
    比传统的能量阈值方法更准确，可以有效过滤背景噪音。

    Args:
        audio_path: 音频文件路径
        min_speech_duration: 最小语音持续时间（秒），短于此的语音会被忽略
        min_silence_duration: 最小静音持续时间（秒），用于分割语音片段
        threshold: VAD置信度阈值（0-1），越高越严格
        sampling_rate: 采样率（Hz），必须与音频匹配

    Returns:
        语音片段列表，每个元素为(开始时间, 结束时间)元组
    """
    # 加载VAD模型和工具函数
    model, utils = _load_silero_vad()
    # 解包工具函数：获取时间戳函数、（跳过）、读取音频函数
    get_speech_timestamps, _, read_audio, *_ = utils

    # 读取音频文件
    wav = read_audio(str(audio_path), sampling_rate=sampling_rate)

    # 使用VAD模型检测语音时间戳
    speech_timestamps = get_speech_timestamps(
        wav,                                               # 音频波形
        model,                                             # VAD模型
        threshold=threshold,                               # 置信度阈值
        min_speech_duration_ms=int(min_speech_duration * 1000),  # 最小语音时长（毫秒）
        min_silence_duration_ms=int(min_silence_duration * 1000),  # 最小静音时长（毫秒）
        sampling_rate=sampling_rate,                       # 采样率
        return_seconds=True,                               # 返回秒而不是采样点
    )

    # 将时间戳字典转换为元组列表
    segments = [(ts["start"], ts["end"]) for ts in speech_timestamps]

    # 合并相邻的语音片段（间隔小于0.5秒的合并）
    merged = []
    for seg in segments:
        # 如果merged非空且当前片段与上一个片段间隔小于0.5秒
        if merged and seg[0] - merged[-1][1] < 0.5:
            # 扩展上一个片段的结束时间
            merged[-1] = (merged[-1][0], seg[1])
        else:
            # 添加为新片段
            merged.append(seg)

    print(f"Silero VAD 检测到 {len(merged)} 个语音片段")
    return merged


def transcribe_with_stable_ts(
    audio_path: Path,
    language: str | None = None,
) -> list[dict]:
    """使用stable-ts进行高精度语音识别

    stable-ts是Whisper的增强版本，通过VAD和静音分析提供更精确的时间戳。

    Args:
        audio_path: 音频文件路径
        language: 指定语言代码，None表示自动检测

    Returns:
        带精确时间戳的识别结果列表
    """
    import warnings  # 警告控制模块

    import stable_whisper  # stable-ts库

    # 抑制各种无关紧要的警告信息
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
    warnings.filterwarnings("ignore", message=".*attention mask.*")
    warnings.filterwarnings("ignore", message=".*WhisperSdpaAttention.*")

    print("正在使用 stable-ts 优化时间戳...")
    # CPU模式下推理会很慢，给用户提示
    if device == "cpu":
        print("⚠️  警告: CPU 模式下推理较慢，请耐心等待...")

    # 加载HuggingFace格式的Whisper模型
    model = stable_whisper.load_hf_whisper(
        str(ASR_MODELS["whisper"]),  # 模型路径
        device=device,                # 计算设备
    )

    # 清除强制解码器ID，允许自动语言检测
    model._pipe.model.config.forced_decoder_ids = None

    print("开始转录音频（可能需要几分钟）...")
    # 使用stable-ts进行转录
    result = model.transcribe(
        str(audio_path),       # 音频路径
        language=language,     # 指定语言
        vad=True,              # 启用VAD预处理
        suppress_silence=True,  # 抑制静音部分
        word_timestamps=True,   # 生成词级时间戳
        verbose=True,           # 显示详细进度
    )

    print("正在调整时间戳...")
    # 基于静音分析调整时间戳边界
    result.adjust_by_silence(
        audio_path,     # 音频文件
        q_levels=20,    # 量化级别，越高越精细
        k_size=5,       # 平滑窗口大小
    )

    # 提取优化后的片段
    refined_chunks = []
    for segment in result.segments:
        text = segment.text.strip()
        if text:  # 只保留非空片段
            refined_chunks.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })

    print(f"stable-ts 优化完成，共 {len(refined_chunks)} 个片段")
    return refined_chunks


def transcribe_audio(pipe, audio_path: Path, language: str | None = None, use_vad: bool = True) -> list[dict]:
    """使用Whisper模型识别音频

    支持两种模式：
    1. VAD模式（默认）：先用Silero VAD检测语音片段，再逐段识别
    2. 原生模式：使用Whisper内置的分块策略

    VAD模式通常更准确，但原生模式在某些情况下可能更快。

    Args:
        pipe: Whisper推理管道
        audio_path: 音频文件路径
        language: 指定语言代码，None表示自动检测
        use_vad: 是否使用Silero VAD预处理

    Returns:
        识别结果列表，每个元素包含start/end/text字段
    """
    import librosa  # 音频处理库

    # 配置生成参数
    generate_kwargs = {"task": "transcribe"}  # 任务类型：转录
    if language:
        generate_kwargs["language"] = language  # 指定源语言

    # ========== VAD模式 ==========
    if use_vad:
        import soundfile as sf
        import tempfile

        # 使用Silero VAD检测语音片段
        speech_segments = detect_speech_segments_silero(audio_path)
        if not speech_segments:
            print("未检测到语音片段")
            return []

        # 使用librosa加载完整音频
        # sr=16000：采样率16kHz，mono=True：转为单声道
        waveform, _ = librosa.load(str(audio_path), sr=16000, mono=True)

        chunks = []  # 存储所有识别结果
        # 遍历每个语音片段
        for seg_start, seg_end in speech_segments:
            # 计算采样点范围
            start_sample = int(seg_start * 16000)  # 开始采样点
            end_sample = int(seg_end * 16000)      # 结束采样点
            # 截取对应的音频片段
            segment_audio = waveform[start_sample:end_sample]

            # transformers 5.x 在 return_timestamps=True 时有 num_frames 问题
            # 需要保存为临时文件再传入
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            # 在 with 块外写入，确保文件句柄已关闭
            sf.write(tmp_path, segment_audio, 16000)

            try:
                # 对该片段进行识别
                result = pipe(
                    tmp_path,
                    return_timestamps=True,         # 返回时间戳
                    generate_kwargs=generate_kwargs, # 生成参数
                )

                # 解析识别结果
                for chunk in result.get("chunks", []):
                    ts = chunk["timestamp"]  # 获取时间戳元组
                    # 跳过无效时间戳
                    if ts[0] is None or ts[1] is None:
                        continue
                    text = chunk["text"].strip()
                    if text:
                        chunks.append({
                            # 注意：时间戳是相对于片段开始的，需要加上片段偏移
                            "start": seg_start + ts[0],
                            "end": seg_start + ts[1],
                            "text": text,
                        })
            finally:
                # 清理临时文件
                Path(tmp_path).unlink(missing_ok=True)

        print(f"识别完成，共 {len(chunks)} 个片段")
        return chunks

    # ========== 原生Whisper模式 ==========
    # 直接对完整音频进行识别
    result = pipe(
        str(audio_path),             # 音频路径
        return_timestamps=True,       # 返回时间戳
        chunk_length_s=15,            # 分块长度（秒）
        stride_length_s=5,            # 分块重叠（秒）
        generate_kwargs=generate_kwargs,
    )

    chunks = []
    prev_end = 0.0  # 上一个片段的结束时间，用于处理None值
    for i, chunk in enumerate(result.get("chunks", [])):
        start, end = chunk["timestamp"]
        # 处理None开始时间：使用上一片段的结束时间
        if start is None:
            start = prev_end
        # 处理None结束时间
        if end is None:
            # 尝试使用下一片段的开始时间
            next_chunks = result.get("chunks", [])[i + 1:]
            if next_chunks and next_chunks[0]["timestamp"][0] is not None:
                end = next_chunks[0]["timestamp"][0]
            else:
                # 否则假设3秒时长
                end = start + 3.0
        text = chunk["text"].strip()
        if text:
            chunks.append({
                "start": start,
                "end": end,
                "text": text,
            })
        prev_end = end  # 更新上一片段结束时间

    print(f"识别完成，共 {len(chunks)} 个片段")
    return chunks


def save_chunks(chunks: list[dict], path: Path) -> None:
    """将识别结果保存到JSON文件

    用于缓存识别结果，避免重复进行耗时的ASR处理。

    Args:
        chunks: 识别结果列表
        path: 保存路径
    """
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入JSON文件，ensure_ascii=False保留中文，indent=2美化格式
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chunks 已保存: {path}")


def load_chunks(path: Path) -> list[dict] | None:
    """从JSON文件加载识别结果

    用于恢复之前保存的识别结果，避免重复处理。

    Args:
        path: JSON文件路径

    Returns:
        识别结果列表，文件不存在时返回None
    """
    if path.exists():
        print(f"加载已有 chunks: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def translate_with_ollama(text: str, target_lang: str, model: str, max_retries: int = 3) -> str:
    """使用Ollama本地/云端模型翻译单条文本

    调用Ollama API进行翻译，支持自动重试以应对临时错误。

    Args:
        text: 待翻译文本
        target_lang: 目标语言（如"中文"、"English"）
        model: Ollama模型名称
        max_retries: 最大重试次数

    Returns:
        翻译后的文本

    Raises:
        httpx.HTTPStatusError: 多次重试后仍失败
    """
    # 构建翻译提示词，明确要求只输出翻译结果
    prompt = f"将以下文本翻译成{target_lang}，只输出翻译结果，不要任何解释：\n\n{text}"

    import time  # 时间模块，用于重试间隔
    # 重试循环
    for attempt in range(max_retries):
        try:
            # 发送POST请求到Ollama API
            response = httpx.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False},  # stream=False获取完整响应
                timeout=60.0,  # 60秒超时
            )
            response.raise_for_status()  # 检查HTTP状态码
            return response.json()["response"].strip()  # 提取并返回翻译结果
        except httpx.HTTPStatusError as e:
            # 如果是500错误且还有重试次数，等待后重试
            if e.response.status_code == 500 and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避：1, 2, 4秒
                print(f"服务器错误，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise  # 其他错误或重试耗尽，抛出异常


def translate_chunks(
    chunks: list[dict], target_lang: str, model: str
) -> list[dict]:
    """使用Ollama逐条翻译所有文本片段

    同步顺序翻译，适用于Ollama本地服务。
    包含进度显示和错误处理。

    Args:
        chunks: 识别结果列表
        target_lang: 目标语言
        model: Ollama模型名称

    Returns:
        添加了translated字段的chunks列表
    """
    import time

    total = len(chunks)  # 总片段数
    for i, chunk in enumerate(chunks):
        # 跳过已翻译的片段
        if chunk.get("translated"):
            continue

        # 带重试的翻译
        for attempt in range(3):
            try:
                chunk["translated"] = translate_with_ollama(
                    chunk["text"], target_lang, model
                )
                break  # 成功则跳出重试循环
            except Exception as e:
                if attempt < 2:
                    wait_time = 2 ** attempt
                    print(f"翻译失败，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    # 重试耗尽，使用原文作为翻译
                    print(f"翻译失败，使用原文: {e}")
                    chunk["translated"] = chunk["text"]

        # 每10条或最后一条显示进度
        if (i + 1) % 10 == 0 or i + 1 == total:
            print(f"翻译进度: {i + 1}/{total}")
        # 请求间隔，避免过于频繁
        time.sleep(0.3)

    print(f"翻译完成，模型: {model}，目标语言: {target_lang}")
    return chunks


# ============================================================
# 智能分段处理
# ============================================================

# 标点符号分类
STRONG_PUNCTUATION = set("。.？?！!；;")  # 强标点，必断
WEAK_PUNCTUATION = set("，,：:、")  # 弱标点，超长时断
SEMANTIC_CONNECTORS = {"但是", "然后", "所以", "因此", "不过", "而且", "并且", "或者", "如果", "那么"}

# 细粒度分割参数
MAX_CHUNK_DURATION = 4.0  # 单个字幕最大持续时间（秒）


def split_text_smartly(
    text: str,
    max_chars: int = 40,
    max_lines: int = 2,
) -> list[str]:
    """智能分割文本为多个字幕块

    分割优先级：
    1. 强标点（句号、问号、感叹号）必断
    2. 弱标点（逗号、分号）超过max_chars时断
    3. 语义边界（连词）无标点且超长时断
    4. 强制截断：超过max_chars*1.5时硬切

    Args:
        text: 原始文本
        max_chars: 每行最大字符数
        max_lines: 每屏最大行数

    Returns:
        分割后的文本块列表
    """
    if not text or len(text) <= max_chars * max_lines:
        return [text] if text else []

    blocks = []
    current_block = ""
    current_line = ""
    line_count = 0

    i = 0
    while i < len(text):
        char = text[i]
        current_line += char

        # 检查是否在强标点处断开
        if char in STRONG_PUNCTUATION:
            if line_count < max_lines - 1 and len(current_block) + len(current_line) <= max_chars * max_lines:
                current_block += current_line
                line_count += 1
            else:
                if current_block:
                    blocks.append(current_block.strip())
                current_block = current_line
                line_count = 1
            current_line = ""

        # 检查是否超过单行长度限制
        elif len(current_line) >= max_chars:
            # 尝试在弱标点处断开
            break_pos = -1
            for j in range(len(current_line) - 1, max(0, len(current_line) - 15), -1):
                if current_line[j] in WEAK_PUNCTUATION:
                    break_pos = j + 1
                    break

            # 尝试在空格处断开
            if break_pos == -1:
                for j in range(len(current_line) - 1, max(0, len(current_line) - 15), -1):
                    if current_line[j] == " ":
                        break_pos = j + 1
                        break

            # 尝试在语义连接词处断开
            if break_pos == -1:
                for connector in SEMANTIC_CONNECTORS:
                    pos = current_line.rfind(connector, max(0, len(current_line) - 20))
                    if pos > 0:
                        break_pos = pos
                        break

            # 强制截断
            if break_pos == -1:
                break_pos = max_chars

            # 执行断开
            segment = current_line[:break_pos]
            remaining = current_line[break_pos:]

            if line_count < max_lines - 1:
                current_block += segment + "\n"
                line_count += 1
            else:
                if current_block:
                    current_block += segment
                    blocks.append(current_block.strip())
                else:
                    blocks.append(segment.strip())
                current_block = ""
                line_count = 0

            current_line = remaining

        i += 1

    # 处理剩余内容
    if current_line:
        current_block += current_line
    if current_block:
        blocks.append(current_block.strip())

    return blocks


def segment_chunks_smartly(
    chunks: list[dict],
    max_chars: int = 40,
    max_lines: int = 2,
    max_duration: float = MAX_CHUNK_DURATION,
) -> list[dict]:
    """智能分段处理识别结果

    对ASR输出的chunks进行智能分段，优化字幕显示效果。
    同时基于字符数和时长进行分割，确保字幕粒度足够细。
    时间戳按字符比例分配。

    Args:
        chunks: 原始识别结果列表
        max_chars: 每行最大字符数
        max_lines: 每屏最大行数
        max_duration: 单个字幕最大持续时间（秒）

    Returns:
        分段后的识别结果列表
    """
    result = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        start = chunk.get("start", 0)
        end = chunk.get("end", start + 1)
        duration = end - start

        # 计算基于时长需要的最小分段数
        min_splits_by_duration = max(1, int(duration / max_duration + 0.5))
        
        # 计算基于字符数需要的最小分段数
        max_chars_per_block = max_chars * max_lines
        min_splits_by_chars = max(1, int(len(text) / max_chars_per_block + 0.5))
        
        # 取两者较大值
        target_splits = max(min_splits_by_duration, min_splits_by_chars)

        # 智能分割文本
        if target_splits == 1 and len(text) <= max_chars_per_block:
            result.append(chunk)
            continue
            
        # 调整max_chars以产生足够的分段
        adjusted_max_chars = max(10, len(text) // target_splits // max_lines)
        blocks = split_text_smartly(text, min(max_chars, adjusted_max_chars), max_lines)

        if len(blocks) <= 1:
            # 如果分割函数没能分割，但时长过长，强制均匀分割
            if duration > max_duration:
                blocks = force_split_text(text, target_splits)
            else:
                result.append(chunk)
                continue

        # 按字符比例分配时间
        total_chars = sum(len(b) for b in blocks)
        current_time = start

        for block in blocks:
            block_duration = duration * len(block) / total_chars if total_chars > 0 else duration / len(blocks)
            block_end = min(current_time + block_duration, end)

            new_chunk = chunk.copy()
            new_chunk["text"] = block
            new_chunk["start"] = current_time
            new_chunk["end"] = block_end

            # 如果有翻译，按比例分配翻译文本
            if "translated" in chunk:
                new_chunk["translated"] = chunk["translated"]

            result.append(new_chunk)
            current_time = block_end

    print(f"智能分段完成: {len(chunks)} -> {len(result)} 个片段")
    return result


def force_split_text(text: str, num_splits: int) -> list[str]:
    """强制将文本均匀分割为指定数量的块
    
    在标点符号和空格处优先分割，否则在字符边界分割。
    
    Args:
        text: 待分割文本
        num_splits: 目标分割数量
        
    Returns:
        分割后的文本块列表
    """
    if num_splits <= 1:
        return [text]
    
    target_len = len(text) // num_splits
    blocks = []
    current_pos = 0
    
    for i in range(num_splits - 1):
        end_pos = current_pos + target_len
        if end_pos >= len(text):
            break
            
        # 在目标位置附近寻找最佳分割点
        best_pos = end_pos
        search_range = min(15, target_len // 2)
        
        # 优先在强标点处分割
        for j in range(end_pos, max(current_pos, end_pos - search_range), -1):
            if j < len(text) and text[j] in STRONG_PUNCTUATION:
                best_pos = j + 1
                break
        else:
            # 次选在弱标点处分割
            for j in range(end_pos, max(current_pos, end_pos - search_range), -1):
                if j < len(text) and text[j] in WEAK_PUNCTUATION:
                    best_pos = j + 1
                    break
            else:
                # 再次选在空格处分割
                for j in range(end_pos, max(current_pos, end_pos - search_range), -1):
                    if j < len(text) and text[j] == " ":
                        best_pos = j + 1
                        break
        
        block = text[current_pos:best_pos].strip()
        if block:
            blocks.append(block)
        current_pos = best_pos
    
    # 添加剩余部分
    remaining = text[current_pos:].strip()
    if remaining:
        blocks.append(remaining)
    
    return blocks if blocks else [text]


def generate_ass_subtitle(chunks: list[dict], output_path: Path) -> Path:
    """生成ASS格式双语字幕文件

    ASS（Advanced SubStation Alpha）是一种功能丰富的字幕格式，
    支持多种样式、定位和特效。本函数生成包含原文和译文的双语字幕。

    Args:
        chunks: 包含text和translated字段的片段列表
        output_path: 输出文件路径

    Returns:
        生成的字幕文件路径
    """
    # ASS文件头部模板
    # [Script Info]: 脚本元信息
    # [V4+ Styles]: 样式定义
    # [Events]: 字幕事件（对话）
    ass_header = """[Script Info]
Title: Bilingual Subtitles
ScriptType: v4.00+
PlayDepth: 0
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Original,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,20,20,60,1
Style: Translated,Arial,42,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # 样式说明：
    # Original: 原文样式，白色(FFFFFF)，48号字，距底部60像素
    # Translated: 译文样式，青色(00FFFF)，42号字，距底部20像素

    def format_time(seconds: float) -> str:
        """将秒数转换为ASS时间格式 H:MM:SS.cc

        Args:
            seconds: 时间（秒）

        Returns:
            ASS格式时间字符串，如"0:01:23.45"
        """
        h = int(seconds // 3600)         # 小时
        m = int((seconds % 3600) // 60)  # 分钟
        s = seconds % 60                  # 秒（含小数）
        return f"{h}:{m:02d}:{s:05.2f}"  # 格式化：分钟和秒补零

    lines = [ass_header]  # 从头部开始
    for chunk in chunks:
        start = format_time(chunk["start"])
        end = format_time(chunk["end"])
        # 将换行符替换为ASS的换行标记\N
        original = chunk["text"].replace("\n", "\\N")
        translated = chunk.get("translated", "").replace("\n", "\\N")

        # 添加原文对话行
        lines.append(f"Dialogue: 0,{start},{end},Original,,0,0,0,,{original}")
        # 添加译文对话行
        lines.append(f"Dialogue: 0,{start},{end},Translated,,0,0,0,,{translated}")

    # 写入文件
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"字幕已生成: {output_path}")
    return output_path


def generate_srt_subtitle(chunks: list[dict], output_path: Path) -> Path:
    """生成SRT格式字幕文件

    SRT是最通用的字幕格式，几乎所有播放器都支持。

    Args:
        chunks: 包含text和可选translated字段的片段列表
        output_path: 输出文件路径

    Returns:
        生成的字幕文件路径
    """
    def format_time_srt(seconds: float) -> str:
        """将秒数转换为SRT时间格式 HH:MM:SS,mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, chunk in enumerate(chunks, 1):
        start = format_time_srt(chunk["start"])
        end = format_time_srt(chunk["end"])
        text = chunk["text"]
        translated = chunk.get("translated", "")

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        if translated:
            lines.append(f"{text}\n{translated}")
        else:
            lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"SRT字幕已生成: {output_path}")
    return output_path


def generate_vtt_subtitle(chunks: list[dict], output_path: Path) -> Path:
    """生成WebVTT格式字幕文件

    WebVTT是HTML5视频的标准字幕格式，支持样式和定位。

    Args:
        chunks: 包含text和可选translated字段的片段列表
        output_path: 输出文件路径

    Returns:
        生成的字幕文件路径
    """
    def format_time_vtt(seconds: float) -> str:
        """将秒数转换为VTT时间格式 HH:MM:SS.mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for i, chunk in enumerate(chunks, 1):
        start = format_time_vtt(chunk["start"])
        end = format_time_vtt(chunk["end"])
        text = chunk["text"]
        translated = chunk.get("translated", "")

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        if translated:
            lines.append(f"{text}\n{translated}")
        else:
            lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"VTT字幕已生成: {output_path}")
    return output_path


def generate_json_subtitle(chunks: list[dict], output_path: Path) -> Path:
    """生成JSON格式字幕文件

    JSON格式便于程序处理和二次开发。

    Args:
        chunks: 识别结果列表
        output_path: 输出文件路径

    Returns:
        生成的字幕文件路径
    """
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"JSON字幕已生成: {output_path}")
    return output_path


def generate_subtitles(
    chunks: list[dict],
    output_base: Path,
    formats: str = "ass",
) -> list[Path]:
    """根据指定格式生成字幕文件

    Args:
        chunks: 识别结果列表
        output_base: 输出文件基础路径（不含扩展名）
        formats: 逗号分隔的格式列表，如 "srt,vtt,ass,json"

    Returns:
        生成的字幕文件路径列表
    """
    format_list = [f.strip().lower() for f in formats.split(",")]
    output_paths = []

    generators = {
        "ass": generate_ass_subtitle,
        "srt": generate_srt_subtitle,
        "vtt": generate_vtt_subtitle,
        "json": generate_json_subtitle,
    }

    for fmt in format_list:
        if fmt in generators:
            output_path = output_base.with_suffix(f".{fmt}")
            generators[fmt](chunks, output_path)
            output_paths.append(output_path)
        else:
            print(f"警告: 不支持的字幕格式 '{fmt}'，跳过")

    return output_paths


def burn_subtitles(video_path: Path, subtitle_path: Path, output_path: Path) -> Path:
    """将字幕硬烧录（嵌入）到视频中

    使用FFmpeg的subtitles滤镜将ASS字幕渲染到视频画面上。
    这样生成的视频在任何播放器上都能显示字幕。

    注意：需要处理FFmpeg滤镜的路径转义问题。

    Args:
        video_path: 输入视频路径
        subtitle_path: 字幕文件路径
        output_path: 输出视频路径

    Returns:
        输出视频路径
    """
    import subprocess  # 子进程模块

    # 获取绝对路径字符串
    abs_video_path = str(video_path.absolute())
    abs_subtitle_path = str(subtitle_path.absolute())
    abs_output_path = str(output_path.absolute())

    # ====== FFmpeg路径转义处理 ======
    # FFmpeg的subtitles滤镜对路径中的特殊字符有严格要求：
    # 1. 反斜杠\需要转义（Windows路径问题）
    # 2. 冒号:是滤镜参数分隔符，需要转义
    # 3. 单引号需要特殊处理

    # 将反斜杠替换为正斜杠（FFmpeg在所有平台都支持正斜杠）
    safe_sub_path = abs_subtitle_path.replace("\\", "/")

    # 转义冒号（Windows盘符如C:）和单引号
    safe_sub_path = safe_sub_path.replace(":", "\\:").replace("'", "'\\''")

    # 构建FFmpeg命令
    cmd = [
        "ffmpeg",                      # FFmpeg可执行文件
        "-y",                          # 覆盖输出文件
        "-i", abs_video_path,          # 输入视频
        # 视频滤镜：使用subtitles滤镜烧录字幕
        # filename='...' 格式可以处理包含特殊字符的路径
        "-vf", f"subtitles=filename='{safe_sub_path}'",
        "-c:a", "copy",                # 音频直接复制（不重编码）
        abs_output_path                # 输出文件
    ]

    # 执行命令
    # 注意：不使用shell=True，因为我们传递的是列表而非字符串
    subprocess.run(cmd, check=True)

    print(f"字幕视频已生成: {output_path}")
    return output_path


def main() -> None:
    """主函数：解析命令行参数并执行完整处理流程

    处理流程：
    1. 解析命令行参数
    2. 下载YouTube视频
    3. 提取音频
    4. 语音识别（ASR）
    5. 翻译
    6. 生成字幕
    7. 烧录字幕到视频
    """
    # ====== 创建命令行参数解析器 ======
    parser = argparse.ArgumentParser(
        description="下载YouTube视频，识别语音，翻译并添加双语字幕"
    )
    # 位置参数：YouTube视频URL（必需）
    parser.add_argument("url", help="YouTube视频URL")
    # 可选参数：目标翻译语言
    parser.add_argument(
        "--target-lang", "-t",
        default="zh-CN",
        help="目标翻译语言代码 (默认: zh-CN)",
    )
    # 可选参数：源语言
    parser.add_argument(
        "--source-lang", "-s",
        default=None,
        help="源语言代码，如 en, ja, zh (默认: 自动检测)",
    )
    # 可选参数：下载目录
    parser.add_argument(
        "--download-dir",
        default=str(DOWNLOAD_DIR),
        help=f"视频下载目录 (默认: {DOWNLOAD_DIR})",
    )
    # 可选参数：结果输出目录
    parser.add_argument(
        "--result-dir", "-o",
        default=str(RESULT_DIR),
        help=f"结果输出目录 (默认: {RESULT_DIR})",
    )
    # 可选参数：Ollama翻译模型选择
    parser.add_argument(
        "--model", "-m",
        default=OLLAMA_MODELS[0],
        choices=OLLAMA_MODELS,
        help=f"Ollama翻译模型 (默认: {OLLAMA_MODELS[0]})",
    )
    # 可选参数：ASR模型选择
    parser.add_argument(
        "--asr", "-a",
        default=DEFAULT_ASR_MODEL,
        choices=list(ASR_MODELS.keys()),
        help=f"ASR模型选择 (默认: {DEFAULT_ASR_MODEL})",
    )

    # 可选参数：批处理大小
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Whisper 批处理大小，增大可加速但需更多显存 (默认: 1)",
    )
    # 可选参数：使用stable-ts优化时间戳
    parser.add_argument(
        "--refine",
        action="store_true",
        help="使用 stable-ts 优化时间戳精度 (更准确但更慢)",
    )
    # 可选参数：禁用VAD
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="禁用 Silero VAD，使用原生 Whisper 分段",
    )
    # FunASR 专用参数
    parser.add_argument(
        "--funasr-batch-size",
        type=int,
        default=60,
        help="FunASR 批处理时长（秒） (默认: 60)",
    )
    parser.add_argument(
        "--funasr-merge-length",
        type=int,
        default=15,
        help="FunASR 合并片段最大时长（秒） (默认: 15)",
    )
    parser.add_argument(
        "--funasr-vad",
        action="store_true",
        help="FunASR 启用 Silero VAD 预分片",
    )
    parser.add_argument(
        "--funasr-refine",
        action="store_true",
        help="FunASR 使用 stable-ts 精调时间戳",
    )
    # 字幕格式参数
    parser.add_argument(
        "--max-chars",
        type=int,
        default=40,
        help="字幕每行最大字符数 (默认: 40)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=2,
        help="字幕每屏最大行数 (默认: 2)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_CHUNK_DURATION,
        help=f"单个字幕最大持续时间（秒），超过则强制分割 (默认: {MAX_CHUNK_DURATION})",
    )
    parser.add_argument(
        "--subtitle-format",
        default="ass",
        help="字幕输出格式，逗号分隔多个格式：srt,vtt,ass,json (默认: ass)",
    )
    parser.add_argument(
        "--use-youtube-sub",
        action="store_true",
        help="优先使用YouTube原有字幕（如果存在），跳过ASR",
    )
    parser.add_argument(
        "--no-youtube-sub",
        action="store_true",
        help="强制使用ASR，即使YouTube有字幕",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # ====== 初始化目录 ======
    download_dir = Path(args.download_dir)  # 视频下载目录
    result_dir = Path(args.result_dir)      # 结果输出目录
    audio_dir = AUDIO_DIR                   # 音频目录
    oss_dir = OSS_DIR                       # 字幕目录
    chunks_dir = CHUNKS_DIR                 # 识别缓存目录
    # 创建所有必需的目录
    result_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    oss_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # ====== 步骤1：下载视频 ======
    print("正在下载视频...")
    video_path = download_youtube_video(args.url, download_dir)

    # ====== 步骤2/3：尝试使用YouTube原有字幕或语音识别 ======
    youtube_sub_path = None
    used_youtube_sub = False

    # 构建缓存文件路径（基于视频名和ASR模型）
    chunks_path = chunks_dir / f"{video_path.stem}_{args.asr}.json"
    youtube_chunks_path = chunks_dir / f"{video_path.stem}_youtube.json"

    # 如果启用YouTube字幕优先，先尝试下载
    if args.use_youtube_sub and not args.no_youtube_sub:
        chunks = load_chunks(youtube_chunks_path)
        if chunks is not None:
            print("使用已缓存的YouTube字幕")
            used_youtube_sub = True
            chunks_path = youtube_chunks_path
        else:
            print("正在检查YouTube原有字幕...")
            youtube_sub_path = download_youtube_subtitles(args.url, oss_dir, args.source_lang)
            if youtube_sub_path:
                print("正在解析YouTube字幕...")
                chunks = parse_subtitle_to_chunks(youtube_sub_path)
                if chunks:
                    used_youtube_sub = True
                    chunks_path = youtube_chunks_path
                    save_chunks(chunks, chunks_path)
                    print(f"成功从YouTube字幕获取 {len(chunks)} 个片段")
                else:
                    print("YouTube字幕解析失败，将使用ASR")
            else:
                print("未找到YouTube字幕，将使用ASR")

    # 尝试加载已有的识别结果
    if not used_youtube_sub:
        chunks = load_chunks(chunks_path)

    if chunks is None:
        # 没有缓存，需要进行识别
        print("正在提取音频...")
        audio_path = extract_audio(video_path, audio_dir)

        if args.refine and args.asr == "whisper":
            # 使用stable-ts进行高精度识别
            print("使用 stable-ts 进行高精度识别...")
            chunks = transcribe_with_stable_ts(audio_path, args.source_lang)
        else:
            # 使用标准ASR模型
            print(f"正在加载 {args.asr} 模型...")
            if args.asr == "whisper":
                # 加载Whisper模型
                pipe = load_whisper_pipeline(batch_size=args.batch_size, model_key=args.asr)
                print("正在识别语音...")
                use_vad = not args.no_vad  # 根据参数决定是否使用VAD
                chunks = transcribe_audio(pipe, audio_path, args.source_lang, use_vad=use_vad)
            else:
                model = load_funasr_model(args.asr)
                print("正在识别语音...")
                chunks = transcribe_audio_funasr(
                    model,
                    audio_path,
                    args.source_lang,
                    batch_size_s=args.funasr_batch_size,
                    merge_length_s=args.funasr_merge_length,
                    use_vad=args.funasr_vad,
                    refine_timestamps=args.funasr_refine,
                )
        save_chunks(chunks, chunks_path)

    # ====== 步骤4：翻译 ======
    # 检查是否所有片段都已翻译
    has_translation = all(chunk.get("translated") for chunk in chunks)

    if not has_translation:
        print("正在翻译...")
        chunks = translate_chunks(chunks, args.target_lang, args.model)
        save_chunks(chunks, chunks_path)
    else:
        print("已有翻译，跳过翻译步骤")

    # ====== 步骤5：智能分段 ======
    if args.max_chars or args.max_lines or args.max_duration:
        print("正在进行智能分段...")
        chunks = segment_chunks_smartly(chunks, args.max_chars, args.max_lines, args.max_duration)
        save_chunks(chunks, chunks_path)

    # ====== 步骤6：生成字幕 ======
    subtitle_base = oss_dir / f"{video_path.stem}_{args.asr}"
    subtitle_paths = generate_subtitles(chunks, subtitle_base, args.subtitle_format)

    # 获取ASS字幕路径用于烧录（如果生成了ASS）
    subtitle_path = subtitle_base.with_suffix(".ass")
    if not subtitle_path.exists() and subtitle_paths:
        subtitle_path = subtitle_paths[0]  # 使用第一个生成的字幕

    # ====== 步骤7：烧录字幕 ======
    output_video = result_dir / f"{video_path.stem}_subtitled.mp4"

    if output_video.exists():
        # 输出视频已存在，跳过烧录
        print(f"字幕视频已存在，跳过烧录: {output_video}")
    else:
        # 执行字幕烧录
        burn_subtitles(video_path, subtitle_path, output_video)

    # 输出完成信息
    print(f"\n完成！输出文件: {output_video}")


# 脚本入口点
if __name__ == "__main__":
    main()
