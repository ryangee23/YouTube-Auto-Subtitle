"""Silero VAD 语音活动检测"""

from pathlib import Path

import torch

# Silero VAD 全局缓存（单例模式）
_silero_vad_model = None
_silero_vad_utils = None


def _load_silero_vad():
    """懒加载Silero VAD（语音活动检测）模型

    使用torch.hub从GitHub加载Silero VAD模型。
    采用全局缓存避免重复加载，提高效率。

    Returns:
        tuple: (VAD模型, 工具函数集合)
    """
    global _silero_vad_model, _silero_vad_utils
    if _silero_vad_model is None:
        _silero_vad_model, _silero_vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
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
    model, utils = _load_silero_vad()
    get_speech_timestamps, _, read_audio, *_ = utils

    wav = read_audio(str(audio_path), sampling_rate=sampling_rate)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        min_speech_duration_ms=int(min_speech_duration * 1000),
        min_silence_duration_ms=int(min_silence_duration * 1000),
        sampling_rate=sampling_rate,
        return_seconds=True,
    )

    segments = [(ts["start"], ts["end"]) for ts in speech_timestamps]

    # 合并相邻的语音片段（间隔小于0.5秒的合并）
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < 0.5:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    print(f"Silero VAD 检测到 {len(merged)} 个语音片段")
    return merged
