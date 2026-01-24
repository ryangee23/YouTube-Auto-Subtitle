"""ASR模块：语音识别模型统一接口

支持的模型：
- Whisper Large V3 Turbo
- FunASR Nano (中英日)
- FunASR MLT (31语言)
"""

from .base import ASRChunk, ASRConfig
from .funasr import FunASRModel, load_funasr_model, transcribe_audio_funasr
from .utils import refine_timestamps_with_stable_ts
from .vad import detect_speech_segments_silero
from .whisper import (
    load_whisper_pipeline,
    transcribe_audio,
    transcribe_with_stable_ts,
)

__all__ = [
    "ASRChunk",
    "ASRConfig",
    "FunASRModel",
    "detect_speech_segments_silero",
    "load_funasr_model",
    "load_whisper_pipeline",
    "refine_timestamps_with_stable_ts",
    "transcribe_audio",
    "transcribe_audio_funasr",
    "transcribe_with_stable_ts",
]
