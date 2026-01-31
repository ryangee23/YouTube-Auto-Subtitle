"""ASR基础类型定义"""

from pathlib import Path
from typing import TypedDict

import torch

# 计算设备配置
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float32

# ASR模型路径配置
ASR_MODELS = {
    "whisper": Path("./model/whisper-large-v3-turbo"),
    "funasr-nano": Path("./model/Fun-ASR-Nano-2512"),
    "funasr-mlt": Path("./model/Fun-ASR-MLT-Nano-2512"),
    "qwen-asr": "Qwen/Qwen3-ASR-1.7B",  # HuggingFace 模型名称
    "qwen-asr-small": "Qwen/Qwen3-ASR-0.6B",  # 轻量版
}
DEFAULT_ASR_MODEL = "whisper"


class ASRChunk(TypedDict, total=False):
    """ASR识别结果片段"""

    start: float
    end: float
    text: str
    translated: str


class ASRConfig(TypedDict, total=False):
    """ASR配置参数"""

    language: str | None
    batch_size: int
    use_vad: bool
    refine_timestamps: bool
