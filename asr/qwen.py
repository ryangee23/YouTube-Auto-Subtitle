"""Qwen3-ASR 语音识别

支持 Qwen3-ASR-1.7B 和 Qwen3-ASR-0.6B 模型，
支持52种语言和方言的语言识别和语音识别。
"""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel

from .base import ASRChunk, device

# 模型目录配置
MODEL_DIR = Path("./model")
QWEN_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B": MODEL_DIR / "Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B": MODEL_DIR / "Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ForcedAligner-0.6B": MODEL_DIR / "Qwen3-ForcedAligner-0.6B",
}


def _ensure_model_downloaded(repo_id: str) -> Path:
    """确保模型已下载到本地目录

    Args:
        repo_id: HuggingFace 模型 ID (如 "Qwen/Qwen3-ASR-1.7B")

    Returns:
        本地模型绝对路径
    """
    local_path = QWEN_ASR_MODELS.get(repo_id)
    if local_path is None:
        raise ValueError(f"未知模型: {repo_id}")

    # 转换为绝对路径
    local_path = local_path.resolve()

    # 检查模型是否已下载（检查 config.json 是否存在）
    if not (local_path / "config.json").exists():
        print(f"下载模型 {repo_id} 到 {local_path}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )
        print(f"模型下载完成: {local_path}")

    return local_path


def load_qwen_asr_model(
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    use_timestamp: bool = False,
):
    """加载 Qwen3-ASR 模型

    Args:
        model_name: 模型名称，支持:
            - "Qwen/Qwen3-ASR-1.7B" (默认，52语言，高精度)
            - "Qwen/Qwen3-ASR-0.6B" (52语言，更快速)
        use_timestamp: 是否启用时间戳输出（需要 ForcedAligner）

    Returns:
        Qwen3ASRModel 实例
    """
    device_map = "mps" if device == "mps" else "cpu"
    dtype = torch.float32

    model_path = _ensure_model_downloaded(model_name)

    if use_timestamp:
        aligner_path = _ensure_model_downloaded("Qwen/Qwen3-ForcedAligner-0.6B")
        model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            forced_aligner=str(aligner_path),
            forced_aligner_kwargs={"dtype": dtype, "device_map": device_map},
        )
    else:
        model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
        )

    print(f"已加载 Qwen3-ASR 模型: {model_path} (device={device_map})")
    return model


def transcribe_audio_qwen(
    model: Qwen3ASRModel,
    audio_path: Path,
    language: str | None = None,
    use_timestamp: bool = True,
) -> list[ASRChunk]:
    """使用 Qwen3-ASR 模型识别音频

    Args:
        model: Qwen3ASRModel 实例
        audio_path: 音频文件路径
        language: 指定语言代码 (如 "zh", "en", "ja", "Chinese", "English")，None表示自动检测
        use_timestamp: 是否返回时间戳

    Returns:
        识别结果列表，每个元素包含 start/end/text 字段
    """
    audio_str = str(audio_path)

    results = model.transcribe(
        audio=audio_str,
        language=language,
        return_time_stamps=use_timestamp,
    )

    chunks: list[ASRChunk] = []

    if not results:
        print("未识别到语音内容")
        return chunks

    result = results[0]  # 单个音频的结果

    if use_timestamp and result.time_stamps is not None and len(result.time_stamps) > 0:
        for ts in result.time_stamps:
            text = ts.text.strip() if hasattr(ts, "text") else ""
            if text:
                chunks.append({
                    "start": ts.start_time if hasattr(ts, "start_time") else 0.0,
                    "end": ts.end_time if hasattr(ts, "end_time") else 0.0,
                    "text": text,
                })
    else:
        text = result.text.strip() if hasattr(result, "text") else ""
        if text:
            chunks.append({
                "start": 0.0,
                "end": 0.0,
                "text": text,
            })

    print(f"识别完成，共 {len(chunks)} 个片段")
    return chunks
