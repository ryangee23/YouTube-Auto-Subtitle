"""Whisper Large V3 Turbo 语音识别"""

from pathlib import Path

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from .base import ASR_MODELS, ASRChunk, device, torch_dtype
from .vad import detect_speech_segments_silero


def load_whisper_pipeline(batch_size: int = 1, model_key: str = "whisper"):
    """加载Whisper语音识别模型并创建推理管道

    加载OpenAI Whisper模型，配置数据类型和设备，
    并构建HuggingFace pipeline用于推理。

    Args:
        batch_size: 批处理大小，增大可加速但需要更多显存
        model_key: 模型键名，支持 'whisper'

    Returns:
        配置好的HuggingFace ASR pipeline对象
    """
    model_path = str(ASR_MODELS["whisper"])

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.config.forced_decoder_ids = None
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_path)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        batch_size=batch_size,
    )


def transcribe_with_stable_ts(
    audio_path: Path,
    language: str | None = None,
) -> list[ASRChunk]:
    """使用stable-ts进行高精度语音识别

    stable-ts是Whisper的增强版本，通过VAD和静音分析提供更精确的时间戳。

    Args:
        audio_path: 音频文件路径
        language: 指定语言代码，None表示自动检测

    Returns:
        带精确时间戳的识别结果列表
    """
    import warnings

    import stable_whisper

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
    warnings.filterwarnings("ignore", message=".*attention mask.*")
    warnings.filterwarnings("ignore", message=".*WhisperSdpaAttention.*")

    print("正在使用 stable-ts 优化时间戳...")
    if device == "cpu":
        print("⚠️  警告: CPU 模式下推理较慢，请耐心等待...")

    model = stable_whisper.load_hf_whisper(
        str(ASR_MODELS["whisper"]),
        device=device,
    )
    model._pipe.model.config.forced_decoder_ids = None

    print("开始转录音频（可能需要几分钟）...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        vad=True,
        suppress_silence=True,
        word_timestamps=True,
        verbose=True,
    )

    print("正在调整时间戳...")
    result.adjust_by_silence(
        audio_path,
        q_levels=20,
        k_size=5,
    )

    refined_chunks: list[ASRChunk] = []
    for segment in result.segments:
        text = segment.text.strip()
        if text:
            refined_chunks.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })

    print(f"stable-ts 优化完成，共 {len(refined_chunks)} 个片段")
    return refined_chunks


def transcribe_audio(
    pipe, audio_path: Path, language: str | None = None, use_vad: bool = True
) -> list[ASRChunk]:
    """使用Whisper模型识别音频

    支持两种模式：
    1. VAD模式（默认）：先用Silero VAD检测语音片段，再逐段识别
    2. 原生模式：使用Whisper内置的分块策略

    Args:
        pipe: Whisper推理管道
        audio_path: 音频文件路径
        language: 指定语言代码，None表示自动检测
        use_vad: 是否使用Silero VAD预处理

    Returns:
        识别结果列表，每个元素包含start/end/text字段
    """
    import librosa

    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    if use_vad:
        import tempfile

        import soundfile as sf

        speech_segments = detect_speech_segments_silero(audio_path)
        if not speech_segments:
            print("未检测到语音片段")
            return []

        waveform, _ = librosa.load(str(audio_path), sr=16000, mono=True)

        chunks: list[ASRChunk] = []
        for seg_start, seg_end in speech_segments:
            start_sample = int(seg_start * 16000)
            end_sample = int(seg_end * 16000)
            segment_audio = waveform[start_sample:end_sample]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            sf.write(tmp_path, segment_audio, 16000)

            try:
                result = pipe(
                    {"raw": segment_audio, "sampling_rate": 16000},
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs,
                    language=language,
                )

                for chunk in result.get("chunks", []):
                    ts = chunk["timestamp"]
                    if ts[0] is None or ts[1] is None:
                        continue
                    text = chunk["text"].strip()
                    if text:
                        chunks.append({
                            "start": seg_start + ts[0],
                            "end": seg_start + ts[1],
                            "text": text,
                        })
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        print(f"识别完成，共 {len(chunks)} 个片段")
        return chunks

    import numpy as np

    audio_array, _ = librosa.load(str(audio_path), sr=16000, mono=True)

    result = pipe(
        audio_array,
        return_timestamps=True,
        chunk_length_s=15,
        stride_length_s=5,
        generate_kwargs=generate_kwargs,
        language=language,
    )

    chunks: list[ASRChunk] = []
    prev_end = 0.0
    for i, chunk in enumerate(result.get("chunks", [])):
        start, end = chunk["timestamp"]
        if start is None:
            start = prev_end
        if end is None:
            next_chunks = result.get("chunks", [])[i + 1:]
            if next_chunks and next_chunks[0]["timestamp"][0] is not None:
                end = next_chunks[0]["timestamp"][0]
            else:
                end = start + 3.0
        text = chunk["text"].strip()
        if text:
            chunks.append({
                "start": start,
                "end": end,
                "text": text,
            })
        prev_end = end

    print(f"识别完成，共 {len(chunks)} 个片段")
    return chunks
