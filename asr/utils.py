"""ASR共享工具函数"""

from pathlib import Path

from .base import ASRChunk, device


def refine_timestamps_with_stable_ts(
    audio_path: Path,
    chunks: list[ASRChunk],
    language: str | None = None,
) -> list[ASRChunk]:
    """使用stable-ts精调时间戳

    利用stable-ts的对齐功能，根据音频波形优化时间戳精度。

    Args:
        audio_path: 音频文件路径
        chunks: 原始识别结果
        language: 语言代码 (如 'ko', 'zh', 'en')

    Returns:
        时间戳精调后的识别结果
    """
    import warnings

    import stable_whisper

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("使用 stable-ts 精调时间戳...")

    try:
        model = stable_whisper.load_model("base", device=device)
        full_text = " ".join(c["text"] for c in chunks)
        # 传递语言参数给 align 函数，避免 "expected argument for language" 错误
        result = model.align(str(audio_path), full_text, language=language)

        refined_chunks: list[ASRChunk] = []
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
