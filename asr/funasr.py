"""FunASR 语音识别模型（阿里达摩院）

支持模型：
- FunASR Nano: 中英日三语
- FunASR MLT: 31种语言
"""

from pathlib import Path

from .base import ASR_MODELS, ASRChunk, device
from .utils import refine_timestamps_with_stable_ts
from .vad import detect_speech_segments_silero


class FunASRModel:
    """FunASR模型封装类"""

    def __init__(self, model_name: str = "funasr-nano"):
        """初始化FunASR模型

        Args:
            model_name: 模型名称，'funasr-nano'或'funasr-mlt'
        """
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self):
        """加载FunASR模型"""
        import sys

        from funasr import AutoModel
        from funasr.models.ctc import ctc

        model_path = ASR_MODELS[self.model_name]
        sys.modules["ctc"] = ctc
        sys.path.insert(0, str(model_path))
        try:
            import model as _  # noqa: F401
        finally:
            sys.path.pop(0)

        return AutoModel(
            model=str(model_path),
            trust_remote_code=True,
            device=device,
            disable_update=True,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        batch_size_s: int = 60,
        merge_length_s: int = 15,
        use_vad: bool = False,
        refine_timestamps: bool = False,
    ) -> list[ASRChunk]:
        """识别音频

        Args:
            audio_path: 音频文件路径
            language: 语言代码（zh/en/ja/ko/yue）
            batch_size_s: 批处理时长（秒）
            merge_length_s: 合并片段最大时长（秒）
            use_vad: 是否使用VAD预分片
            refine_timestamps: 是否使用stable-ts精调

        Returns:
            识别结果列表
        """
        return transcribe_audio_funasr(
            self.model,
            audio_path,
            language,
            batch_size_s,
            merge_length_s,
            use_vad,
            refine_timestamps,
        )


def load_funasr_model(model_name: str):
    """加载FunASR语音识别模型

    Args:
        model_name: 模型名称，'funasr-nano'或'funasr-mlt'

    Returns:
        加载好的FunASR模型对象
    """
    import sys

    from funasr import AutoModel
    from funasr.models.ctc import ctc

    model_path = ASR_MODELS[model_name]
    sys.modules["ctc"] = ctc
    sys.path.insert(0, str(model_path))
    try:
        import model as _  # noqa: F401
    finally:
        sys.path.pop(0)

    model = AutoModel(
        model=str(model_path),
        trust_remote_code=True,
        device=device,
        disable_update=True,
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
) -> list[ASRChunk]:
    """使用FunASR模型识别音频

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
    import torchaudio.transforms as T

    # 对于 SenseVoice 等模型，通常需要标准 ISO 代码 (zh, en, ko, ja)
    # 对于旧款 Paraformer，可能需要中文名。优先尝试原始输入。
    lang = language

    import re

    def clean_text(text: str) -> str:
        """清理 SenseVoice 的富文本标签 (如 <|HAPPY|>)"""
        if not text:
            return ""
        # 移除 <|...|> 格式的标签
        text = re.sub(r'<\|.*?\|>', '', text)
        return text.strip()

    chunks: list[ASRChunk] = []

    if use_vad:
        print("使用 Silero VAD 预分片...")
        vad_segments = detect_speech_segments_silero(audio_path)

        if not vad_segments:
            print("VAD 未检测到语音片段，使用整段音频")
            vad_segments = [(0, None)]

        waveform, sample_rate = torchaudio.load(str(audio_path))
        # FunASR 内部通常期望 16kHz 采样率
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        for seg_idx, (seg_start, seg_end) in enumerate(vad_segments):
            msg = f"处理片段 {seg_idx + 1}/{len(vad_segments)}: {seg_start:.2f}s"
            if seg_end:
                msg += f" - {seg_end:.2f}s"
            else:
                msg += " - 结尾"
            print(msg)

            start_sample = int(seg_start * sample_rate)
            end_sample = int(seg_end * sample_rate) if seg_end else waveform.shape[1]
            segment_waveform = waveform[:, start_sample:end_sample]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            torchaudio.save(tmp_path, segment_waveform, sample_rate)

            try:
                result = model.generate(
                    input=tmp_path,
                    language=lang,
                    use_itn=True,
                    batch_size_s=batch_size_s,
                    merge_length_s=merge_length_s,
                )

                if result and len(result) > 0:
                    res = result[0]
                    # 优先处理带有时间戳信息的 sentence_info (Paraformer/SenseVoice)
                    if "sentence_info" in res and res["sentence_info"]:
                        for sent in res["sentence_info"]:
                            start_ms = sent.get("start", 0)
                            end_ms = sent.get("end", start_ms + 3000)
                            text = clean_text(sent.get("text", ""))
                            if text:
                                chunks.append({
                                    "start": seg_start + start_ms / 1000.0,
                                    "end": seg_start + end_ms / 1000.0,
                                    "text": text,
                                })
                    # 兼容 SenseVoice 某些版本的 timestamp 输出
                    elif "timestamp" in res and res["timestamp"]:
                        # 假设 timestamp 格式为 [[start, end], [start, end], ...]
                        # 并且 text 对应这些时间戳
                        text_raw = res.get("text", "")
                        # 这里简便处理，如果只有一个文本对应多个时间戳，或者 text 是分好段的
                        text = clean_text(text_raw)
                        if text:
                            # 尝试获取第一段和最后一段的时间
                            ts = res["timestamp"]
                            start_ms = ts[0][0] if ts else 0
                            end_ms = ts[-1][1] if ts else start_ms + 3000
                            chunks.append({
                                "start": seg_start + start_ms / 1000.0,
                                "end": seg_start + end_ms / 1000.0,
                                "text": text,
                            })
                    elif "text" in res:
                        text = clean_text(res["text"])
                        if text:
                            chunks.append({
                                "start": seg_start,
                                "end": seg_end if seg_end else seg_start + 5.0,
                                "text": text,
                            })
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    else:
        # 即使不使用 VAD，也确保加载和可能的重采样以获得最佳效果
        # 虽然 AutoModel 内部会处理，但显式处理更可控
        result = model.generate(
            input=str(audio_path),
            language=lang,
            use_itn=True,
            batch_size_s=batch_size_s,
            merge_length_s=merge_length_s,
        )

        if result and len(result) > 0:
            res = result[0]
            if "sentence_info" in res and res["sentence_info"]:
                for sent in res["sentence_info"]:
                    start_ms = sent.get("start", 0)
                    end_ms = sent.get("end", start_ms + 3000)
                    text = clean_text(sent.get("text", ""))
                    if text:
                        chunks.append({
                            "start": start_ms / 1000.0,
                            "end": end_ms / 1000.0,
                            "text": text,
                        })
            elif "timestamp" in res and res["timestamp"]:
                text = clean_text(res.get("text", ""))
                if text:
                    ts = res["timestamp"]
                    start_ms = ts[0][0]
                    end_ms = ts[-1][1]
                    chunks.append({
                        "start": start_ms / 1000.0,
                        "end": end_ms / 1000.0,
                        "text": text,
                    })
            elif "text" in res:
                text = clean_text(res["text"])
                if text:
                    chunks.append({
                        "start": 0.0,
                        "end": 5.0, # 默认一个较短的时间，后续会被 stable-ts 修正或合并
                        "text": text,
                    })

    print(f"FunASR 识别完成，共 {len(chunks)} 个片段")

    if refine_timestamps and chunks:
        chunks = refine_timestamps_with_stable_ts(audio_path, chunks, language=lang)

    return chunks
