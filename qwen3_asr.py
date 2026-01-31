#!/usr/bin/env python3
"""Qwen3-ASR 语音识别工具

使用 Qwen3-ASR-1.7B 进行语音识别，支持52种语言。
"""

import argparse
import json
import os
import sys
from pathlib import Path

from asr import load_qwen_asr_model, transcribe_audio_qwen


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 语音识别 (52语言)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run qwen_asr.py audio/test.wav
  uv run qwen_asr.py audio/test.wav -l zh
  uv run qwen_asr.py audio/test.wav -m qwen-asr-small
  uv run qwen_asr.py audio/test.wav --no-timestamps

支持的语言:
  中文(zh), 英语(en), 粤语(yue), 日语(ja), 韩语(ko), 
  法语(fr), 德语(de), 西班牙语(es), 葡萄牙语(pt), 
  俄语(ru), 阿拉伯语(ar), 等52种语言
        """,
    )
    parser.add_argument("audio_file", help="音频文件路径")
    parser.add_argument(
        "-l", "--language",
        help="指定语言代码 (如 zh, en, ja)，默认自动检测",
    )
    parser.add_argument(
        "-m", "--model",
        choices=["qwen-asr", "qwen-asr-small"],
        default="qwen-asr",
        help="模型选择: qwen-asr (1.7B, 默认) 或 qwen-asr-small (0.6B)",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="不输出时间戳",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出文件路径 (JSON格式)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"错误: 音频文件不存在: {args.audio_file}")
        sys.exit(1)

    audio_path = Path(args.audio_file)

    model_map = {
        "qwen-asr": "Qwen/Qwen3-ASR-1.7B",
        "qwen-asr-small": "Qwen/Qwen3-ASR-0.6B",
    }
    model_name = model_map[args.model]

    use_timestamps = not args.no_timestamps

    print(f"加载模型: {model_name}")
    model = load_qwen_asr_model(
        model_name=model_name,
        use_timestamp=use_timestamps,
    )

    print(f"开始识别: {audio_path}")
    chunks = transcribe_audio_qwen(
        model=model,
        audio_path=audio_path,
        language=args.language,
        use_timestamp=use_timestamps,
    )

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output_path}")
    else:
        print("\n识别结果:")
        print("-" * 50)
        for chunk in chunks:
            if use_timestamps and chunk.get("start", 0) > 0:
                print(f"[{chunk['start']:.2f} - {chunk['end']:.2f}] {chunk['text']}")
            else:
                print(chunk["text"])


if __name__ == "__main__":
    main()
