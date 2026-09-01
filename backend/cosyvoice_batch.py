"""Generate all translated segments for one video with a single CosyVoice load."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn.functional as functional


def load_wav(path: str, target_sr: int, min_sr: int = 16000) -> torch.Tensor:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    speech = torch.from_numpy(np.asarray(samples).T).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        if sample_rate < min_sr:
            raise ValueError(f"Reference audio must be at least {min_sr} Hz.")
        target_length = round(speech.shape[-1] * target_sr / sample_rate)
        speech = functional.interpolate(
            speech.unsqueeze(0), size=target_length, mode="linear", align_corners=False
        ).squeeze(0)
    return speech


def main(manifest_path: Path, prompt_wav: Path, output_dir: Path) -> None:
    cosyvoice_root = Path(os.getenv("COSYVOICE_HOME", r"D:\tools\CosyVoice"))
    model_dir = cosyvoice_root / "pretrained_models" / "Fun-CosyVoice3-0.5B"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice model is unavailable: {model_dir}")

    ort.preload_dlls()
    sys.path.insert(0, str(cosyvoice_root))
    sys.path.append(str(cosyvoice_root / "third_party" / "Matcha-TTS"))
    os.chdir(cosyvoice_root)

    import cosyvoice.cli.frontend as frontend

    frontend.load_wav = load_wav
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed

    segments = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel(model_dir=str(model_dir))

    for segment in segments:
        set_all_random_seed(20260821 + int(segment["id"]))
        style = segment.get("speech_style") or "Warm, calm documentary narration."
        instruction = (
            "You are a helpful assistant. Speak in natural English with this delivery: "
            f"{style}<|endofprompt|>"
        )
        tts_text = segment.get("tts_text") or segment["translated_text"]
        speech = next(model.inference_instruct2(
            tts_text,
            instruction,
            str(prompt_wav),
            stream=False,
            text_frontend=False,
        ))["tts_speech"]
        output = output_dir / f"segment_{segment['id']:04d}.wav"
        sf.write(output, speech.squeeze().cpu().numpy(), model.sample_rate)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: cosyvoice_batch.py SEGMENTS_JSON PROMPT_WAV OUTPUT_DIR")
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
