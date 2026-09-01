"""Download and smoke-test the local ASR and TTS providers on Windows.

Run after installing ``requirements.txt``. The test intentionally uses short,
generated audio: it verifies provider loading and GPU execution without needing
the user's video sample.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
TESTS_DIR = MODELS_DIR / "smoke_tests"
PIPER_VOICE = "en_US-lessac-medium"
WHISPER_MODEL = "small"
PIPER_HOME = Path(os.environ.get("PIPER_HOME", r"D:\tools\piper_local"))
PIPER_EXECUTABLE = PIPER_HOME / "piper" / "piper.exe"
PIPER_DATA_DIR = PIPER_HOME / "piper" / "espeak-ng-data"
PIPER_MODEL = PIPER_HOME / f"{PIPER_VOICE}.onnx"


def write_silence_wav(path: Path) -> None:
    sample_rate = 16_000
    duration_s = 0.5
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(sample_rate * duration_s))


def main() -> None:
    for directory in (WHISPER_DIR, TESTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg") or r"D:\tools\ffmpeg\bin\ffmpeg.exe"
    if not Path(ffmpeg).is_file():
        raise RuntimeError("FFmpeg was not found. Configure Path or update the fallback path.")

    gpu_count = ctranslate2.get_cuda_device_count()
    if gpu_count < 1:
        raise RuntimeError("CTranslate2 cannot see an NVIDIA CUDA device.")

    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
        download_root=str(WHISPER_DIR),
    )
    silent_wav = TESTS_DIR / "silence.wav"
    write_silence_wav(silent_wav)
    list(whisper.transcribe(str(silent_wav), beam_size=1)[0])

    if not PIPER_EXECUTABLE.is_file() or not PIPER_DATA_DIR.is_dir() or not PIPER_MODEL.is_file():
        raise RuntimeError(
            "Verified Piper runtime is missing. Expected D:\\tools\\piper_local with piper.exe, "
            "espeak-ng-data, and en_US-lessac-medium.onnx."
        )

    # Piper's Windows executable requires ASCII-only paths, so both the runtime
    # and this test WAV live under D:\\tools rather than the Chinese project path.
    piper_wav = PIPER_HOME / "piper_test.wav"
    subprocess.run(
        [
            str(PIPER_EXECUTABLE),
            "-m",
            str(PIPER_MODEL),
            "-f",
            str(piper_wav),
            "--espeak_data",
            str(PIPER_DATA_DIR),
        ],
        input="This is a local English text to speech test.\n",
        text=True,
        cwd=PIPER_EXECUTABLE.parent,
        check=True,
    )
    if not piper_wav.is_file() or piper_wav.stat().st_size == 0:
        raise RuntimeError("Piper did not produce a test WAV file.")

    report = {
        "ffmpeg": ffmpeg,
        "cuda_devices": gpu_count,
        "whisper_model": WHISPER_MODEL,
        "whisper_device": "cuda",
        "piper_voice": PIPER_VOICE,
        "piper_runtime": str(PIPER_EXECUTABLE),
        "piper_test_wav": str(piper_wav),
    }
    (TESTS_DIR / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
