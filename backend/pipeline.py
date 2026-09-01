"""Background media stages for the first 译影同声 MVP milestone."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import wave
from datetime import UTC, datetime
from pathlib import Path

from faster_whisper import WhisperModel
from terminology import Terminology


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class MediaPipeline:
    """Extract audio and transcribe one stored job at a time."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self._model: WhisperModel | None = None
        self._model_lock = threading.Lock()
        self._ffmpeg = (
            os.getenv("FFMPEG_PATH")
            or shutil.which("ffmpeg")
            or r"D:\tools\ffmpeg\bin\ffmpeg.exe"
        )
        self._ffprobe = Path(self._ffmpeg).with_name("ffprobe.exe")
        self._timing_tolerance_s = float(os.getenv("TIMING_TOLERANCE_S", "0.45"))
        glossary = Path(os.getenv("TERMINOLOGY_PATH", str(Path(__file__).parent.parent / "术语库" / "v1 .json")))
        self._terminology = Terminology(glossary)

    def submit_translations(
        self, job_id: str, submitted_segments: object, subtitle_mode: object = None
    ) -> dict:
        """Validate Nuwax's reviewed English text before synthesis begins."""
        metadata = self.storage_dir / job_id / "job.json"
        if not metadata.is_file():
            raise FileNotFoundError("Job not found.")

        job = self._load_job(metadata)
        if job.get("status") not in {"awaiting_translation", "awaiting_timing_review"}:
            raise ValueError("This job is not ready to accept translations.")
        if not isinstance(submitted_segments, list):
            raise ValueError("'segments' must be a JSON array.")

        reviewed_by_id: dict[int, dict[str, object]] = {}
        for item in submitted_segments:
            if not isinstance(item, dict) or not isinstance(item.get("id"), int):
                raise ValueError("Each translation must contain an integer 'id'.")
            text = item.get("translated_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each translation needs non-empty 'translated_text'.")
            if item["id"] in reviewed_by_id:
                raise ValueError("Translation segment IDs must be unique.")
            corrected_source = item.get("corrected_source_text")
            if corrected_source is not None and (
                not isinstance(corrected_source, str) or not corrected_source.strip()
            ):
                raise ValueError("'corrected_source_text' must be a non-empty string when provided.")
            subtitle_text = item.get("subtitle_text", text)
            if not isinstance(subtitle_text, str) or not subtitle_text.strip():
                raise ValueError("'subtitle_text' must be a non-empty string when provided.")
            tts_text = item.get("tts_text")
            if tts_text is not None and (not isinstance(tts_text, str) or not tts_text.strip()):
                raise ValueError("'tts_text' must be a non-empty string when provided.")
            speech_style = item.get("speech_style", "Warm, calm documentary narration.")
            if not isinstance(speech_style, str) or not speech_style.strip():
                raise ValueError("'speech_style' must be a non-empty string when provided.")
            target_duration = item.get("target_duration_s")
            if target_duration is not None and (
                isinstance(target_duration, bool) or not isinstance(target_duration, (int, float))
            ):
                raise ValueError("'target_duration_s' must be a number when provided.")
            reviewed_by_id[item["id"]] = {
                "translated_text": text.strip(),
                "corrected_source_text": corrected_source.strip()
                if isinstance(corrected_source, str)
                else None,
                "subtitle_text": subtitle_text.strip(),
                "tts_text": tts_text.strip() if isinstance(tts_text, str) else None,
                "speech_style": speech_style.strip(),
                "target_duration_s": float(target_duration) if target_duration is not None else None,
            }

        expected_ids = {segment["id"] for segment in job.get("segments", [])}
        if set(reviewed_by_id) != expected_ids:
            raise ValueError("Translations must cover every transcript segment exactly once.")

        chosen_subtitle_mode = subtitle_mode or job.get("subtitle_mode", "english_below_chinese")
        if chosen_subtitle_mode not in {"english_below_chinese", "english_burned", "sidecar_only"}:
            raise ValueError("Unsupported subtitle_mode.")

        for segment in job["segments"]:
            reviewed = reviewed_by_id[segment["id"]]
            source_window = segment["end"] - segment["start"]
            target_duration = reviewed["target_duration_s"] or source_window
            if target_duration <= 0 or target_duration > source_window + 0.05:
                raise ValueError("'target_duration_s' must be positive and within the source time window.")
            corrected_source, term_matches = self._terminology.correct(
                reviewed["corrected_source_text"] or segment["corrected_source_text"] or segment["source_text"]
            )
            tts_text = reviewed["tts_text"] or self._terminology.tts_text(
                reviewed["translated_text"], term_matches
            )
            errors = self._terminology.validate(
                reviewed["translated_text"], reviewed["subtitle_text"], tts_text, term_matches
            )
            if errors:
                raise ValueError("Terminology validation failed: " + " ".join(errors))
            segment["translated_text"] = reviewed["translated_text"]
            segment["corrected_source_text"] = corrected_source
            segment["subtitle_text"] = reviewed["subtitle_text"]
            segment["tts_text"] = tts_text
            segment["term_matches"] = term_matches
            segment["speech_style"] = reviewed["speech_style"]
            segment["target_duration_s"] = round(target_duration, 3)
            segment["playback_rate"] = 1.0

        job["subtitle_mode"] = chosen_subtitle_mode

        self._update(job, status="queued", stage="translation_received", progress=65)
        self._save_job(metadata, job)
        return job

    def synthesize(self, job_id: str) -> None:
        """Turn Nuwax-reviewed English segments into an aligned MP4."""
        job_directory = self.storage_dir / job_id
        metadata = job_directory / "job.json"

        try:
            job = self._load_job(metadata)
            if not job.get("segments") or any(
                not segment.get("translated_text") for segment in job["segments"]
            ):
                raise RuntimeError("The job has no complete English translations.")

            self._update(job, status="running", stage="synthesizing_speech", progress=70)
            self._save_job(metadata, job)
            staging = self._cosyvoice_staging_dir(job_id)
            source_audio = job_directory / "artifacts" / "source_16k_mono.wav"
            prompt_wav = staging / "reference.wav"
            self._prepare_voice_prompt(source_audio, job["segments"], prompt_wav)
            audio_files = self._synthesize_segments(job["segments"], staging, prompt_wav)

            report_path = job_directory / "artifacts" / "timing_report.json"
            self._write_timing_report(job["segments"], report_path)
            job["artifacts"] = {
                **job.get("artifacts", {}),
                "timing_report": "artifacts/timing_report.json",
            }
            timing_violations = [
                segment["id"]
                for segment in job["segments"]
                if segment["timing_review_needed"]
            ]
            if timing_violations:
                job["timing_violations"] = timing_violations
                self._update(
                    job,
                    status="awaiting_timing_review",
                    stage="timing_review",
                    progress=82,
                )
                self._save_job(metadata, job)
                return

            self._update(job, stage="separating_background", progress=86)
            self._save_job(metadata, job)
            source_video = self._source_video(job_directory)
            background_audio = job_directory / "artifacts" / "background_no_vocals.wav"
            self._separate_background(source_video, job_id, background_audio)

            self._update(job, stage="rendering_subtitles", progress=90)
            self._save_job(metadata, job)
            subtitle_srt = job_directory / "artifacts" / "english_subtitles.srt"
            subtitle_ass = job_directory / "artifacts" / "english_subtitles.ass"
            self._write_subtitles(
                job["segments"],
                subtitle_srt,
                subtitle_ass,
                self._probe_video_size(source_video),
            )
            staged_ass = staging / "english_subtitles.ass"
            shutil.copy2(subtitle_ass, staged_ass)

            self._update(job, stage="rendering_video", progress=94)
            self._save_job(metadata, job)
            output = job_directory / "artifacts" / "english_dub.mp4"
            source_audio = job_directory / "artifacts" / "source_16k_mono.wav"
            self._render_video(
                source_video,
                source_audio,
                background_audio,
                job["segments"],
                audio_files,
                staged_ass,
                job["subtitle_mode"],
                output,
            )

            job["artifacts"] = {
                **job.get("artifacts", {}),
                "english_video": "artifacts/english_dub.mp4",
                "background_audio": "artifacts/background_no_vocals.wav",
                "english_subtitles_srt": "artifacts/english_subtitles.srt",
                "english_subtitles_ass": "artifacts/english_subtitles.ass",
            }
            job.pop("timing_violations", None)
            self._update(job, status="completed", stage="completed", progress=100)
            self._save_job(metadata, job)
        except Exception as exc:
            if metadata.is_file():
                job = self._load_job(metadata)
                self._update(job, status="failed", stage="failed", progress=100)
                job["error"] = str(exc)
                self._save_job(metadata, job)

    def process(self, job_id: str) -> None:
        job_directory = self.storage_dir / job_id
        metadata = job_directory / "job.json"

        try:
            job = self._load_job(metadata)
            self._update(job, status="running", stage="extracting_audio", progress=10)
            self._save_job(metadata, job)

            source = self._source_video(job_directory)
            audio = job_directory / "artifacts" / "source_16k_mono.wav"
            audio.parent.mkdir(exist_ok=True)
            self._extract_audio(source, audio)

            self._update(job, stage="transcribing", progress=30)
            self._save_job(metadata, job)
            transcript = self._transcribe(audio, job["source_language"])

            transcript_path = job_directory / "artifacts" / "transcript.json"
            transcript_path.write_text(
                json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            job["segments"] = transcript["segments"]
            job["artifacts"] = {
                **job.get("artifacts", {}),
                "source_audio": "artifacts/source_16k_mono.wav",
                "transcript": "artifacts/transcript.json",
            }
            self._update(
                job,
                status="awaiting_translation",
                stage="transcribed",
                progress=60,
            )
            self._save_job(metadata, job)
        except Exception as exc:  # Keep a pollable error record for the agent.
            if metadata.is_file():
                job = self._load_job(metadata)
                self._update(job, status="failed", stage="failed", progress=100)
                job["error"] = str(exc)
                self._save_job(metadata, job)

    def _extract_audio(self, source: Path, output: Path) -> None:
        ffmpeg = Path(self._ffmpeg)
        if not ffmpeg.is_file():
            raise RuntimeError("FFmpeg was not found. Add it to Path or set FFMPEG_PATH.")

        subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce an audio file.")

    def _transcribe(self, audio: Path, source_language: str) -> dict:
        model = self._get_model()
        segments, info = model.transcribe(
            str(audio),
            language=source_language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )

        transcript_segments = []
        for index, segment in enumerate(segments, start=1):
            source_text = segment.text.strip()
            corrected_source_text, term_matches = self._terminology.correct(source_text)
            transcript_segments.append(
                {
                    "id": index,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "source_text": source_text,
                    "translated_text": "",
                    "corrected_source_text": corrected_source_text,
                    "subtitle_text": "",
                    "speech_style": "",
                    "tts_text": "",
                    "term_matches": term_matches,
                    "words": [
                        {
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "text": word.word,
                        }
                        for word in (segment.words or [])
                    ],
                }
            )

        return {
            "source_language": info.language,
            "language_probability": round(info.language_probability, 4),
            "segments": transcript_segments,
        }

    def _get_model(self) -> WhisperModel:
        with self._model_lock:
            if self._model is None:
                model_dir = Path(__file__).resolve().parent / "models" / "whisper"
                configured_model = os.getenv("WHISPER_MODEL_PATH")
                model_ref = configured_model if configured_model and Path(configured_model).is_dir() else "small"
                self._model = WhisperModel(
                    model_ref,
                    device="cuda",
                    compute_type="float16",
                    download_root=str(model_dir),
                )
            return self._model

    def _cosyvoice_staging_dir(self, job_id: str) -> Path:
        python = Path(os.getenv("COSYVOICE_PYTHON", r"D:\tools\conda_envs\yiying-cosyvoice\python.exe"))
        model = Path(os.getenv("COSYVOICE_HOME", r"D:\tools\CosyVoice")) / "pretrained_models" / "Fun-CosyVoice3-0.5B"
        if not python.is_file() or not model.is_dir():
            raise RuntimeError("Verified CosyVoice runtime or model is unavailable under D:\\tools.")
        staging = Path(os.getenv("COSYVOICE_STAGING_DIR", r"D:\tools\cosyvoice_jobs")) / job_id
        staging.mkdir(parents=True, exist_ok=True)
        return staging

    def _prepare_voice_prompt(self, source_audio: Path, segments: list[dict], output: Path) -> None:
        if not source_audio.is_file():
            raise RuntimeError("Source audio is unavailable for CosyVoice reference speech.")
        reference = max(segments, key=lambda segment: segment["end"] - segment["start"])
        duration = min(8.0, reference["end"] - reference["start"])
        subprocess.run(
            [
                str(self._require_ffmpeg()),
                "-y",
                "-ss",
                f"{reference['start']:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(source_audio),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _synthesize_segments(self, segments: list[dict], staging: Path, prompt_wav: Path) -> list[Path]:
        manifest = staging / "segments.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "id": segment["id"],
                        "tts_text": segment.get("tts_text") or segment["translated_text"],
                        "speech_style": segment.get("speech_style"),
                    }
                    for segment in segments
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        runtime = Path(os.getenv("COSYVOICE_PYTHON", r"D:\tools\conda_envs\yiying-cosyvoice\python.exe"))
        script = Path(__file__).with_name("cosyvoice_batch.py")
        environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
        subprocess.run(
            [str(runtime), str(script), str(manifest), str(prompt_wav), str(staging)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        files: list[Path] = []

        for index, segment in enumerate(segments):
            output = staging / f"segment_{segment['id']:04d}.wav"
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"CosyVoice did not produce audio for segment {segment['id']}.")

            source_window = segment["end"] - segment["start"]
            target_duration = segment.get("target_duration_s", source_window)
            pause_after = self._planned_pause(segment, index, len(segments))
            speech_target = target_duration - pause_after
            duration = self._wav_duration(output)
            playback_rate = duration / speech_target
            if abs(duration - speech_target) > 0.1 and 0.85 <= playback_rate <= 1.15:
                adjusted = output.with_name(f"{output.stem}_adjusted.wav")
                subprocess.run(
                    [
                        str(self._require_ffmpeg()),
                        "-y",
                        "-i",
                        str(output),
                        "-filter:a",
                        f"atempo={playback_rate:.6f}",
                        str(adjusted),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                adjusted.replace(output)
                duration = self._wav_duration(output)
            else:
                playback_rate = 1.0
            timeline_duration = duration + pause_after
            duration_error = timeline_duration - target_duration
            segment["tts_duration"] = round(duration, 3)
            segment["source_window"] = round(source_window, 3)
            segment["target_duration_s"] = round(target_duration, 3)
            segment["duration_error_s"] = round(duration_error, 3)
            segment["start_offset_s"] = 0.0
            segment["end_offset_s"] = round(segment["start"] + timeline_duration - segment["end"], 3)
            segment["playback_rate"] = round(playback_rate, 3)
            segment["pause_after_s"] = pause_after
            if duration_error > self._timing_tolerance_s:
                segment["timing_reason"] = "overlong"
            elif duration_error < -self._timing_tolerance_s:
                segment["timing_reason"] = "too_short"
            else:
                segment["timing_reason"] = "accepted"
            segment["timing_review_needed"] = segment["timing_reason"] != "accepted"
            files.append(output)

        for index, segment in enumerate(segments[:-1]):
            actual_end = segment["start"] + segment["tts_duration"]
            next_start = segments[index + 1]["start"]
            segment["additional_silence_gap_s"] = round(max(0.0, next_start - actual_end), 3)
        if segments:
            segments[-1]["additional_silence_gap_s"] = 0.0

        return files

    @staticmethod
    def _planned_pause(segment: dict, index: int, total_segments: int) -> float:
        if index == total_segments - 1:
            return 0.0
        text = str(segment.get("translated_text", "")).rstrip()
        if text.endswith((".", "?", "!")):
            return 0.24
        if text.endswith((",", ";", ":")):
            return 0.12
        return 0.0

    def _render_video(
        self,
        source_video: Path,
        source_audio: Path,
        background_audio: Path,
        segments: list[dict],
        audio_files: list[Path],
        subtitle_ass: Path,
        subtitle_mode: str,
        output: Path,
    ) -> None:
        if not source_audio.is_file() or not background_audio.is_file():
            raise RuntimeError("Source timing audio or separated background audio is unavailable.")
        output.parent.mkdir(exist_ok=True)
        timeline_duration = self._wav_duration(source_audio)
        filter_parts = [f"[1:a]aresample=44100,atrim=duration={timeline_duration:.3f}[background]"]
        mix_inputs = ["[background]"]
        for index, (segment, audio_file) in enumerate(zip(segments, audio_files), start=2):
            delay_ms = round(segment["start"] * 1000)
            label = f"voice{index - 1}"
            filter_parts.append(
                f"[{index}:a]aresample=44100,adelay={delay_ms}:all=1[{label}]"
            )
            mix_inputs.append(f"[{label}]")
        filter_parts.append(
            "".join(mix_inputs)
            + f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0:normalize=0,"
            + f"atrim=duration={timeline_duration:.3f}[dub]"
        )

        # Input 0 is the video, 1 is Demucs' no-vocals stem, and 2..N are TTS WAV files.
        command = [str(self._require_ffmpeg()), "-y", "-i", str(source_video)]
        command.extend(["-i", str(background_audio)])
        for audio_file in audio_files:
            command.extend(["-i", str(audio_file)])
        video_filter = None
        if subtitle_mode != "sidecar_only":
            subtitle_filename = subtitle_ass.resolve().as_posix().replace(":", r"\:")
            video_filter = f"subtitles=filename='{subtitle_filename}':charenc=UTF-8"
        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "0:v:0",
                "-map",
                "[dub]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
            ]
        )
        if video_filter:
            command.extend(["-vf", video_filter])
        command.append(str(output))
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce the English-dubbed video.")

    def _separate_background(self, source_video: Path, job_id: str, output: Path) -> None:
        """Use Demucs to retain accompaniment and ambience without the Chinese vocal stem."""
        staging_root = Path(os.getenv("DEMUCS_STAGING_DIR", r"D:\tools\demucs_jobs"))
        staging = staging_root / job_id
        input_dir = staging / "input"
        output_dir = staging / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        staged_video = input_dir / "source.mp4"
        shutil.copy2(source_video, staged_video)
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--name",
            os.getenv("DEMUCS_MODEL", "htdemucs"),
        ]
        demucs_repo = os.getenv("DEMUCS_REPO")
        if demucs_repo and Path(demucs_repo).is_dir():
            command.extend(["--repo", demucs_repo])
        command.extend([
            "--two-stems",
            "vocals",
            "--device",
            "cuda",
            "--out",
            str(output_dir),
            str(staged_video),
        ])
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Demucs background separation failed: {result.stderr[-1000:]}")
        candidates = list(output_dir.rglob("no_vocals.wav"))
        if len(candidates) != 1:
            raise RuntimeError("Demucs did not produce exactly one no_vocals.wav stem.")
        output.parent.mkdir(exist_ok=True)
        shutil.copy2(candidates[0], output)

    def _probe_video_size(self, source_video: Path) -> tuple[int, int]:
        if not self._ffprobe.is_file():
            return (1920, 1080)
        result = subprocess.run(
            [
                str(self._ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(source_video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(result.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])

    def _write_subtitles(
        self, segments: list[dict], srt_path: Path, ass_path: Path, video_size: tuple[int, int]
    ) -> None:
        width, height = video_size
        srt_lines: list[str] = []
        ass_lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
            f"Style: English,Arial,{max(26, round(height / 26))},&H00FFFFFF,&H00000000,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,36,36,{max(30, round(height / 30))},1",
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        ]
        for index, segment in enumerate(segments, start=1):
            text = str(segment.get("subtitle_text") or segment["translated_text"])
            srt_lines.extend(
                [
                    str(index),
                    f"{self._srt_time(segment['start'])} --> {self._srt_time(segment['end'])}",
                    text,
                    "",
                ]
            )
            ass_text = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")
            ass_lines.append(
                f"Dialogue: 0,{self._ass_time(segment['start'])},{self._ass_time(segment['end'])},English,,0,0,0,,{ass_text}"
            )
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        ass_path.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")

    def _write_timing_report(self, segments: list[dict], report_path: Path) -> None:
        report_path.write_text(
            json.dumps(
                {
                    "timing_tolerance_s": self._timing_tolerance_s,
                    "segments": [
                        {
                            "id": segment["id"],
                            "corrected_source_text": segment.get("corrected_source_text", segment["source_text"]),
                            "translated_text": segment["translated_text"],
                            "tts_text": segment.get("tts_text", segment["translated_text"]),
                            "term_matches": segment.get("term_matches", []),
                            "speech_style": segment.get("speech_style", ""),
                            "target_duration_s": segment.get("target_duration_s"),
                            "tts_duration_s": segment.get("tts_duration"),
                            "duration_error_s": segment.get("duration_error_s"),
                            "start_offset_s": segment.get("start_offset_s"),
                            "end_offset_s": segment.get("end_offset_s"),
                            "additional_silence_gap_s": segment.get("additional_silence_gap_s"),
                            "pause_after_s": segment.get("pause_after_s", 0.0),
                            "playback_rate": segment.get("playback_rate", 1.0),
                            "status": segment.get("timing_reason", "pending"),
                        }
                        for segment in segments
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _srt_time(seconds: float) -> str:
        milliseconds = round(seconds * 1000)
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        secs, milliseconds = divmod(milliseconds, 1_000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def _ass_time(seconds: float) -> str:
        centiseconds = round(seconds * 100)
        hours, centiseconds = divmod(centiseconds, 360_000)
        minutes, centiseconds = divmod(centiseconds, 6_000)
        secs, centiseconds = divmod(centiseconds, 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def _require_ffmpeg(self) -> Path:
        ffmpeg = Path(self._ffmpeg)
        if not ffmpeg.is_file():
            raise RuntimeError("FFmpeg was not found. Add it to Path or set FFMPEG_PATH.")
        return ffmpeg

    @staticmethod
    def _wav_duration(path: Path) -> float:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()

    @staticmethod
    def _source_video(job_directory: Path) -> Path:
        candidates = list((job_directory / "input").glob("video.*"))
        if len(candidates) != 1 or not candidates[0].is_file():
            raise RuntimeError("The uploaded source video is unavailable.")
        return candidates[0]

    @staticmethod
    def _load_job(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_job(path: Path, job: dict) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _update(job: dict, **changes: object) -> None:
        job.update(changes)
        job["updated_at"] = utc_now()
