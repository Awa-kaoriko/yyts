# External media backend

This service is the external execution layer for the Nuwax `译影同声` agent.
It performs local ASR, reviewed English TTS, BGM/ambience preservation,
subtitle rendering, and structured timing checks as the execution layer for
the Nuwax agent.

## Local setup

Run these commands from `backend` in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

The health check is `http://127.0.0.1:5000/api/health`.

## Reproducible setup from GitHub

From a fresh clone, run the repository-level script in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\start_backend.ps1
```

`setup.ps1` installs the two pinned Python environments, checks out the pinned
CosyVoice source revision, and downloads the pinned Whisper, Demucs, and
CosyVoice model revisions into project-local directories. The revisions and
the environment lock files are recorded in `models.lock.json`,
`requirements-backend-lock.txt`, and `requirements-cosyvoice-lock.txt`.

The setup script requires Python 3.12 and Python 3.10 launchers, Git, an
NVIDIA driver/CUDA runtime, and FFmpeg. It does not change persistent system
environment variables. `start_backend.ps1` sets process-local paths only and
starts Flask on port 5000. The existing fallback paths remain available when
the script is not used.

## Local model setup

After FFmpeg is available, install the ASR and TTS dependencies and run the
smoke test:

```powershell
python -m pip install -r requirements.txt
python verify_local_models.py
```

The smoke test downloads `small` for faster-whisper into `backend/models/`.
The verified English TTS runtime is CosyVoice 3 under `D:\tools\CosyVoice`,
with its Python environment at `D:\tools\conda_envs\yiying-cosyvoice`.

The requirements install the CUDA-enabled PyTorch runtime and Demucs. Demucs
downloads its first separation model when a job reaches the mix stage; the
resulting `no_vocals` stem keeps original accompaniment and ambience without
mixing the Chinese speech back into the English result.

## Upload contract

`POST /api/jobs` uses `multipart/form-data`:

- `video` (required): MP4, MOV, AVI, or MKV
- `target_language` (optional, defaults to `en`)
- `voice` (optional)
- `term_constraints` (optional)
- `subtitle_mode` (optional): `english_below_chinese` (default),
  `english_burned`, or `sidecar_only`

The response contains an asynchronous `job_id`. Poll `GET /api/jobs/{job_id}`
instead of waiting for media processing in the Nuwax workflow. The first
worker stage extracts 16 kHz mono WAV audio and transcribes it with local
Whisper on CUDA. When `status` becomes `awaiting_translation`, retrieve the
timestamped Chinese transcript from `GET /api/jobs/{job_id}/transcript`.

When the Nuwax workflow has corrected and translated every segment, submit it
to `POST /api/jobs/{job_id}/translations` as JSON:

```json
{
  "subtitle_mode": "english_below_chinese",
  "segments": [
    {
      "id": 1,
      "corrected_source_text": "经术语校正的中文原文",
      "translated_text": "English sentence for segment one.",
      "subtitle_text": "English sentence for segment one.",
      "target_duration_s": 3.1,
      "speech_style": "Warm, calm documentary narration."
    }
  ]
}
```

The backend validates that every transcript segment is present, synthesizes
the English audio with local CosyVoice, preserves each original start time, and
creates `GET /api/jobs/{job_id}/result`. A segment more than 0.45 seconds too
long **or too short** relative to its target duration pauses the job at
`awaiting_timing_review` and is flagged as `timing_review_needed` in the job
data. Nuwax rewrites or resynthesizes those segments and posts the complete
segment array again; only an all-clear job renders the final video.

For an accepted job, Demucs separates the source vocal stem from original
accompaniment/ambience. The final MP4 mixes the `no_vocals` stem at its
original level with English speech; it does not reuse the full Chinese source
track and does not automatically duck BGM. English subtitles are burned into
the MP4 by default and are also exported as sidecar files.

## Output artifacts

- `GET /api/jobs/{job_id}/result`: English-dubbed MP4 with burned English subtitles.
- `GET /api/jobs/{job_id}/artifacts/english_subtitles.srt`: English SRT.
- `GET /api/jobs/{job_id}/artifacts/english_subtitles.ass`: styled subtitle source.
- `GET /api/jobs/{job_id}/artifacts/timing_report.json`: per-segment duration,
  offset, silence-gap, style, and review-status evidence.
- `GET /api/jobs/{job_id}/artifacts/background_no_vocals.wav`: retained
  accompaniment/ambience stem used for the final mix.

## Current boundary

The service accepts up to 1 GB by default and stores uploads under
`backend/storage/`, which is intentionally git-ignored. Before external
deployment, configure an authentication mechanism, a fixed production CORS
origin, object storage, and a public HTTPS endpoint.
