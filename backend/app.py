"""External media-job API called by the Nuwax agent HTTP plugin.

This first milestone accepts and persists video uploads, then exposes a stable
job contract. Media processing providers will update the job after the upload
path has been verified with a real Nuwax workflow.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from pipeline import MediaPipeline


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.getenv("MEDIA_STORAGE_DIR", BASE_DIR / "storage"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
pipeline = MediaPipeline(STORAGE_DIR)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="media-job")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def job_dir(job_id: str) -> Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job_id")
    return STORAGE_DIR / job_id


def metadata_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def save_job(job: dict) -> None:
    directory = job_dir(job["job_id"])
    directory.mkdir(parents=True, exist_ok=False)
    metadata_path(job["job_id"]).write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_job(job_id: str) -> dict | None:
    try:
        path = metadata_path(job_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_uploaded_video(job_id: str) -> Path | None:
    input_dir = job_dir(job_id) / "input"
    candidates = list(input_dir.glob("video.*"))
    return candidates[0] if len(candidates) == 1 and candidates[0].is_file() else None


def error(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def validate_upload(file: FileStorage | None) -> tuple[str, str] | None:
    if file is None or not file.filename:
        return None
    safe_name = secure_filename(file.filename)
    extension = Path(safe_name).suffix.lower()
    if not safe_name or extension not in ALLOWED_EXTENSIONS:
        return None
    return safe_name, extension


def register_job(
    job_id: str,
    source_filename: str,
    source_language: str = "zh",
    target_language: str = "en",
    voice: str = "",
    subtitle_mode: str = "english_below_chinese",
    term_constraints: str = "",
) -> dict:
    """Persist common job metadata and start the asynchronous pipeline."""
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "uploaded",
        "progress": 0,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "source_filename": source_filename,
        "source_language": source_language,
        "target_language": target_language,
        "voice": voice,
        "subtitle_mode": subtitle_mode,
        "term_constraints": term_constraints,
        "artifacts": {},
        "segments": [],
    }
    metadata_path(job_id).write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    executor.submit(pipeline.process, job_id)
    return job


def validate_video_url(video_url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(video_url)
    allowed_hosts = {
        host.strip().lower()
        for host in os.getenv(
            "VIDEO_URL_ALLOWED_HOSTS",
            "agent.nuwax.com,agent-statics-tc.nuwax.com,s3p.nuwax.com",
        ).split(",")
        if host.strip()
    }
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or hostname not in allowed_hosts:
        raise ValueError("video_url must be an HTTPS URL from an allowed Nuwax host.")
    return parsed


def download_video_url(
    video_url: str, destination: Path, parsed: urllib.parse.ParseResult | None = None
) -> tuple[str, str]:
    """Download a validated Nuwax file URL into destination."""
    parsed = parsed or validate_video_url(video_url)

    request = urllib.request.Request(
        video_url,
        headers={"User-Agent": "yiying-tongsheng-media-api/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
                )
            content_type = response.headers.get_content_type().lower()
            if content_type.startswith("text/") or content_type == "application/json":
                raise ValueError("video_url did not return a video file.")

            total = 0
            with destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError(
                            f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
                        )
                    output.write(chunk)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"video_url download failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"video_url download failed: {exc.reason}") from exc

    url_name = Path(urllib.parse.unquote(parsed.path)).name
    extension = Path(url_name).suffix.lower()
    content_extension = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
    }.get(content_type)
    if extension not in ALLOWED_EXTENSIONS:
        extension = content_extension or ".mp4"
    source_filename = secure_filename(url_name) or f"video{extension}"
    if Path(source_filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        source_filename = f"video{extension}"
    return source_filename, extension


@app.after_request
def add_cors_headers(response):
    # Restrict this with CORS_ALLOW_ORIGIN after the Nuwax production domain is known.
    response.headers["Access-Control-Allow-Origin"] = os.getenv("CORS_ALLOW_ORIGIN", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return error(f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.", 413)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "yiying-tongsheng-media-api"})


@app.post("/api/jobs")
def create_job():
    """Create an asynchronous media job from a multipart video upload."""
    upload = request.files.get("video")
    validated = validate_upload(upload)
    if validated is None:
        return error("Provide video as MP4, MOV, AVI, or MKV in the 'video' field.", 400)

    filename, extension = validated
    job_id = uuid.uuid4().hex
    job_directory = job_dir(job_id)
    input_directory = job_directory / "input"
    input_directory.mkdir(parents=True)
    video_path = input_directory / f"video{extension}"
    upload.save(video_path)

    job = register_job(
        job_id,
        filename,
        source_language=request.form.get("source_language", "zh"),
        target_language=request.form.get("target_language", "en"),
        voice=request.form.get("voice", ""),
        subtitle_mode=request.form.get("subtitle_mode", "english_below_chinese"),
        term_constraints=request.form.get("term_constraints", ""),
    )

    return jsonify(
        {
            "job_id": job_id,
            "status": job["status"],
            "stage": job["stage"],
            "message": "Video uploaded. Audio extraction and transcription have started.",
        }
    ), 201


@app.post("/api/jobs/from-url")
def create_job_from_url():
    """Download a Nuwax-hosted video URL and create the same asynchronous job."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("video_url"), str):
        return error("Provide video_url in a JSON request body.", 400)

    video_url = payload["video_url"].strip()
    if not video_url:
        return error("video_url must not be empty.", 400)

    try:
        parsed_url = validate_video_url(video_url)
    except ValueError as exc:
        return error(str(exc), 400)

    job_id = uuid.uuid4().hex
    job_directory = job_dir(job_id)
    input_directory = job_directory / "input"
    input_directory.mkdir(parents=True)
    temporary_path = input_directory / "downloaded_video"

    try:
        source_filename, extension = download_video_url(
            video_url, temporary_path, parsed=parsed_url
        )
        video_path = input_directory / f"video{extension}"
        temporary_path.replace(video_path)
        job = register_job(
            job_id,
            source_filename,
            source_language=str(payload.get("source_language", "zh")),
            target_language=str(payload.get("target_language", "en")),
            voice=str(payload.get("voice", "")),
            subtitle_mode=str(
                payload.get("subtitle_mode", "english_below_chinese")
            ),
            term_constraints=str(payload.get("term_constraints", "")),
        )
    except ValueError as exc:
        shutil.rmtree(job_directory, ignore_errors=True)
        return error(str(exc), 400)
    except OSError as exc:
        shutil.rmtree(job_directory, ignore_errors=True)
        return error(f"Could not save downloaded video: {exc}", 500)

    return jsonify(
        {
            "job_id": job_id,
            "status": job["status"],
            "stage": job["stage"],
            "message": "Video URL downloaded. Audio extraction and transcription have started.",
        }
    ), 201


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = load_job(job_id)
    if job is None:
        return error("Job not found.", 404)
    return jsonify(job)


@app.get("/api/job-status")
def get_job_status_by_query():
    """Query a job without requiring a path variable in the Nuwax HTTP node."""
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return error("Provide job_id as a query parameter.", 400)
    job = load_job(job_id)
    if job is None:
        return error("Job not found.", 404)
    return jsonify(job)


@app.get("/api/jobs/<job_id>/source")
def get_source_video(job_id: str):
    if load_job(job_id) is None:
        return error("Job not found.", 404)
    source = get_uploaded_video(job_id)
    if source is None:
        return error("Uploaded video is unavailable.", 404)
    return send_file(source, mimetype="video/mp4", as_attachment=False)


@app.get("/api/jobs/<job_id>/transcript")
def get_transcript(job_id: str):
    job = load_job(job_id)
    if job is None:
        return error("Job not found.", 404)
    transcript = job_dir(job_id) / "artifacts" / "transcript.json"
    if not transcript.is_file():
        return error("Transcript is not ready.", 409)
    return send_file(transcript, mimetype="application/json", as_attachment=False)


@app.get("/api/transcript")
def get_transcript_by_query():
    """Read a transcript using job_id in the query string."""
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return error("Provide job_id as a query parameter.", 400)
    job = load_job(job_id)
    if job is None:
        return error("Job not found.", 404)
    transcript = job_dir(job_id) / "artifacts" / "transcript.json"
    if not transcript.is_file():
        return error("Transcript is not ready.", 409)
    return send_file(transcript, mimetype="application/json", as_attachment=False)


@app.get("/api/transcript-wait")
def get_transcript_wait():
    """Wait briefly for asynchronous transcription, then return the transcript."""
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return error("Provide job_id as a query parameter.", 400)
    try:
        timeout_s = min(max(float(request.args.get("timeout_s", "120")), 1.0), 180.0)
    except ValueError:
        return error("timeout_s must be a number.", 400)

    if load_job(job_id) is None:
        return error("Job not found.", 404)

    transcript = job_dir(job_id) / "artifacts" / "transcript.json"
    deadline = time.monotonic() + timeout_s
    while not transcript.is_file() and time.monotonic() < deadline:
        time.sleep(1)

    if not transcript.is_file():
        job = load_job(job_id) or {}
        return jsonify(
            {
                "error": "Transcript is not ready.",
                "job_id": job_id,
                "status": job.get("status"),
                "stage": job.get("stage"),
                "progress": job.get("progress"),
            }
        ), 409
    return send_file(transcript, mimetype="application/json", as_attachment=False)


def accept_translations(job_id: str, payload: object):
    if not isinstance(payload, dict):
        return error("Provide a JSON object with a 'segments' array.", 400)

    try:
        job = pipeline.submit_translations(
            job_id, payload.get("segments"), payload.get("subtitle_mode")
        )
    except FileNotFoundError:
        return error("Job not found.", 404)
    except ValueError as exc:
        return error(str(exc), 409)

    executor.submit(pipeline.synthesize, job_id)
    return jsonify(
        {
            "job_id": job_id,
            "status": job["status"],
            "stage": job["stage"],
            "message": "English translations accepted. Speech synthesis has started.",
        }
    ), 202


@app.post("/api/jobs/<job_id>/translations")
def submit_translations(job_id: str):
    return accept_translations(job_id, request.get_json(silent=True))


@app.post("/api/translations")
def submit_translations_by_query():
    """Submit reviewed segments with job_id as a query parameter for Nuwax."""
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return error("Provide job_id as a query parameter.", 400)
    return accept_translations(job_id, request.get_json(silent=True))


@app.get("/api/jobs/<job_id>/result")
def get_result_video(job_id: str):
    job = load_job(job_id)
    if job is None:
        return error("Job not found.", 404)
    result = job_dir(job_id) / "artifacts" / "english_dub.mp4"
    if not result.is_file():
        return error("English-dubbed video is not ready.", 409)
    return send_file(result, mimetype="video/mp4", as_attachment=False)


def public_url(path: str) -> str:
    forwarded_proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    scheme = forwarded_proto.split(",", 1)[0].strip() or request.scheme
    return f"{scheme}://{request.host}{path}"


@app.get("/api/result")
def get_result_video_by_query():
    """Return the rendered MP4 without requiring a path variable in Nuwax."""
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return error("Provide job_id as a query parameter.", 400)
    return get_result_video(job_id)


@app.get("/api/timing-report")
def get_timing_report_by_query():
    """Read the TTS timing report with job_id as a query parameter for Nuwax."""
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return error("Provide job_id as a query parameter.", 400)
    if load_job(job_id) is None:
        return error("Job not found.", 404)

    report = job_dir(job_id) / "artifacts" / "timing_report.json"
    if not report.is_file():
        return error("Timing report is not ready.", 409)
    return send_file(report, mimetype="application/json", as_attachment=False)


@app.get("/api/result-wait")
def get_result_wait():
    """Wait for asynchronous synthesis and return URLs usable by the agent."""
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return error("Provide job_id as a query parameter.", 400)
    try:
        # Quick Tunnel may return Cloudflare 524 before a long origin request
        # finishes, so keep this request below the proxy read timeout.
        timeout_s = min(max(float(request.args.get("timeout_s", "90")), 1.0), 90.0)
    except ValueError:
        return error("timeout_s must be a number.", 400)

    job = load_job(job_id)
    if job is None:
        return error("Job not found.", 404)

    result = job_dir(job_id) / "artifacts" / "english_dub.mp4"
    deadline = time.monotonic() + timeout_s
    while not result.is_file() and time.monotonic() < deadline:
        time.sleep(1)

    job = load_job(job_id) or job
    result_url = public_url(f"/api/result?job_id={job_id}")
    subtitles_url = public_url(
        f"/api/jobs/{job_id}/artifacts/english_subtitles.srt"
    )
    timing_report_url = public_url(
        f"/api/jobs/{job_id}/artifacts/timing_report.json"
    )
    if result.is_file():
        return jsonify(
            {
                "ready": True,
                "job_id": job_id,
                "status": job.get("status", "completed"),
                "stage": job.get("stage", "completed"),
                "progress": job.get("progress", 100),
                "result_url": result_url,
                "subtitles_url": subtitles_url,
                "timing_report_url": timing_report_url,
            }
        )

    response = {
        "ready": False,
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage"),
        "progress": job.get("progress"),
        "result_url": result_url,
        "subtitles_url": subtitles_url,
        "timing_report_url": timing_report_url,
    }
    if job.get("error"):
        response["detail"] = job["error"]
    return jsonify(response)


@app.get("/api/jobs/<job_id>/artifacts/<artifact_name>")
def get_artifact(job_id: str, artifact_name: str):
    job = load_job(job_id)
    if job is None:
        return error("Job not found.", 404)
    allowed = {
        "english_subtitles.srt": "text/plain; charset=utf-8",
        "english_subtitles.ass": "text/plain; charset=utf-8",
        "timing_report.json": "application/json",
        "background_no_vocals.wav": "audio/wav",
    }
    if artifact_name not in allowed:
        return error("Unknown artifact.", 404)
    artifact = job_dir(job_id) / "artifacts" / artifact_name
    if not artifact.is_file():
        return error("Artifact is not ready.", 409)
    return send_file(artifact, mimetype=allowed[artifact_name], as_attachment=False)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
