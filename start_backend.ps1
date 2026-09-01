<# Start the backend with project-local paths without changing persistent system settings. #>
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
$BackendRoot = Join-Path $RepoRoot 'backend'
$BackendPython = Join-Path $BackendRoot '.venv\Scripts\python.exe'
$RuntimeRoot = Join-Path $BackendRoot 'runtime'
$WhisperDir = Join-Path $BackendRoot 'models\whisper\faster-whisper-small'
$DemucsDir = Join-Path $BackendRoot 'models\demucs\HTDemucs'
$CosyPython = Join-Path $RuntimeRoot 'cosyvoice_env\Scripts\python.exe'
$CosyModel = Join-Path $RuntimeRoot 'CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B'
$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ffmpeg) { $ffmpeg = Join-Path $RuntimeRoot 'ffmpeg\bin\ffmpeg.exe' }

if (-not (Test-Path $BackendPython)) { throw 'backend .venv is missing. Run setup.ps1 first.' }
if (-not (Test-Path $WhisperDir)) { throw 'Pinned Whisper model is missing. Run setup.ps1 first.' }
if (-not (Test-Path $DemucsDir)) { throw 'Pinned Demucs model is missing. Run setup.ps1 first.' }
if (-not (Test-Path $CosyPython)) { throw 'CosyVoice Python environment is missing. Run setup.ps1 first.' }
if (-not (Test-Path $CosyModel)) { throw 'Pinned CosyVoice model is missing. Run setup.ps1 first.' }
if (-not (Test-Path $ffmpeg)) { throw 'FFmpeg is missing. Install it or place it under runtime\ffmpeg\bin.' }

$env:TORCH_HOME = Join-Path $BackendRoot 'models\torch'
$env:WHISPER_MODEL_PATH = $WhisperDir
$env:DEMUCS_REPO = $DemucsDir
$env:DEMUCS_MODEL = 'htdemucs'
$env:FFMPEG_PATH = $ffmpeg
$env:COSYVOICE_HOME = Join-Path $RuntimeRoot 'CosyVoice'
$env:COSYVOICE_PYTHON = $CosyPython
$env:COSYVOICE_STAGING_DIR = Join-Path $RuntimeRoot 'cosyvoice_jobs'
$env:DEMUCS_STAGING_DIR = Join-Path $RuntimeRoot 'demucs_jobs'
$env:TERMINOLOGY_PATH = Join-Path $RepoRoot '术语库\v1 .json'
$env:MEDIA_STORAGE_DIR = Join-Path $BackendRoot 'storage'

Push-Location $BackendRoot
try { & $BackendPython '.\app.py' } finally { Pop-Location }
