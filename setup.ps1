<#
Install the pinned backend/CosyVoice environments and download the pinned models.
Run from a fresh clone with PowerShell 5+ or PowerShell 7.
#>
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
$BackendRoot = Join-Path $RepoRoot 'backend'
$RuntimeRoot = Join-Path $BackendRoot 'runtime'
$BackendVenv = Join-Path $BackendRoot '.venv'
$CosyHome = Join-Path $RuntimeRoot 'CosyVoice'
$CosyEnv = Join-Path $RuntimeRoot 'cosyvoice_env'
$WhisperDir = Join-Path $BackendRoot 'models\whisper\faster-whisper-small'
$DemucsDir = Join-Path $BackendRoot 'models\demucs\HTDemucs'
$TorchHome = Join-Path $BackendRoot 'models\torch'
$LockData = Get-Content (Join-Path $BackendRoot 'models.lock.json') -Raw | ConvertFrom-Json

function Find-Python([string]$Version) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidate = (& py "-$Version" -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $candidate) { return $candidate.Trim() }
    }
    throw "Python $Version is required. Install it and rerun this script."
}

function Ensure-Venv([string]$Python, [string]$Path) {
    if (-not (Test-Path (Join-Path $Path 'Scripts\python.exe'))) {
        & $Python -m venv $Path
    }
}

function Install-Locked([string]$Python, [string]$LockFile) {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed for $Python" }
    & $Python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r $LockFile
    if ($LASTEXITCODE -ne 0) { throw "Pinned dependency installation failed for $Python" }
}

$py312 = Find-Python '3.12'
$py310 = Find-Python '3.10'
Ensure-Venv $py312 $BackendVenv
Ensure-Venv $py310 $CosyEnv
$backendPython = Join-Path $BackendVenv 'Scripts\python.exe'
$cosyPython = Join-Path $CosyEnv 'Scripts\python.exe'

foreach ($directory in @(
    (Join-Path $BackendRoot 'models\whisper'),
    (Join-Path $BackendRoot 'models\demucs'),
    $TorchHome,
    (Join-Path $BackendRoot 'storage'),
    (Join-Path $RuntimeRoot 'cosyvoice_jobs'),
    (Join-Path $RuntimeRoot 'demucs_jobs')
)) { New-Item -ItemType Directory -Force $directory | Out-Null }

Install-Locked $backendPython (Join-Path $BackendRoot 'requirements-backend-lock.txt')
Install-Locked $cosyPython (Join-Path $BackendRoot 'requirements-cosyvoice-lock.txt')

if (-not (Test-Path (Join-Path $CosyHome '.git'))) {
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $CosyHome
    if ($LASTEXITCODE -ne 0) { throw 'CosyVoice clone failed.' }
} else {
    git -C $CosyHome fetch --all --tags
    if ($LASTEXITCODE -ne 0) { throw 'CosyVoice fetch failed.' }
}
git -C $CosyHome checkout --detach $LockData.cosyvoice.source_commit
if ($LASTEXITCODE -ne 0) { throw 'CosyVoice checkout failed.' }
git -C $CosyHome submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw 'CosyVoice submodule initialization failed.' }

& $backendPython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$($LockData.backend.whisper.repo_id)', revision='$($LockData.backend.whisper.revision)', local_dir=r'$WhisperDir')"
if ($LASTEXITCODE -ne 0) { throw 'Whisper model download failed.' }
& $backendPython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$($LockData.backend.demucs.repo_id)', revision='$($LockData.backend.demucs.revision)', local_dir=r'$DemucsDir')"
if ($LASTEXITCODE -ne 0) { throw 'Demucs model download failed.' }
& $cosyPython -m pip install huggingface_hub modelscope==1.20.0
if ($LASTEXITCODE -ne 0) { throw 'Model download client installation failed.' }
& $cosyPython -c "from modelscope import snapshot_download; snapshot_download('$($LockData.cosyvoice.model_repo)', revision='$($LockData.cosyvoice.model_revision)', local_dir=r'$(Join-Path $CosyHome 'pretrained_models\Fun-CosyVoice3-0.5B')')"
if ($LASTEXITCODE -ne 0) { throw 'CosyVoice model download failed.' }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        & $winget.Source install --id Gyan.FFmpeg.Shared -e --accept-source-agreements --accept-package-agreements
    }
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'FFmpeg is not on PATH. Install a Windows FFmpeg build and set FFMPEG_PATH before starting.'
}

Write-Host 'Pinned environments and models are ready.' -ForegroundColor Green
Write-Host 'Run .\start_backend.ps1 to start the local API.'
