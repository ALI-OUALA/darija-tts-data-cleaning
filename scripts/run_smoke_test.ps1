$ErrorActionPreference = "Stop"

$pyVenv = ".\.venv311\Scripts\python.exe"
if (-not (Test-Path $pyVenv)) {
  throw "Missing .venv311. Run scripts/setup_windows_py311.ps1 first."
}

if (-not (Test-Path ".\cafe-small-clean.zip")) {
  throw "Dataset zip not found: .\cafe-small-clean.zip"
}

Write-Host "== Running smoke test (max 3 files) ==" -ForegroundColor Cyan
& $pyVenv ".\darija_tts_whisperx_colab.py" `
  --smoke-test `
  --max-files 3 `
  --zip-path ".\cafe-small-clean.zip" `
  --extract-root ".\data" `
  --output-dir ".\tts_export" `
  --lockfile ".\requirements.windows-py311.lock.txt" `
  --resume
if ($LASTEXITCODE -ne 0) {
  throw "Smoke test failed with exit code $LASTEXITCODE"
}

Write-Host "Smoke test finished." -ForegroundColor Green
