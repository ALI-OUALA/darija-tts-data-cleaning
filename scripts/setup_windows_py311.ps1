param(
  [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

function Resolve-Python311 {
  param([string]$Requested)
  if ($Requested -and (Test-Path $Requested)) {
    return (Resolve-Path $Requested).Path
  }

  $candidates = @(
    "C:\Users\ali ouala eddine\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Python311\python.exe"
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { return $c }
  }

  $pyCmd = Get-Command py -ErrorAction SilentlyContinue
  if ($pyCmd) {
    return "py -3.11"
  }

  throw "Python 3.11 executable not found. Pass -PythonExe <path_to_python311.exe>."
}

Write-Host "== Darija WhisperX setup (Windows / Python 3.11) ==" -ForegroundColor Cyan
$resolved = Resolve-Python311 -Requested $PythonExe
Write-Host "Using Python launcher/executable: $resolved"

$venvDir = ".venv311"
if (-not (Test-Path $venvDir)) {
  if ($resolved -eq "py -3.11") {
    py -3.11 -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment with py -3.11 (exit $LASTEXITCODE)." }
  } else {
    & $resolved -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment with $resolved (exit $LASTEXITCODE)." }
  }
}

$pyVenv = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $pyVenv)) {
  throw "Virtual environment python not found at $pyVenv"
}

& $pyVenv -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)." }
& $pyVenv -m pip install --no-input --no-compile -r ".\requirements.windows-py311.lock.txt"
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed (exit $LASTEXITCODE)." }

Write-Host ""
Write-Host "Running preflight..."
& $pyVenv ".\scripts\preflight_check.py" --strict-lock
if ($LASTEXITCODE -ne 0) { throw "Preflight failed (exit $LASTEXITCODE)." }

Write-Host ""
Write-Host "Setup completed." -ForegroundColor Green
Write-Host "Next: .\scripts\run_smoke_test.ps1"
