# Troubleshooting

## 1) Python version error
Symptom:
- `ERROR [preflight/E_PYTHON_VERSION]`

Fix:
- Use Python 3.11 explicitly.
- Run setup script:

```powershell
.\scripts\setup_windows_py311.ps1
```

Or pass explicit path:

```powershell
.\scripts\setup_windows_py311.ps1 -PythonExe "C:\Path\To\Python311\python.exe"
```

## 2) ffmpeg not found
Symptom:
- preflight fails `ffmpeg_in_path`

Fix:
- Install ffmpeg and restart terminal:

```powershell
winget install --id Gyan.FFmpeg
```

Verify:

```powershell
ffmpeg -version
```

## 3) Lockfile mismatch
Symptom:
- preflight reports package version mismatches

Fix:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r .\requirements.windows-py311.lock.txt
```

Then re-run:

```powershell
.\.venv311\Scripts\python.exe .\scripts\preflight_check.py --strict-lock
```

## 4) Model download/network errors
Symptom:
- WhisperX load errors or Hugging Face download failures

Fix:
- Check internet access.
- Re-run command.
- If behind proxy/firewall, configure `HTTP_PROXY`/`HTTPS_PROXY`.

## 5) PyTorch weights-only / pyannote checkpoint load error
Symptom:
- `_pickle.UnpicklingError: Weights only load failed`
- mentions `omegaconf.listconfig.ListConfig` in traceback

Fix:
- Use the updated script version in this project (it auto-sets compatibility at runtime).
- Confirm in `tts_export/logs/runtime_info.json` that `torch_checkpoint_loading` is present.
- If you manually run custom scripts, set:

```powershell
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"
```

## 6) Out-of-memory or slow execution
Symptom:
- alignment/transcription fails or runs too slowly

Fixes:
- Use smoke mode first:

```powershell
.\scripts\run_smoke_test.ps1
```

- Lower workload:
  - `--whisper-model-size small`
  - `--batch-size 1`
  - `--max-files N`

## 6) Intel Arc acceleration not active
Symptom:
- runtime reports CPU use

Notes:
- CPU fallback is expected and supported.
- This project guarantees reliability on CPU; acceleration is opportunistic.

## 8) Resume behavior confusion
Symptom:
- Some files are skipped when rerunning

Reason:
- `--resume` intentionally skips already completed artifacts.

Fix:
- Remove output directory for a full clean run:

```powershell
Remove-Item -Recurse -Force .\tts_export
```

Then rerun with `--run-full`.
