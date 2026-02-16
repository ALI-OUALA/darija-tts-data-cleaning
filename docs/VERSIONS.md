# Version Matrix

This project is hardened for **Python 3.11.x on Windows**.

## Runtime baseline
- OS: Windows 10/11
- Python: 3.11.x
- ffmpeg: available in PATH
- Torch checkpoint loading compatibility: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (set by runtime)

## Lockfile source
- File: `requirements.windows-py311.lock.txt`
- Date: 2026-02-11
- Method: generated from a successful resolver/install run in local `.venv311`

## Pinned package set
- whisperx `3.7.6`
- faster-whisper `1.2.1`
- ctranslate2 `4.7.1`
- torch `2.8.0`
- torchaudio `2.8.0`
- pyannote-audio `3.4.0`
- transformers `4.57.6`
- huggingface-hub `0.36.2`
- tokenizers `0.22.2`
- onnxruntime `1.24.1`
- av `16.1.0`
- numpy `2.3.5`
- pandas `3.0.0`
- scipy `1.17.0`
- scikit-learn `1.8.0`
- soundfile `0.13.1`
- librosa `0.11.0`
- tqdm `4.67.3`
- pyarrow `23.0.0`
- datasets `4.5.0`
- ipywidgets `8.1.8`
- psutil `7.2.2`

## Install command
```powershell
.\.venv311\Scripts\python.exe -m pip install --no-input --no-compile -r .\requirements.windows-py311.lock.txt
```

## Regeneration process
1. Create fresh Python 3.11 virtual environment.
2. Install/resolve all pipeline dependencies successfully.
3. Freeze exact versions into lockfile.
4. Re-run strict preflight and smoke test.
