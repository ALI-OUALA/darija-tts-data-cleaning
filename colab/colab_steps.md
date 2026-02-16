# Colab steps for Darija TTS release packaging

Use these steps to run packaging and validation from Google Colab.

## 1. Prepare runtime

1. Open a new Colab notebook.
2. Enable a standard Python runtime.
3. Upload your project folder or mount Google Drive.

## 2. Install dependencies

```bash
!python -m pip install -U pip
!python -m pip install -r requirements.txt
```

## 3. Verify source export

Confirm that `tts_export/` includes:

- `tts_export/wavs/*.wav`
- `tts_export/metadata.csv`

## 4. Build release artifacts

```bash
!python scripts/pack_release.py \
  --project_root . \
  --tts_export_dir ./tts_export \
  --out_dir ./dist
```

If validation reports non-compliant audio and you want automatic conversion:

```bash
!python scripts/pack_release.py \
  --project_root . \
  --tts_export_dir ./tts_export \
  --out_dir ./dist \
  --fix_audio
```

## 5. Validate output dataset

```bash
!python scripts/validate_dataset.py \
  --dataset_dir ./dist/tts_dataset_ready \
  --json_out ./dist/validate_summary.json
```

## 6. Download artifacts

Download:

- `dist/tts_dataset_ready.zip`
- `dist/tts_export_project.zip`
- `dist/validate_summary.json`

## Optional references

Use these local references if you include them in your repo:

- `colab/darija_tts_whisperx_colab.ipynb`
- `colab/darija_tts_whisperx_colab.py`
