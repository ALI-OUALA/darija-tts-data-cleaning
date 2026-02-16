# Sample usage for trainer integration

This example is trainer agnostic. It shows how to load the packaged dataset and
map rows into your own training framework.

## Dataset assumptions

- Root folder: `dist/tts_dataset_ready`
- Audio files: `dist/tts_dataset_ready/wavs/*.wav`
- Metadata file: `dist/tts_dataset_ready/metadata.csv`
- Delimiter: pipe (`|`)
- Metadata format:
  - `wavs/<file>.wav|raw_text`
  - or `wavs/<file>.wav|raw_text|normalized_text`

## Python loader snippet

```python
from pathlib import Path
import csv

dataset_root = Path("dist/tts_dataset_ready")
metadata_path = dataset_root / "metadata.csv"

records = []
with metadata_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle, delimiter="|")
    for row in reader:
        if not row:
            continue
        rel_wav = row[0]
        raw_text = row[1] if len(row) > 1 else ""
        normalized_text = row[2] if len(row) > 2 else raw_text
        wav_path = dataset_root / rel_wav
        records.append(
            {
                "audio_path": str(wav_path),
                "text": raw_text,
                "normalized_text": normalized_text,
            }
        )

print(f"Loaded {len(records)} training records.")
```

## Integration notes

1. Preserve text columns exactly; do not rewrite transcript content.
2. Keep file paths relative to dataset root for portability.
3. Run `scripts/validate_dataset.py` in CI before starting training jobs.
