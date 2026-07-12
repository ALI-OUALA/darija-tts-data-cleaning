# Algerian Darija TTS Data Cleaning

A reproducible WhisperX-based pipeline for turning Algerian Darija recordings and transcripts into a clean, training-ready TTS dataset.

## Release snapshot

| Metric | Result |
| --- | ---: |
| Validated segments | 940 |
| Mean segment duration | 6.597 s |
| Audio format | 16 kHz mono |
| Clipping rate | 0.00% |
| Mean alignment coverage | 99.30% |
| Segments with coverage ≥ 0.95 | 96.60% |

## What the pipeline does

- preserves transcript text while refining timestamps with forced alignment
- applies sentence and pause-aware segmentation
- trims and filters clips using duration and quality checks
- validates audio format and metadata consistency
- exports `wavs/*.wav`, `metadata.csv`, and machine-readable validation reports
- generates QC plots for release review

## Pipeline

```text
Audio + transcripts
        ↓
WhisperX rough timing + forced alignment
        ↓
Sentence / pause segmentation
        ↓
Duration and quality filtering
        ↓
Training-ready WAV files + metadata + reports
```

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Package a local TTS export:

```bash
python scripts/pack_release.py \
  --project_root . \
  --tts_export_dir ./tts_export \
  --out_dir ./dist
```

Validate the packaged dataset:

```bash
python scripts/validate_dataset.py \
  --dataset_dir ./dist/tts_dataset_ready \
  --json_out ./dist/validate_summary.json
```

## Quality-control results

<!-- RESULTS_SUMMARY_START -->
| Metric | Value |
| --- | --- |
| Segments | 940 |
| Duration min (s) | 1.554 |
| Duration mean (s) | 6.597 |
| Duration median (s) | 6.448 |
| Duration p90 (s) | 11.004 |
| Duration max (s) | 12.149 |
| % < 2s | 5.00% |
| % > 12s | 0.74% |
| Sample rates | 16000 |
| Channels | 1 |
| Clipping rate | 0.00% |
| Alignment coverage min | 0.9022 |
| Alignment coverage mean | 0.9930 |
| Alignment coverage median | 1.0000 |
| Alignment coverage >= 0.95 | 96.60% |

![Segment duration histogram](docs/figures/duration_hist.png)
![Segment duration CDF](docs/figures/duration_cdf.png)
![Alignment coverage histogram](docs/figures/alignment_coverage_hist.png)
![Dropped segments by reason](docs/figures/dropped_by_reason.png)
<!-- RESULTS_SUMMARY_END -->

## Repository structure

```text
.
├── darija_tts_whisperx_colab.py
├── darija_tts_whisperx_colab.ipynb
├── scripts/
│   ├── pack_release.py
│   ├── make_report.py
│   ├── preflight_check.py
│   └── validate_dataset.py
├── docs/
│   ├── report.md
│   ├── TROUBLESHOOTING.md
│   └── figures/
├── examples/
│   └── sample_usage.md
└── colab/
    └── colab_steps.md
```

## Colab workflow

Use [`colab/colab_steps.md`](colab/colab_steps.md) for the end-to-end Colab workflow. The main reference files are:

- `darija_tts_whisperx_colab.ipynb`
- `darija_tts_whisperx_colab.py`

## Release policy

Heavy and generated artifacts such as `data/`, `tts_export/`, `dist/`, and archive files are intentionally excluded from version control. The repository focuses on the reproducible pipeline, validation tooling, reports, and release documentation.

## Credits

- [WhisperX](https://github.com/m-bain/whisperX) for alignment and timestamp refinement
- the Chatterbox ecosystem for TTS fine-tuning workflow inspiration