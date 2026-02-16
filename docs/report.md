# Darija TTS export report

_Generated on 2026-02-15 19:42:57 UTC_

## Overview

This report summarizes the exported Darija TTS dataset used for LoRA fine-tuning preparation. It reports duration distribution, audio format consistency, alignment coverage, and dropped segment reasons when source QC files are available.

## Methodology

The report aggregates metrics from `qc_post_export.csv`, `alignment_report.csv`, and `dropped_segments.csv` when present. If post-export QC data is unavailable, the script falls back to direct WAV header scanning for duration and format statistics.

All plots are generated with matplotlib without modifying source audio or transcript text.

## Dataset summary

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

## Alignment status breakdown

| Status | Count |
| --- | --- |
| ok | 147 |

## Dropped segments by reason

| Reason | Count |
| --- | --- |
| too_short_unmergeable | 7 |
| below_min_duration_after_trim | 1 |

## Source files used

| Source | Purpose |
| --- | --- |
| `reports/qc_post_export.csv` | Duration, sample rate, channels, clipping |
| `reports/alignment_report.csv` | Coverage statistics and alignment status |
| `reports/dropped_segments.csv` | Drop reason counts |
| `dataset/wavs/*.wav` | Fallback duration scan when QC post export is missing |

## Figures

![Segment duration histogram](figures/duration_hist.png)

![Segment duration CDF](figures/duration_cdf.png)

![Alignment coverage histogram](figures/alignment_coverage_hist.png)

![Dropped segments by reason](figures/dropped_by_reason.png)
