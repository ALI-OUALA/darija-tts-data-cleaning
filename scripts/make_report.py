#!/usr/bin/env python3
"""Generate dataset stats, charts, and markdown reports for release packaging."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("make_report")

RESULTS_START = "<!-- RESULTS_SUMMARY_START -->"
RESULTS_END = "<!-- RESULTS_SUMMARY_END -->"


def configure_logging(verbose: bool = False) -> None:
    """Configure logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def parse_bool(value: Any) -> bool | None:
    """Parse bool-like values from CSV strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], ratio: float) -> float:
    """Compute a percentile using linear interpolation."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    clamped = min(max(ratio, 0.0), 1.0)
    position = (len(values) - 1) * clamped
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def read_metadata(dataset_dir: Path) -> list[str]:
    """Read metadata references, returning normalized wav paths."""
    metadata_path = dataset_dir / "metadata.csv"
    references: list[str] = []
    if not metadata_path.exists():
        return references

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if not row or not row[0].strip():
                continue
            first = row[0].replace("\\", "/").strip()
            name = Path(first).name
            if name.lower().endswith(".wav"):
                references.append(f"wavs/{name}")
    return references


def scan_wavs(dataset_dir: Path, references: list[str]) -> dict[str, Any]:
    """Scan wav files directly when QC CSV is missing."""
    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    filename_to_duration: dict[str, float] = {}

    names = [Path(ref).name for ref in references] if references else []
    wav_paths = (
        [dataset_dir / "wavs" / wav_name for wav_name in names]
        if names
        else sorted((dataset_dir / "wavs").glob("*.wav"))
    )

    for wav_path in wav_paths:
        if not wav_path.exists():
            continue
        with wave.open(str(wav_path), "rb") as handle:
            rate = int(handle.getframerate())
            frames = int(handle.getnframes())
            channel_count = int(handle.getnchannels())
        duration = frames / rate if rate else 0.0
        key = f"wavs/{wav_path.name}"
        filename_to_duration[key] = duration
        durations.append(duration)
        sample_rates.update([rate])
        channels.update([channel_count])

    return {
        "durations": durations,
        "sample_rates": sample_rates,
        "channels": channels,
        "clipping_rate": None,
        "filename_to_duration": filename_to_duration,
    }


def read_qc_post(reports_dir: Path) -> dict[str, Any]:
    """Read duration and audio stats from qc_post_export.csv if present."""
    qc_path = reports_dir / "qc_post_export.csv"
    if not qc_path.exists():
        return {
            "durations": [],
            "sample_rates": Counter(),
            "channels": Counter(),
            "clipping_rate": None,
            "filename_to_duration": {},
        }

    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    clipping_flags: list[bool] = []
    filename_to_duration: dict[str, float] = {}

    with qc_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = str(row.get("filename", "")).replace("\\", "/").strip()
            if filename and filename.lower().endswith(".wav"):
                filename = f"wavs/{Path(filename).name}"

            duration = safe_float(row.get("duration_sec"))
            if duration is not None:
                durations.append(duration)
                if filename:
                    filename_to_duration[filename] = duration

            sample_rate = safe_float(row.get("sample_rate"))
            if sample_rate is not None:
                sample_rates.update([int(sample_rate)])

            channel_count = safe_float(row.get("channels"))
            if channel_count is not None:
                channels.update([int(channel_count)])

            clip_flag = parse_bool(row.get("is_clipping"))
            if clip_flag is None:
                peak = safe_float(row.get("peak_amplitude"))
                if peak is not None:
                    clip_flag = peak >= 0.999
            if clip_flag is not None:
                clipping_flags.append(clip_flag)

    clipping_rate = (
        (sum(clipping_flags) / len(clipping_flags))
        if clipping_flags
        else None
    )
    return {
        "durations": durations,
        "sample_rates": sample_rates,
        "channels": channels,
        "clipping_rate": clipping_rate,
        "filename_to_duration": filename_to_duration,
    }


def read_alignment_stats(reports_dir: Path) -> dict[str, Any]:
    """Read alignment coverage statistics from alignment_report.csv."""
    path = reports_dir / "alignment_report.csv"
    coverages: list[float] = []
    status_counts: Counter[str] = Counter()
    if not path.exists():
        return {"coverages": coverages, "status_counts": status_counts}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = safe_float(row.get("coverage"))
            if value is not None:
                coverages.append(value)
            status = str(row.get("status", "")).strip()
            if status:
                status_counts.update([status])

    return {"coverages": coverages, "status_counts": status_counts}


def read_drop_reasons(reports_dir: Path) -> Counter[str]:
    """Read dropped segment reasons from dropped_segments.csv."""
    path = reports_dir / "dropped_segments.csv"
    reasons: Counter[str] = Counter()
    if not path.exists():
        return reasons

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reason = str(row.get("reason", "")).strip() or "unknown"
            reasons.update([reason])
    return reasons


def save_duration_histogram(durations: list[float], out_path: Path) -> None:
    """Save a duration histogram."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = max(12, min(60, int(math.sqrt(len(durations)) * 2)))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(durations, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_title("Segment durations histogram")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_duration_cdf(durations: list[float], out_path: Path) -> None:
    """Save a duration CDF plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_values = sorted(durations)
    n_items = len(sorted_values)
    cdf_y = [(idx + 1) / n_items for idx in range(n_items)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(sorted_values, cdf_y, color="#ff7f0e", linewidth=2)
    ax.set_title("Segment durations CDF")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_alignment_histogram(coverages: list[float], out_path: Path) -> None:
    """Save alignment coverage histogram."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(coverages, bins=20, range=(0, 1), color="#2ca02c", edgecolor="white")
    ax.set_title("Alignment coverage histogram")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_drop_reason_bar(reasons: Counter[str], out_path: Path) -> None:
    """Save dropped-segments reason bar chart."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(reasons.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(labels)), values, color="#d62728")
    ax.set_title("Dropped segments by reason")
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def format_pct(value: float | None) -> str:
    """Format percent values."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def format_float(value: float | None, digits: int = 3) -> str:
    """Format float values."""
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def compute_summary(
    durations: list[float],
    sample_rates: Counter[int],
    channels: Counter[int],
    clipping_rate: float | None,
    coverages: list[float],
) -> dict[str, Any]:
    """Compute aggregate summary stats."""
    if not durations:
        raise ValueError("No segment durations available from reports or wav scan.")

    sorted_durations = sorted(durations)
    duration_summary = {
        "segments": len(sorted_durations),
        "min": min(sorted_durations),
        "mean": sum(sorted_durations) / len(sorted_durations),
        "median": statistics.median(sorted_durations),
        "p90": percentile(sorted_durations, 0.9),
        "max": max(sorted_durations),
        "pct_lt_2": sum(val < 2.0 for val in sorted_durations) / len(sorted_durations),
        "pct_gt_12": sum(val > 12.0 for val in sorted_durations) / len(sorted_durations),
    }

    alignment_summary: dict[str, float] | None = None
    if coverages:
        ordered = sorted(coverages)
        alignment_summary = {
            "min": min(ordered),
            "mean": sum(ordered) / len(ordered),
            "median": statistics.median(ordered),
            "pct_ge_095": sum(val >= 0.95 for val in ordered) / len(ordered),
        }

    return {
        "duration": duration_summary,
        "sample_rates": dict(sorted(sample_rates.items())),
        "channels": dict(sorted(channels.items())),
        "clipping_rate": clipping_rate,
        "alignment": alignment_summary,
    }


def relative_link(base: Path, target: Path) -> str:
    """Build a relative markdown link path."""
    return str(target.resolve().relative_to(base.resolve())).replace("\\", "/")


def build_results_block(summary: dict[str, Any], figure_links: dict[str, str]) -> str:
    """Build README Results Summary block content."""
    duration = summary["duration"]
    rows = [
        ["Segments", str(duration["segments"])],
        ["Duration min (s)", format_float(duration["min"], digits=3)],
        ["Duration mean (s)", format_float(duration["mean"], digits=3)],
        ["Duration median (s)", format_float(duration["median"], digits=3)],
        ["Duration p90 (s)", format_float(duration["p90"], digits=3)],
        ["Duration max (s)", format_float(duration["max"], digits=3)],
        ["% < 2s", format_pct(duration["pct_lt_2"])],
        ["% > 12s", format_pct(duration["pct_gt_12"])],
        ["Sample rates", ", ".join(str(k) for k in summary["sample_rates"].keys()) or "N/A"],
        ["Channels", ", ".join(str(k) for k in summary["channels"].keys()) or "N/A"],
        ["Clipping rate", format_pct(summary["clipping_rate"])],
    ]
    alignment = summary.get("alignment")
    if alignment:
        rows.extend(
            [
                ["Alignment coverage min", format_float(alignment["min"], digits=4)],
                ["Alignment coverage mean", format_float(alignment["mean"], digits=4)],
                ["Alignment coverage median", format_float(alignment["median"], digits=4)],
                ["Alignment coverage >= 0.95", format_pct(alignment["pct_ge_095"])],
            ]
        )

    figure_lines = [
        f"![Segment duration histogram]({figure_links['duration_hist']})",
        f"![Segment duration CDF]({figure_links['duration_cdf']})",
    ]
    if "alignment_hist" in figure_links:
        figure_lines.append(
            f"![Alignment coverage histogram]({figure_links['alignment_hist']})"
        )
    if "dropped_bar" in figure_links:
        figure_lines.append(
            f"![Dropped segments by reason]({figure_links['dropped_bar']})"
        )

    return "\n".join(
        [
            RESULTS_START,
            "### Results summary",
            "",
            md_table(["Metric", "Value"], rows),
            "",
            "### Figures",
            "",
            *figure_lines,
            RESULTS_END,
        ]
    )


def update_readme_results(readme_path: Path, block: str) -> None:
    """Replace or append results markers in README."""
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        content = "# Darija TTS dataset release\n"

    start = content.find(RESULTS_START)
    end = content.find(RESULTS_END)
    if start != -1 and end != -1 and start < end:
        end += len(RESULTS_END)
        updated = f"{content[:start]}{block}{content[end:]}"
    else:
        updated = f"{content.rstrip()}\n\n## Results Summary\n\n{block}\n"
    readme_path.write_text(updated, encoding="utf-8")


def build_report_markdown(
    generated_at: str,
    summary: dict[str, Any],
    alignment_status_counts: Counter[str],
    drop_reasons: Counter[str],
    figure_links: dict[str, str],
) -> str:
    """Build docs/report.md body."""
    duration = summary["duration"]
    summary_rows = [
        ["Segments", str(duration["segments"])],
        ["Duration min (s)", format_float(duration["min"], 3)],
        ["Duration mean (s)", format_float(duration["mean"], 3)],
        ["Duration median (s)", format_float(duration["median"], 3)],
        ["Duration p90 (s)", format_float(duration["p90"], 3)],
        ["Duration max (s)", format_float(duration["max"], 3)],
        ["% < 2s", format_pct(duration["pct_lt_2"])],
        ["% > 12s", format_pct(duration["pct_gt_12"])],
        ["Sample rates", ", ".join(str(k) for k in summary["sample_rates"].keys()) or "N/A"],
        ["Channels", ", ".join(str(k) for k in summary["channels"].keys()) or "N/A"],
        ["Clipping rate", format_pct(summary["clipping_rate"])],
    ]
    alignment = summary.get("alignment")
    if alignment:
        summary_rows.extend(
            [
                ["Alignment coverage min", format_float(alignment["min"], 4)],
                ["Alignment coverage mean", format_float(alignment["mean"], 4)],
                ["Alignment coverage median", format_float(alignment["median"], 4)],
                ["Alignment coverage >= 0.95", format_pct(alignment["pct_ge_095"])],
            ]
        )

    sections = [
        "# Darija TTS export report",
        "",
        f"_Generated on {generated_at}_",
        "",
        "## Overview",
        "",
        (
            "This report summarizes the exported Darija TTS dataset used for "
            "LoRA fine-tuning preparation. It reports duration distribution, "
            "audio format consistency, alignment coverage, and dropped segment "
            "reasons when source QC files are available."
        ),
        "",
        "## Methodology",
        "",
        (
            "The report aggregates metrics from `qc_post_export.csv`, "
            "`alignment_report.csv`, and `dropped_segments.csv` when present. "
            "If post-export QC data is unavailable, the script falls back to "
            "direct WAV header scanning for duration and format statistics."
        ),
        "",
        (
            "All plots are generated with matplotlib without modifying source "
            "audio or transcript text."
        ),
        "",
        "## Dataset summary",
        "",
        md_table(["Metric", "Value"], summary_rows),
        "",
    ]

    if alignment_status_counts:
        rows = [[status, str(count)] for status, count in alignment_status_counts.items()]
        sections.extend(
            [
                "## Alignment status breakdown",
                "",
                md_table(["Status", "Count"], rows),
                "",
            ]
        )

    if drop_reasons:
        rows = [[reason, str(count)] for reason, count in drop_reasons.items()]
        sections.extend(
            [
                "## Dropped segments by reason",
                "",
                md_table(["Reason", "Count"], rows),
                "",
            ]
        )

    sections.extend(
        [
            "## Source files used",
            "",
            md_table(
                ["Source", "Purpose"],
                [
                    ["`reports/qc_post_export.csv`", "Duration, sample rate, channels, clipping"],
                    ["`reports/alignment_report.csv`", "Coverage statistics and alignment status"],
                    ["`reports/dropped_segments.csv`", "Drop reason counts"],
                    ["`dataset/wavs/*.wav`", "Fallback duration scan when QC post export is missing"],
                ],
            ),
            "",
        ]
    )

    figure_lines = [
        f"![Segment duration histogram]({figure_links['duration_hist']})",
        "",
        f"![Segment duration CDF]({figure_links['duration_cdf']})",
        "",
    ]
    if "alignment_hist" in figure_links:
        figure_lines.extend(
            [
                f"![Alignment coverage histogram]({figure_links['alignment_hist']})",
                "",
            ]
        )
    if "dropped_bar" in figure_links:
        figure_lines.extend(
            [
                f"![Dropped segments by reason]({figure_links['dropped_bar']})",
                "",
            ]
        )

    sections.extend(["## Figures", "", *figure_lines])
    return "\n".join(sections).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate markdown report and matplotlib figures from dataset/QC files."
    )
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--reports_dir", default="")
    parser.add_argument("--docs_dir", default="")
    parser.add_argument("--readme_path", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    project_root = Path(args.project_root).resolve()
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else project_root
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else project_root / "reports"
    docs_dir = Path(args.docs_dir).resolve() if args.docs_dir else project_root / "docs"
    readme_path = Path(args.readme_path).resolve() if args.readme_path else project_root / "README.md"
    figures_dir = docs_dir / "figures"
    report_path = docs_dir / "report.md"

    docs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    references = read_metadata(dataset_dir)
    qc = read_qc_post(reports_dir)
    if qc["durations"]:
        LOGGER.info("Loaded %d durations from qc_post_export.csv", len(qc["durations"]))
    else:
        LOGGER.info("qc_post_export.csv missing/unusable; scanning wav headers.")
        qc = scan_wavs(dataset_dir, references)

    alignment = read_alignment_stats(reports_dir)
    drop_reasons = read_drop_reasons(reports_dir)

    summary = compute_summary(
        durations=qc["durations"],
        sample_rates=qc["sample_rates"],
        channels=qc["channels"],
        clipping_rate=qc["clipping_rate"],
        coverages=alignment["coverages"],
    )

    duration_hist_path = figures_dir / "duration_hist.png"
    duration_cdf_path = figures_dir / "duration_cdf.png"
    save_duration_histogram(qc["durations"], duration_hist_path)
    save_duration_cdf(qc["durations"], duration_cdf_path)

    figure_links: dict[str, str] = {
        "duration_hist": relative_link(readme_path.parent, duration_hist_path),
        "duration_cdf": relative_link(readme_path.parent, duration_cdf_path),
    }

    if alignment["coverages"]:
        alignment_hist_path = figures_dir / "alignment_coverage_hist.png"
        save_alignment_histogram(alignment["coverages"], alignment_hist_path)
        figure_links["alignment_hist"] = relative_link(readme_path.parent, alignment_hist_path)
    else:
        LOGGER.info("alignment_report.csv missing or has no coverage values; skipping chart.")

    if drop_reasons:
        drop_bar_path = figures_dir / "dropped_by_reason.png"
        save_drop_reason_bar(drop_reasons, drop_bar_path)
        figure_links["dropped_bar"] = relative_link(readme_path.parent, drop_bar_path)
    else:
        LOGGER.info("dropped_segments.csv missing or empty; skipping chart.")

    results_block = build_results_block(summary, figure_links)
    update_readme_results(readme_path, results_block)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_markdown = build_report_markdown(
        generated_at=generated_at,
        summary=summary,
        alignment_status_counts=alignment["status_counts"],
        drop_reasons=drop_reasons,
        figure_links={
            key: relative_link(report_path.parent, (figures_dir / Path(value).name))
            for key, value in figure_links.items()
        },
    )
    report_path.write_text(report_markdown, encoding="utf-8")

    LOGGER.info("README updated: %s", readme_path)
    LOGGER.info("Report written: %s", report_path)
    LOGGER.info("Figures written under: %s", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
