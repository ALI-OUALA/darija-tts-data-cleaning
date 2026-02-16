#!/usr/bin/env python3
"""Validate a TTS dataset folder with wavs/ and metadata.csv."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("validate_dataset")

EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_SAMPLE_WIDTH = 2  # bytes (PCM16)


@dataclass(frozen=True)
class MetadataRow:
    """Normalized metadata row."""

    row_number: int
    filename: str
    raw_text: str
    normalized_text: str | None
    column_count: int


def configure_logging(verbose: bool = False) -> None:
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
    )


def normalize_metadata_filename(value: str) -> str:
    """Normalize metadata filename to wavs/<name>.wav."""
    cleaned = value.replace("\\", "/").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    name = Path(cleaned).name
    if not name.lower().endswith(".wav"):
        return ""
    return f"wavs/{name}"


def is_header_row(row: list[str]) -> bool:
    """Heuristic header detection."""
    if not row:
        return False
    first = row[0].strip().lower()
    if first.endswith(".wav"):
        return False
    return first in {"filename", "file", "path", "audio_path", "wav"}


def load_metadata_rows(metadata_path: Path) -> tuple[list[MetadataRow], list[str], set[int]]:
    """Parse metadata rows and return parsed rows + parse errors + column counts."""
    rows: list[MetadataRow] = []
    errors: list[str] = []
    column_counts: set[int] = set()

    if not metadata_path.exists():
        return rows, [f"metadata.csv missing: {metadata_path}"], column_counts

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for line_number, raw_row in enumerate(reader, start=1):
            if not raw_row or all(not cell.strip() for cell in raw_row):
                continue
            if line_number == 1 and is_header_row(raw_row):
                LOGGER.info("Detected metadata header row at line 1; skipping it.")
                continue
            if len(raw_row) not in {2, 3}:
                errors.append(
                    f"Line {line_number}: expected 2 or 3 pipe-separated columns, found {len(raw_row)}."
                )
                continue
            normalized_name = normalize_metadata_filename(raw_row[0])
            if not normalized_name:
                errors.append(
                    f"Line {line_number}: first column is not a valid wav reference: {raw_row[0]!r}"
                )
                continue
            column_counts.add(len(raw_row))
            rows.append(
                MetadataRow(
                    row_number=line_number,
                    filename=normalized_name,
                    raw_text=raw_row[1],
                    normalized_text=raw_row[2] if len(raw_row) == 3 else None,
                    column_count=len(raw_row),
                )
            )

    return rows, errors, column_counts


def inspect_wav(wav_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read wav header info or return an error string."""
    try:
        with wave.open(str(wav_path), "rb") as handle:
            return {
                "channels": int(handle.getnchannels()),
                "sample_rate": int(handle.getframerate()),
                "sample_width": int(handle.getsampwidth()),
                "compression": str(handle.getcomptype()),
                "n_frames": int(handle.getnframes()),
            }, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {exc}"


def build_text_summary(summary: dict[str, Any]) -> str:
    """Build a short plain-text summary."""
    lines = [
        f"Dataset: {summary['dataset_dir']}",
        f"Metadata rows: {summary['metadata_row_count']}",
        f"WAV files on disk: {summary['wav_count']}",
        f"Referenced WAV files: {summary['referenced_wav_count']}",
        (
            "Validation result: PASS"
            if summary["is_valid"]
            else "Validation result: FAIL"
        ),
    ]
    if summary["errors"]:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in summary["errors"])
    if summary["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in summary["warnings"])
    return "\n".join(lines)


def validate_dataset(
    dataset_dir: Path,
    strict_unreferenced: bool = False,
) -> dict[str, Any]:
    """Validate dataset structure, metadata integrity, and wav format."""
    dataset_dir = dataset_dir.resolve()
    wav_dir = dataset_dir / "wavs"
    metadata_path = dataset_dir / "metadata.csv"

    errors: list[str] = []
    warnings: list[str] = []

    if not dataset_dir.exists():
        errors.append(f"Dataset directory not found: {dataset_dir}")
    if not wav_dir.exists():
        errors.append(f"Missing wav directory: {wav_dir}")
    if not metadata_path.exists():
        errors.append(f"Missing metadata file: {metadata_path}")

    rows, parse_errors, column_counts = load_metadata_rows(metadata_path)
    errors.extend(parse_errors)

    wav_files = sorted(wav_dir.glob("*.wav")) if wav_dir.exists() else []
    wav_names = {wav_path.name.lower(): wav_path for wav_path in wav_files}

    duplicate_filenames: list[str] = []
    seen: set[str] = set()
    missing_references: list[str] = []
    format_issues: list[dict[str, Any]] = []

    for row in rows:
        normalized_lower = row.filename.lower()
        if normalized_lower in seen:
            duplicate_filenames.append(row.filename)
            continue
        seen.add(normalized_lower)

        wav_name = Path(row.filename).name.lower()
        wav_path = wav_names.get(wav_name)
        if wav_path is None:
            missing_references.append(row.filename)
            continue

        header, wav_error = inspect_wav(wav_path)
        if wav_error:
            format_issues.append({"filename": row.filename, "error": wav_error})
            continue

        if (
            header["channels"] != EXPECTED_CHANNELS
            or header["sample_rate"] != EXPECTED_SAMPLE_RATE
            or header["sample_width"] != EXPECTED_SAMPLE_WIDTH
            or header["compression"] != "NONE"
        ):
            format_issues.append(
                {
                    "filename": row.filename,
                    "channels": header["channels"],
                    "sample_rate": header["sample_rate"],
                    "sample_width": header["sample_width"],
                    "compression": header["compression"],
                }
            )

    if duplicate_filenames:
        errors.append(
            f"Found duplicate metadata filenames ({len(duplicate_filenames)})."
        )
    if missing_references:
        errors.append(
            f"Found metadata references to missing wav files ({len(missing_references)})."
        )
    if format_issues:
        errors.append(
            f"Found wav format violations ({len(format_issues)}); expected mono/16kHz/PCM16."
        )

    referenced_lower = {Path(row.filename).name.lower() for row in rows}
    unreferenced = sorted(
        f"wavs/{wav_path.name}"
        for wav_path in wav_files
        if wav_path.name.lower() not in referenced_lower
    )
    if unreferenced:
        message = f"Found {len(unreferenced)} wav files not referenced by metadata."
        if strict_unreferenced:
            errors.append(message)
        else:
            warnings.append(message)

    if not rows and not parse_errors:
        errors.append("metadata.csv has no usable rows.")

    summary = {
        "dataset_dir": str(dataset_dir),
        "metadata_path": str(metadata_path),
        "wav_dir": str(wav_dir),
        "metadata_row_count": len(rows),
        "metadata_column_counts": sorted(column_counts),
        "wav_count": len(wav_files),
        "referenced_wav_count": len(referenced_lower),
        "duplicate_filenames": sorted(set(duplicate_filenames)),
        "missing_references": sorted(set(missing_references)),
        "format_issues": format_issues,
        "unreferenced_wavs": unreferenced,
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0,
    }
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a dataset folder containing wavs/ and metadata.csv."
    )
    parser.add_argument(
        "--dataset_dir",
        default=".",
        help="Dataset directory path (contains wavs/ and metadata.csv).",
    )
    parser.add_argument(
        "--json_out",
        default="",
        help="Optional JSON output path for machine-readable validation summary.",
    )
    parser.add_argument(
        "--text_out",
        default="",
        help="Optional text summary output path.",
    )
    parser.add_argument(
        "--strict_unreferenced",
        action="store_true",
        help="Treat unreferenced wav files as errors.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    summary = validate_dataset(
        dataset_dir=Path(args.dataset_dir),
        strict_unreferenced=args.strict_unreferenced,
    )
    text_summary = build_text_summary(summary)

    LOGGER.info("\n%s", text_summary)

    json_blob = json.dumps(summary, ensure_ascii=False, indent=2)
    sys.stdout.write(f"{json_blob}\n")

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(f"{json_blob}\n", encoding="utf-8")
        LOGGER.info("Wrote JSON summary: %s", json_path)

    if args.text_out:
        text_path = Path(args.text_out).resolve()
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(f"{text_summary}\n", encoding="utf-8")
        LOGGER.info("Wrote text summary: %s", text_path)

    return 0 if summary["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
