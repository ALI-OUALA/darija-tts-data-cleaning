#!/usr/bin/env python3
"""Build release artifacts for Darija TTS dataset and project packaging."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_dataset import (  # noqa: E402
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLE_RATE,
    EXPECTED_SAMPLE_WIDTH,
    MetadataRow,
    inspect_wav,
    load_metadata_rows,
    validate_dataset,
)

LOGGER = logging.getLogger("pack_release")


def configure_logging(verbose: bool = False) -> None:
    """Configure logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    """Create a zip archive from a directory."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as handle:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(source_dir).as_posix()
            handle.write(file_path, arcname=relative)


def ensure_required_tts_export(tts_export_dir: Path) -> None:
    """Validate that required tts_export layout exists."""
    wav_dir = tts_export_dir / "wavs"
    metadata_path = tts_export_dir / "metadata.csv"
    if not tts_export_dir.exists():
        raise FileNotFoundError(f"tts_export directory missing: {tts_export_dir}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"tts_export/wavs directory missing: {wav_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"tts_export/metadata.csv missing: {metadata_path}")


def gather_rows_and_checks(tts_export_dir: Path) -> tuple[list[MetadataRow], dict[str, Path], list[str]]:
    """Load metadata rows and basic consistency checks."""
    metadata_path = tts_export_dir / "metadata.csv"
    wav_dir = tts_export_dir / "wavs"
    rows, parse_errors, _ = load_metadata_rows(metadata_path)
    if parse_errors:
        raise RuntimeError("Metadata parse errors:\n" + "\n".join(f"- {err}" for err in parse_errors))
    if not rows:
        raise RuntimeError("No usable metadata rows found in metadata.csv.")

    source_paths: dict[str, Path] = {}
    duplicate_rows: list[str] = []
    missing_rows: list[str] = []
    for row in rows:
        key = row.filename.lower()
        if key in source_paths:
            duplicate_rows.append(row.filename)
            continue
        wav_name = Path(row.filename).name
        src_wav = wav_dir / wav_name
        if not src_wav.exists():
            missing_rows.append(row.filename)
            continue
        source_paths[key] = src_wav

    problems: list[str] = []
    if duplicate_rows:
        problems.append(f"Duplicate metadata filenames ({len(duplicate_rows)}).")
    if missing_rows:
        problems.append(f"Metadata references missing wav files ({len(missing_rows)}).")
    return rows, source_paths, problems


def collect_audio_issues(source_paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Inspect source wav files for format mismatches."""
    issues: list[dict[str, Any]] = []
    for normalized_name, source_path in sorted(source_paths.items()):
        header, error = inspect_wav(source_path)
        if error:
            issues.append({"filename": normalized_name, "error": error})
            continue
        if (
            header["channels"] != EXPECTED_CHANNELS
            or header["sample_rate"] != EXPECTED_SAMPLE_RATE
            or header["sample_width"] != EXPECTED_SAMPLE_WIDTH
            or header["compression"] != "NONE"
        ):
            issues.append(
                {
                    "filename": normalized_name,
                    "channels": header["channels"],
                    "sample_rate": header["sample_rate"],
                    "sample_width": header["sample_width"],
                    "compression": header["compression"],
                }
            )
    return issues


def ffmpeg_fix_audio(source_path: Path, destination_path: Path) -> None:
    """Re-encode a wav file to mono/16kHz/PCM16 using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg re-encode failed for {source_path}:\n{proc.stderr.strip() or proc.stdout.strip()}"
        )


def write_dataset_metadata(rows: list[MetadataRow], out_path: Path) -> None:
    """Write normalized metadata while preserving transcript fields."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter="|",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        for row in rows:
            if row.column_count == 2:
                writer.writerow([row.filename, row.raw_text])
            else:
                writer.writerow([row.filename, row.raw_text, row.normalized_text or ""])


def write_dataset_readme(path: Path) -> None:
    """Write a short dataset package note."""
    content = textwrap.dedent(
        """
        # Dataset package

        This folder is the minimal training-ready dataset export.

        ## Contents
        - `wavs/*.wav`
        - `metadata.csv` (`filename|raw_text|normalized_text` or 2-column variant)

        ## Notes
        - Audio is expected to be mono, 16kHz, PCM16 WAV.
        - Text is preserved from the source export metadata.
        """
    ).strip()
    path.write_text(f"{content}\n", encoding="utf-8")


def build_dataset_ready(
    tts_export_dir: Path,
    dataset_dir: Path,
    fix_audio: bool,
) -> dict[str, Any]:
    """Build normalized minimal dataset folder from tts_export."""
    rows, source_paths, row_problems = gather_rows_and_checks(tts_export_dir)
    if row_problems:
        raise RuntimeError("Dataset integrity checks failed:\n" + "\n".join(f"- {p}" for p in row_problems))

    issues = collect_audio_issues(source_paths)
    if issues and not fix_audio:
        preview = "\n".join(f"- {item['filename']}: {item}" for item in issues[:25])
        raise RuntimeError(
            "Audio format issues found and --fix_audio was not enabled.\n"
            f"Expected mono/{EXPECTED_SAMPLE_RATE}Hz/PCM16.\n"
            f"Offending files ({len(issues)} total):\n{preview}"
        )

    if issues and fix_audio and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Audio fixes requested with --fix_audio, but ffmpeg is not available in PATH."
        )

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    wav_out_dir = dataset_dir / "wavs"
    wav_out_dir.mkdir(parents=True, exist_ok=True)

    issue_keys = {item["filename"].lower() for item in issues}
    for normalized_name, source_path in sorted(source_paths.items()):
        destination = wav_out_dir / source_path.name
        if normalized_name in issue_keys and fix_audio:
            ffmpeg_fix_audio(source_path, destination)
        else:
            shutil.copy2(source_path, destination)

    write_dataset_metadata(rows, dataset_dir / "metadata.csv")
    write_dataset_readme(dataset_dir / "README_DATASET.md")

    validation = validate_dataset(dataset_dir=dataset_dir, strict_unreferenced=False)
    if not validation["is_valid"]:
        details = "\n".join(f"- {err}" for err in validation["errors"])
        raise RuntimeError(f"Built dataset failed validation:\n{details}")
    return validation


def copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy a file if it exists."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def ensure_file(path: Path, content: str) -> None:
    """Write fallback file content if path does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def copy_reports(tts_export_dir: Path, reports_dir: Path) -> None:
    """Copy QC/alignment/report artifacts to reports directory."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(tts_export_dir.glob("*.csv")):
        if csv_path.name.lower() == "metadata.csv":
            continue
        shutil.copy2(csv_path, reports_dir / csv_path.name)

    logs_dir = tts_export_dir / "logs"
    if logs_dir.exists():
        for file_path in sorted(logs_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".csv", ".json", ".txt", ".log"}:
                continue
            relative = file_path.relative_to(logs_dir)
            destination = reports_dir / "logs" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, destination)


def build_dist_tree(out_dir: Path) -> str:
    """Build a short tree string for dist artifacts."""
    lines = [f"{out_dir.name}/"]
    if not out_dir.exists():
        return "\n".join(lines)
    entries = sorted(out_dir.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    for entry in entries:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"  - {entry.name}{suffix}")
        if entry.is_dir():
            children = sorted(entry.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            for child in children[:8]:
                child_suffix = "/" if child.is_dir() else ""
                lines.append(f"    - {child.name}{child_suffix}")
            if len(children) > 8:
                lines.append(f"    - ... ({len(children) - 8} more)")
    return "\n".join(lines)


def run_make_report(
    project_release_dir: Path,
    dataset_dir: Path,
) -> None:
    """Run make_report.py inside release project directory."""
    script_path = project_release_dir / "scripts" / "make_report.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--project_root",
        str(project_release_dir),
        "--dataset_dir",
        str(dataset_dir),
        "--reports_dir",
        str(project_release_dir / "reports"),
        "--docs_dir",
        str(project_release_dir / "docs"),
        "--readme_path",
        str(project_release_dir / "README.md"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "make_report.py failed:\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        LOGGER.info("make_report stdout:\n%s", proc.stdout.strip())
    if proc.stderr.strip():
        LOGGER.info("make_report stderr:\n%s", proc.stderr.strip())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build dataset-ready and repo-ready release packages from tts_export."
    )
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--tts_export_dir", default="./tts_export")
    parser.add_argument("--out_dir", default="./dist")
    parser.add_argument(
        "--fix_audio",
        action="store_true",
        help="Re-encode non-compliant wav files to mono/16kHz/PCM16 in dataset output.",
    )
    parser.add_argument(
        "--include_colab_refs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy darija_tts_whisperx_colab.ipynb/.py into colab/ if present.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    project_root = Path(args.project_root).resolve()
    tts_export_dir = Path(args.tts_export_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    dataset_dir = out_dir / "tts_dataset_ready"
    dataset_zip = out_dir / "tts_dataset_ready.zip"
    project_release_dir = out_dir / "tts_export_project"
    project_zip = out_dir / "tts_export_project.zip"

    ensure_required_tts_export(tts_export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building dataset-ready package...")
    validation = build_dataset_ready(
        tts_export_dir=tts_export_dir,
        dataset_dir=dataset_dir,
        fix_audio=args.fix_audio,
    )
    LOGGER.info(
        "Dataset validation passed. rows=%s wavs=%s",
        validation["metadata_row_count"],
        validation["wav_count"],
    )
    zip_directory(dataset_dir, dataset_zip)
    LOGGER.info("Dataset zip created: %s", dataset_zip)

    LOGGER.info("Building repository-ready package...")
    if project_release_dir.exists():
        shutil.rmtree(project_release_dir)
    (project_release_dir / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    (project_release_dir / "reports").mkdir(parents=True, exist_ok=True)
    (project_release_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (project_release_dir / "examples").mkdir(parents=True, exist_ok=True)
    (project_release_dir / "colab").mkdir(parents=True, exist_ok=True)

    required_files = [
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        ".gitignore",
        "requirements.txt",
    ]
    for file_name in required_files:
        src = project_root / file_name
        dst = project_release_dir / file_name
        copied = copy_if_exists(src, dst)
        if not copied:
            raise FileNotFoundError(
                f"Required file missing in project root: {src}. "
                "Create this file before running pack_release."
            )

    source_scripts = [
        "pack_release.py",
        "make_report.py",
        "validate_dataset.py",
    ]
    for script_name in source_scripts:
        src = project_root / "scripts" / script_name
        dst = project_release_dir / "scripts" / script_name
        if not src.exists():
            raise FileNotFoundError(f"Required script missing: {src}")
        shutil.copy2(src, dst)

    ensure_file(
        project_release_dir / "examples" / "sample_usage.md",
        (
            "# Sample trainer usage\n\n"
            "Use `metadata.csv` and `wavs/` from `dist/tts_dataset_ready` with your trainer."
        ),
    )
    copy_if_exists(
        project_root / "examples" / "sample_usage.md",
        project_release_dir / "examples" / "sample_usage.md",
    )

    ensure_file(
        project_release_dir / "colab" / "colab_steps.md",
        "# Colab steps\n\nSee project README for Colab workflow.",
    )
    copy_if_exists(
        project_root / "colab" / "colab_steps.md",
        project_release_dir / "colab" / "colab_steps.md",
    )

    if args.include_colab_refs:
        copy_if_exists(
            project_root / "darija_tts_whisperx_colab.ipynb",
            project_release_dir / "colab" / "darija_tts_whisperx_colab.ipynb",
        )
        copy_if_exists(
            project_root / "darija_tts_whisperx_colab.py",
            project_release_dir / "colab" / "darija_tts_whisperx_colab.py",
        )

    copy_reports(tts_export_dir=tts_export_dir, reports_dir=project_release_dir / "reports")
    run_make_report(project_release_dir=project_release_dir, dataset_dir=dataset_dir)

    zip_directory(project_release_dir, project_zip)
    LOGGER.info("Project zip created: %s", project_zip)

    LOGGER.info("Final artifacts:")
    LOGGER.info("- %s", dataset_dir)
    LOGGER.info("- %s", dataset_zip)
    LOGGER.info("- %s", project_release_dir)
    LOGGER.info("- %s", project_zip)
    LOGGER.info("Top-level tree:\n%s", build_dist_tree(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
