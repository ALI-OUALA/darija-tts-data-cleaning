#!/usr/bin/env python3
"""Standalone preflight checker for the Darija WhisperX pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from darija_tts_whisperx_colab import Config, run_preflight, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight check for local Windows setup.")
    parser.add_argument("--zip-path", default="./cafe-small-clean.zip")
    parser.add_argument("--output-dir", default="./tts_export")
    parser.add_argument("--extract-root", default="./data")
    parser.add_argument("--lockfile", default="./requirements.windows-py311.lock.txt")
    parser.add_argument("--min-disk-gb", type=float, default=20.0)
    parser.add_argument("--min-memory-gb", type=float, default=16.0)
    parser.add_argument("--strict-lock", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = Config(
        zip_path=args.zip_path,
        output_dir=args.output_dir,
        extract_root=args.extract_root,
        min_disk_gb=args.min_disk_gb,
        min_memory_gb=args.min_memory_gb,
    )
    lockfile = Path(args.lockfile)
    ok, report = run_preflight(cfg, lockfile, strict_lock=args.strict_lock)

    logs_dir = Path(cfg.output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / "preflight_report.json"
    write_json(report_path, report)

    print(f"Preflight report written to: {report_path}")
    for c in report["checks"]:
        state = "OK" if c["ok"] else "FAIL"
        print(f"[{state}] {c['name']}: {c['detail']}")
        if (not c["ok"]) and c.get("hint"):
            print(f"      hint: {c['hint']}")

    if report.get("lock_mismatches"):
        print("Lockfile mismatches:")
        for m in report["lock_mismatches"][:30]:
            print(f"  - {m['package']}: expected {m['expected']}, installed {m['installed']}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
