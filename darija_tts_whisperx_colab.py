#!/usr/bin/env python3
"""
Darija TTS Dataset Builder with WhisperX forced alignment (Windows hardened).

Modes:
- --install
- --preflight
- --smoke-test
- --run-full
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOCKFILE = Path("requirements.windows-py311.lock.txt")


class PipelineError(RuntimeError):
    """Structured pipeline error."""

    def __init__(self, stage: str, code: str, message: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(message)


@dataclass
class Config:
    zip_path: str = "./cafe-small-clean.zip"
    extract_root: str = "./data"
    output_dir: str = "./tts_export"
    target_sr: int = 16000

    min_dur: float = 1.5
    max_dur: float = 15.0
    preferred_max_dur: float = 12.0

    pause_gap_threshold: float = 0.6
    coverage_threshold: float = 0.70
    silence_trim_ms: int = 200
    padding_ms: int = 80

    whisper_model_size: str = "small"
    batch_size: int = 4
    compute_type: str = "auto"
    device_preference: str = "auto"  # auto|xpu|cuda|cpu

    min_disk_gb: float = 20.0
    min_memory_gb: float = 16.0
    resume: bool = False
    max_files: int | None = None
    skip_optional_hf: bool = True
    seed: int = 42


WORD_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF']+", re.UNICODE)
PUNCT_CHARS = set(".!?؟،؛…")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_pkg_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def parse_lockfile(lockfile: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    if not lockfile.exists():
        return locked
    for line in lockfile.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "==" not in s:
            continue
        pkg, ver = s.split("==", 1)
        locked[normalize_pkg_name(pkg)] = ver.strip()
    return locked


def get_installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def get_total_memory_gb() -> float:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return 0.0
        return stat.ullTotalPhys / (1024**3)

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024**3)
    except Exception:
        return 0.0


def ensure_python_311() -> None:
    v = sys.version_info
    if not (v.major == 3 and v.minor == 11):
        raise PipelineError(
            "preflight",
            "E_PYTHON_VERSION",
            (
                f"Python 3.11.x is required; found {v.major}.{v.minor}.{v.micro}. "
                "Use scripts/setup_windows_py311.ps1."
            ),
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def word_tokenize_for_coverage(text: str) -> list[str]:
    return WORD_RE.findall(str(text))


def split_text_by_punctuation(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for ch in str(text):
        current.append(ch)
        if ch in PUNCT_CHARS:
            c = "".join(current).strip()
            if c:
                chunks.append(c)
            current = []
    tail = "".join(current).strip()
    if tail:
        chunks.append(tail)
    if not chunks and str(text).strip():
        chunks = [str(text).strip()]
    return chunks


def norm_rel_path(s: str) -> str:
    s = s.replace("\\", "/").strip()
    s = re.sub(r"^\./+", "", s)
    s = re.sub(r"/+", "/", s)
    return s.lower()


def run_preflight(cfg: Config, lockfile: Path, strict_lock: bool = True) -> tuple[bool, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str, hint: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "hint": hint})

    py_ok = sys.version_info.major == 3 and sys.version_info.minor == 11
    add_check(
        "python_version",
        py_ok,
        f"Detected {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Use Python 3.11.x for deterministic compatibility.",
    )

    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_ok = ffmpeg_path is not None
    add_check(
        "ffmpeg_in_path",
        ffmpeg_ok,
        f"ffmpeg path: {ffmpeg_path}" if ffmpeg_ok else "ffmpeg not found in PATH",
        "Install ffmpeg and restart terminal.",
    )
    if ffmpeg_ok:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        line = proc.stdout.splitlines()[0] if proc.stdout else "unknown"
        add_check("ffmpeg_version", proc.returncode == 0, line, "ffmpeg must be runnable.")

    zip_path = Path(cfg.zip_path)
    add_check("zip_exists", zip_path.exists(), str(zip_path), "Place dataset zip at configured path.")
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                _ = zf.namelist()[:3]
            add_check("zip_readable", True, "Zip archive can be opened.")
        except Exception as e:
            add_check("zip_readable", False, f"Zip open error: {e}", "Re-copy zip file.")

    out = Path(cfg.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add_check("output_writable", True, str(out))
    except Exception as e:
        add_check("output_writable", False, f"{out}: {e}", "Choose a writable output directory.")

    try:
        disk = shutil.disk_usage(out if out.exists() else Path("."))
        free_gb = disk.free / (1024**3)
        add_check(
            "disk_space",
            free_gb >= cfg.min_disk_gb,
            f"Free disk: {free_gb:.2f} GB (required >= {cfg.min_disk_gb:.2f} GB)",
            "Free disk space or reduce output footprint.",
        )
    except Exception as e:
        add_check("disk_space", False, str(e), "Could not inspect free disk space.")

    mem_gb = get_total_memory_gb()
    add_check(
        "memory_total",
        mem_gb >= cfg.min_memory_gb,
        f"RAM: {mem_gb:.2f} GB (required >= {cfg.min_memory_gb:.2f} GB)",
        "Lower model size/batch if memory is below threshold.",
    )

    add_check("lockfile_exists", lockfile.exists(), str(lockfile), "Generate lockfile or use provided one.")
    mismatches: list[dict[str, str]] = []
    if lockfile.exists():
        locked = parse_lockfile(lockfile)
        for pkg, expected in locked.items():
            installed = get_installed_version(pkg)
            if installed is None:
                mismatches.append(
                    {"package": pkg, "expected": expected, "installed": "MISSING", "status": "missing"}
                )
            elif installed != expected:
                mismatches.append(
                    {"package": pkg, "expected": expected, "installed": installed, "status": "mismatch"}
                )
        add_check(
            "lockfile_versions",
            len(mismatches) == 0 if strict_lock else True,
            f"Version mismatches: {len(mismatches)}",
            "Run install mode to sync package versions.",
        )

    report = {
        "timestamp_utc": utc_now_iso(),
        "host": {"platform": platform.platform(), "python": sys.version},
        "config": asdict(cfg),
        "checks": checks,
        "lock_mismatches": mismatches,
        "all_ok": all(c["ok"] for c in checks),
    }
    return report["all_ok"], report


def ensure_output_layout(cfg: Config) -> dict[str, Path]:
    output_dir = Path(cfg.output_dir)
    paths = {
        "output_dir": output_dir,
        "wavs_dir": output_dir / "wavs",
        "intermediate_dir": output_dir / "intermediate",
        "logs_dir": output_dir / "logs",
        "aligned_words_dir": output_dir / "intermediate" / "aligned_words",
        "temp_dir": output_dir / "intermediate" / "temp_audio",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def import_runtime_deps() -> dict[str, Any]:
    import librosa
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    import whisperx
    from tqdm.auto import tqdm

    return {
        "librosa": librosa,
        "np": np,
        "pd": pd,
        "sf": sf,
        "torch": torch,
        "whisperx": whisperx,
        "tqdm": tqdm,
    }


def detect_runtime_devices(torch_mod: Any, preference: str) -> dict[str, Any]:
    xpu_available = bool(hasattr(torch_mod, "xpu") and torch_mod.xpu.is_available())
    cuda_available = bool(torch_mod.cuda.is_available())
    probe_order = ["xpu", "cuda", "cpu"]

    pref = preference.lower().strip()
    if pref not in {"auto", "xpu", "cuda", "cpu"}:
        pref = "auto"
    order = probe_order if pref == "auto" else [pref] + [d for d in probe_order if d != pref]

    selected_runtime = "cpu"
    for d in order:
        if d == "xpu" and xpu_available:
            selected_runtime = "xpu"
            break
        if d == "cuda" and cuda_available:
            selected_runtime = "cuda"
            break
        if d == "cpu":
            selected_runtime = "cpu"
            break

    asr_device = "cuda" if cuda_available and selected_runtime == "cuda" else "cpu"
    if selected_runtime == "xpu":
        align_candidates = ["xpu"] + (["cuda"] if cuda_available else []) + ["cpu"]
    elif selected_runtime == "cuda":
        align_candidates = ["cuda", "cpu"]
    else:
        align_candidates = ["cpu"]

    return {
        "probe_order": probe_order,
        "preference": pref,
        "xpu_available": xpu_available,
        "cuda_available": cuda_available,
        "selected_runtime_device": selected_runtime,
        "asr_device": asr_device,
        "align_candidates": align_candidates,
    }


def configure_torch_checkpoint_loading(torch_mod: Any) -> dict[str, Any]:
    """
    PyTorch >=2.6 defaults torch.load(weights_only=True), which breaks
    pyannote checkpoints used by WhisperX VAD/alignment.
    """

    info: dict[str, Any] = {
        "forced_no_weights_only": False,
        "added_safe_globals": [],
        "notes": [],
    }

    force_weights_only = str(os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "")).strip().lower() in {"1", "true"}
    if force_weights_only:
        info["notes"].append("TORCH_FORCE_WEIGHTS_ONLY_LOAD is set; not overriding.")
    else:
        no_weights_only = str(os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "")).strip().lower() in {"1", "true"}
        if not no_weights_only:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            info["forced_no_weights_only"] = True
            info["notes"].append("Set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for pyannote compatibility.")
        else:
            info["notes"].append("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD already enabled.")

    try:
        add_safe = getattr(torch_mod.serialization, "add_safe_globals", None)
        if callable(add_safe):
            from omegaconf import DictConfig, ListConfig

            add_safe([ListConfig, DictConfig])
            info["added_safe_globals"] = [
                "omegaconf.ListConfig",
                "omegaconf.DictConfig",
            ]
    except Exception as e:
        info["notes"].append(f"Could not register safe globals: {type(e).__name__}: {e}")

    return info


def find_transcript_csv(root: Path, pd: Any) -> Path:
    candidates = sorted(root.rglob("*.csv"))
    valid = []
    for p in candidates:
        try:
            head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = {str(c).strip().lower() for c in head.columns}
        if {"path", "text"}.issubset(cols):
            valid.append(p)
    if not valid:
        raise PipelineError("load_dataset", "E_NO_TRANSCRIPT_CSV", "No CSV with columns path,text found.")
    return valid[0]


def resolve_audio_path(raw_path: str, rel_index: dict[str, list[Path]], basename_index: dict[str, list[Path]]) -> tuple[str | None, str]:
    raw = raw_path.strip().replace("\\", "/")
    raw = re.sub(r"^\./+", "", raw)

    candidates = [(raw, "direct_norm")]
    if raw.lower().startswith("cafe/"):
        candidates.append((raw[5:], "strip_cafe"))
    prefix = "cafe/cafe-small-clean/"
    if raw.lower().startswith(prefix):
        candidates.append((raw[len(prefix):], "strip_cafe_cafe-small-clean"))

    for cand, method in candidates:
        k = norm_rel_path(cand)
        hits = rel_index.get(k, [])
        if len(hits) == 1:
            return str(hits[0]), method
        if len(hits) > 1:
            return str(hits[0]), method + "_ambiguous_pick_first"

    base = Path(raw).name.lower()
    base_hits = basename_index.get(base, [])
    if len(base_hits) == 1:
        return str(base_hits[0]), "basename_unique"
    if len(base_hits) > 1:
        return None, "basename_ambiguous"
    return None, "not_found"


def make_fixed_windows(audio_duration: float, window_sec: float = 20.0) -> list[dict[str, Any]]:
    windows = []
    t = 0.0
    while t < audio_duration:
        e = min(audio_duration, t + window_sec)
        windows.append({"start": t, "end": e, "text": ""})
        t = e
    if not windows and audio_duration > 0:
        windows = [{"start": 0.0, "end": audio_duration, "text": ""}]
    return windows


def distribute_transcript_over_segments(transcript: str, rough_segments: list[dict[str, Any]], audio_duration: float, np: Any) -> list[dict[str, Any]]:
    words = transcript.split()
    if not words:
        return []

    segments = list(rough_segments) if rough_segments else make_fixed_windows(audio_duration=audio_duration, window_sec=20.0)
    durations = [max(0.01, float(seg["end"] - seg["start"])) for seg in segments]
    total_dur = float(sum(durations))
    alloc = [max(1, int(round(len(words) * (d / total_dur)))) for d in durations]

    while sum(alloc) > len(words):
        idx = int(np.argmax(alloc))
        if alloc[idx] > 1:
            alloc[idx] -= 1
        else:
            break
    while sum(alloc) < len(words):
        idx = int(np.argmax(durations))
        alloc[idx] += 1

    forced_segments: list[dict[str, Any]] = []
    cursor = 0
    for seg, n_words in zip(segments, alloc):
        chunk_words = words[cursor : cursor + n_words]
        cursor += n_words
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            forced_segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": chunk_text})

    if cursor < len(words):
        leftover = " ".join(words[cursor:]).strip()
        if forced_segments:
            forced_segments[-1]["text"] = (forced_segments[-1]["text"] + " " + leftover).strip()
        else:
            forced_segments.append({"start": 0.0, "end": float(audio_duration), "text": leftover})

    if not forced_segments:
        forced_segments = [{"start": 0.0, "end": float(audio_duration), "text": transcript.strip()}]
    return forced_segments


def words_duration(words: list[dict[str, Any]]) -> float:
    if not words:
        return 0.0
    return float(words[-1]["end"] - words[0]["start"])


def text_from_words(words: list[dict[str, Any]]) -> str:
    toks = []
    for w in words:
        t = str(w.get("orig_word") or w.get("word") or "").strip()
        if t:
            toks.append(t)
    txt = " ".join(toks)
    txt = re.sub(r"\s+([\.\!\?؟،؛…])", r"\1", txt)
    return normalize_whitespace(txt)


def sentence_word_counts(raw_text: str) -> tuple[list[str], list[int]]:
    sents = split_text_by_punctuation(raw_text)
    counts = [max(1, len(word_tokenize_for_coverage(s))) for s in sents]
    return sents, counts


def allocate_words_to_sentences(aligned_words: list[dict[str, Any]], raw_text: str) -> list[dict[str, Any]]:
    sents, counts = sentence_word_counts(raw_text)
    if not aligned_words:
        return []

    chunks = []
    cursor = 0
    n = len(aligned_words)
    for i, (sent_text, cnt) in enumerate(zip(sents, counts)):
        part = aligned_words[cursor:] if i == len(sents) - 1 else aligned_words[cursor : min(n, cursor + cnt)]
        cursor += cnt
        if part:
            chunks.append({"text": normalize_whitespace(sent_text), "words": part})

    if not chunks:
        chunks = [{"text": text_from_words(aligned_words), "words": aligned_words}]
    assigned = sum(len(c["words"]) for c in chunks)
    if assigned < n and chunks:
        chunks[-1]["words"].extend(aligned_words[assigned:])
    return chunks


def split_words_by_pause(words: list[dict[str, Any]], gap_threshold: float) -> list[list[dict[str, Any]]]:
    if not words:
        return []
    parts = []
    start_idx = 0
    for i in range(len(words) - 1):
        gap = float(words[i + 1]["start"] - words[i]["end"])
        if gap > gap_threshold:
            parts.append(words[start_idx : i + 1])
            start_idx = i + 1
    parts.append(words[start_idx:])
    return [p for p in parts if p]


def split_chunk_to_target(words: list[dict[str, Any]], target_max_dur: float) -> list[list[dict[str, Any]]]:
    if not words:
        return []
    if words_duration(words) <= target_max_dur or len(words) <= 1:
        return [words]

    candidates = []
    for i in range(len(words) - 1):
        gap = float(words[i + 1]["start"] - words[i]["end"])
        left_dur = float(words[i]["end"] - words[0]["start"])
        right_dur = float(words[-1]["end"] - words[i + 1]["start"])
        balance = abs(left_dur - right_dur)
        candidates.append((i, gap, balance))
    split_idx = sorted(candidates, key=lambda x: (-x[1], x[2]))[0][0] + 1 if candidates else len(words) // 2
    return split_chunk_to_target(words[:split_idx], target_max_dur) + split_chunk_to_target(words[split_idx:], target_max_dur)


def make_segment(words: list[dict[str, Any]], default_text: str | None = None) -> dict[str, Any]:
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "duration": float(words[-1]["end"] - words[0]["start"]),
        "words": words,
        "text": normalize_whitespace(default_text if default_text else text_from_words(words)),
    }


def combine_segments(seg_a: dict[str, Any], seg_b: dict[str, Any]) -> dict[str, Any]:
    words = sorted(seg_a["words"] + seg_b["words"], key=lambda x: (x["start"], x["end"]))
    return make_segment(words)


def trim_silence_with_cap(y: Any, sr: int, librosa_mod: Any, max_trim_ms: int = 200, db_threshold: float = -35.0) -> Any:
    if y.size == 0:
        return y
    frame_length = max(256, int(sr * 0.02))
    hop_length = max(128, int(sr * 0.01))
    rms = librosa_mod.feature.rms(y=y.astype("float32"), frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return y
    max_rms = float(rms.max())
    if max_rms <= 1e-8:
        return y

    threshold = max_rms * (10 ** (db_threshold / 20.0))
    active = (rms >= threshold).nonzero()[0]
    if active.size == 0:
        return y

    start_sample = int(librosa_mod.frames_to_samples(int(active[0]), hop_length=hop_length))
    end_sample = int(librosa_mod.frames_to_samples(int(active[-1]) + 1, hop_length=hop_length))
    start_sample = max(0, min(start_sample, len(y)))
    end_sample = max(start_sample, min(end_sample, len(y)))

    max_trim = int(sr * (max_trim_ms / 1000.0))
    left_trim = min(start_sample, max_trim)
    right_trim = min(len(y) - end_sample, max_trim)
    new_start = left_trim
    new_end = len(y) - right_trim
    if new_end <= new_start:
        return y
    return y[new_start:new_end]


def ffmpeg_extract_to_wav16k_mono(src_path: str, start_sec: float, end_sec: float, out_path: Path, sr: int) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        src_path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def run_install(lockfile: Path) -> None:
    ensure_python_311()
    if not lockfile.exists():
        raise PipelineError("install", "E_LOCKFILE_MISSING", f"Lockfile not found: {lockfile}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input", "--no-compile", "-r", str(lockfile)],
        check=True,
    )
    print("Dependency installation completed.")


def run_pipeline(cfg: Config, mode: str, lockfile: Path) -> dict[str, Any]:
    deps = import_runtime_deps()
    librosa = deps["librosa"]
    np = deps["np"]
    pd = deps["pd"]
    sf = deps["sf"]
    torch = deps["torch"]
    whisperx = deps["whisperx"]
    tqdm = deps["tqdm"]

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    paths = ensure_output_layout(cfg)
    output_dir = paths["output_dir"]
    wavs_dir = paths["wavs_dir"]
    logs_dir = paths["logs_dir"]
    aligned_words_dir = paths["aligned_words_dir"]
    temp_dir = paths["temp_dir"]

    runtime_info: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "mode": mode,
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "lockfile": str(lockfile),
        "lockfile_exists": lockfile.exists(),
    }

    device_info = detect_runtime_devices(torch, cfg.device_preference)
    runtime_info["device_info"] = device_info
    runtime_info["torch_checkpoint_loading"] = configure_torch_checkpoint_loading(torch)

    extract_root = Path(cfg.extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    zip_path = Path(cfg.zip_path)
    if not zip_path.exists():
        raise PipelineError("load_dataset", "E_ZIP_NOT_FOUND", f"Zip not found: {zip_path}")
    if cfg.resume and any(extract_root.rglob("*.wav")):
        print(f"[resume] Skipping unzip; data already exists in {extract_root}")
    else:
        print(f"Unzipping {zip_path} -> {extract_root}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

    transcript_csv_path = find_transcript_csv(extract_root, pd)
    transcripts_df = pd.read_csv(transcript_csv_path)
    transcripts_df.columns = [str(c).strip().lower() for c in transcripts_df.columns]
    transcripts_df = transcripts_df[["path", "text"]].copy()
    transcripts_df["path"] = transcripts_df["path"].fillna("").astype(str).str.strip()
    transcripts_df["text"] = transcripts_df["text"].fillna("").astype(str)
    transcripts_df = transcripts_df[
        ~((transcripts_df["path"] == "") & (transcripts_df["text"].str.strip() == ""))
    ].copy()
    transcripts_df = transcripts_df.reset_index(drop=True)
    transcripts_df["row_id"] = transcripts_df.index.astype(int)
    print(f"Transcript rows: {len(transcripts_df)}")

    all_wavs = sorted(extract_root.rglob("*.wav"))
    rel_index: dict[str, list[Path]] = {}
    basename_index: dict[str, list[Path]] = {}
    for wav in all_wavs:
        rel = norm_rel_path(str(wav.relative_to(extract_root)))
        rel_index.setdefault(rel, []).append(wav)
        basename_index.setdefault(wav.name.lower(), []).append(wav)

    resolved_paths = []
    resolve_methods = []
    for p in transcripts_df["path"].tolist():
        rp, rm = resolve_audio_path(p, rel_index, basename_index)
        resolved_paths.append(rp)
        resolve_methods.append(rm)
    transcripts_df["resolved_audio_path"] = resolved_paths
    transcripts_df["resolve_method"] = resolve_methods
    transcripts_df["audio_found"] = transcripts_df["resolved_audio_path"].notna()
    transcripts_df[
        ["row_id", "path", "text", "resolved_audio_path", "resolve_method", "audio_found"]
    ].to_csv(logs_dir / "path_resolution_report.csv", index=False)

    qc_records: list[dict[str, Any]] = []
    for row in tqdm(transcripts_df.itertuples(index=False), total=len(transcripts_df), desc="Pre-QC"):
        if not row.audio_found:
            continue
        audio_path = Path(row.resolved_audio_path)
        try:
            y, sr = sf.read(audio_path, always_2d=True)
            n_samples, n_channels = y.shape
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            duration = float(n_samples / sr) if sr > 0 else 0.0
            qc_records.append(
                {
                    "row_id": int(row.row_id),
                    "audio_path": str(audio_path),
                    "duration_sec": duration,
                    "sample_rate": int(sr),
                    "channels": int(n_channels),
                    "peak_amplitude": peak,
                    "is_clipping": bool(peak >= 0.999),
                }
            )
        except Exception as e:
            qc_records.append(
                {
                    "row_id": int(row.row_id),
                    "audio_path": str(audio_path),
                    "duration_sec": np.nan,
                    "sample_rate": np.nan,
                    "channels": np.nan,
                    "peak_amplitude": np.nan,
                    "is_clipping": np.nan,
                    "qc_error": f"{type(e).__name__}: {e}",
                }
            )
    qc_pre_df = pd.DataFrame(qc_records)
    qc_pre_df.to_csv(output_dir / "qc_pre_alignment.csv", index=False)

    asr_compute_type = cfg.compute_type
    if asr_compute_type == "auto":
        asr_compute_type = "float16" if device_info["asr_device"] == "cuda" else "int8"
    asr_batch_size = cfg.batch_size if device_info["asr_device"] == "cuda" else max(1, min(2, cfg.batch_size))

    print(
        f"Runtime selection: runtime={device_info['selected_runtime_device']} "
        f"asr={device_info['asr_device']} asr_compute={asr_compute_type} batch={asr_batch_size}"
    )

    asr_model = whisperx.load_model(cfg.whisper_model_size, device=device_info["asr_device"], compute_type=asr_compute_type)
    asr_cpu_model = None

    align_models: dict[str, tuple[Any, Any]] = {}
    align_load_errors: list[dict[str, str]] = []

    def get_align_model_for_device(device_name: str) -> tuple[Any, Any]:
        if device_name in align_models:
            return align_models[device_name]
        try:
            model, metadata = whisperx.load_align_model(language_code="ar", device=device_name)
            align_models[device_name] = (model, metadata)
            return model, metadata
        except Exception as e:
            align_load_errors.append({"device": device_name, "error": f"{type(e).__name__}: {e}"})
            raise

    selected_align_device = None
    for candidate in device_info["align_candidates"]:
        try:
            get_align_model_for_device(candidate)
            selected_align_device = candidate
            break
        except Exception:
            continue
    if selected_align_device is None:
        raise PipelineError("alignment_model_load", "E_ALIGN_MODEL_LOAD", f"Unable to load alignment model: {align_load_errors}")

    runtime_info["align_model_device"] = selected_align_device
    runtime_info["align_load_errors"] = align_load_errors
    write_json(logs_dir / "runtime_info.json", runtime_info)

    rows_for_alignment = transcripts_df[transcripts_df["audio_found"]].copy()
    if mode == "smoke-test" and cfg.max_files is None:
        cfg.max_files = 3
    if cfg.max_files is not None:
        rows_for_alignment = rows_for_alignment.head(int(cfg.max_files)).copy()

    alignment_report_path = output_dir / "alignment_report.csv"
    prior_alignment_df = pd.DataFrame()
    prior_done_ids: set[int] = set()
    aligned_ok_cache: dict[int, dict[str, Any]] = {}
    if cfg.resume and alignment_report_path.exists():
        try:
            prior_alignment_df = pd.read_csv(alignment_report_path)
            if "row_id" in prior_alignment_df.columns and "status" in prior_alignment_df.columns:
                prior_done_ids = set(
                    prior_alignment_df[prior_alignment_df["status"].isin(["ok", "needs_review"])]["row_id"].astype(int).tolist()
                )
        except Exception:
            prior_alignment_df = pd.DataFrame()
            prior_done_ids = set()

    if cfg.resume:
        for p in sorted(aligned_words_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                if payload.get("status") == "ok":
                    rid = int(payload["row_id"])
                    aligned_ok_cache[rid] = {
                        "row_id": rid,
                        "audio_path": payload["audio_path"],
                        "duration": float(payload["duration"]),
                        "raw_text": payload["raw_text"],
                        "aligned_words": payload["aligned_words"],
                    }
            except Exception:
                continue
    if prior_done_ids:
        rows_for_alignment = rows_for_alignment[~rows_for_alignment["row_id"].isin(prior_done_ids)].copy()

    qc_duration_map = {
        int(r.row_id): float(r.duration_sec)
        for r in qc_pre_df.dropna(subset=["duration_sec"]).itertuples(index=False)
    }

    alignment_rows: list[dict[str, Any]] = []
    processing_errors: list[dict[str, Any]] = []
    failures_by_stage: dict[str, int] = {}
    asr_cpu_fallback_used = False
    align_cpu_fallback_used = False

    def incr_failure(stage: str) -> None:
        failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1

    def build_rough_segments(audio_path: str) -> list[dict[str, Any]]:
        nonlocal asr_cpu_model, asr_cpu_fallback_used
        audio = whisperx.load_audio(audio_path)
        try:
            result = asr_model.transcribe(audio, batch_size=asr_batch_size, language="ar")
        except Exception:
            if device_info["asr_device"] != "cpu":
                asr_cpu_fallback_used = True
                if asr_cpu_model is None:
                    asr_cpu_model = whisperx.load_model(cfg.whisper_model_size, device="cpu", compute_type="int8")
                result = asr_cpu_model.transcribe(audio, batch_size=1, language="ar")
            else:
                raise
        rough = []
        for seg in result.get("segments", []):
            try:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
            except Exception:
                continue
            if e <= s:
                continue
            rough.append({"start": s, "end": e, "text": str(seg.get("text", "")).strip()})
        return rough

    for row in tqdm(rows_for_alignment.itertuples(index=False), total=len(rows_for_alignment), desc="Forced alignment"):
        row_id = int(row.row_id)
        raw_text = str(row.text)
        audio_path = str(row.resolved_audio_path)
        try:
            duration = float(qc_duration_map.get(row_id, 0.0))
            if duration <= 0:
                info = sf.info(audio_path)
                duration = float(info.frames / info.samplerate)

            if normalize_whitespace(raw_text) == "":
                alignment_rows.append(
                    {
                        "row_id": row_id,
                        "audio_path": audio_path,
                        "total_words": 0,
                        "aligned_words": 0,
                        "coverage": 0.0,
                        "status": "needs_review",
                        "reason": "empty_text",
                    }
                )
                continue

            rough_segments = build_rough_segments(audio_path)
            forced_segments = distribute_transcript_over_segments(raw_text, rough_segments, duration, np)
            audio_arr = whisperx.load_audio(audio_path)

            try:
                align_model, align_metadata = get_align_model_for_device(selected_align_device)
                aligned_result = whisperx.align(
                    forced_segments,
                    align_model,
                    align_metadata,
                    audio_arr,
                    selected_align_device,
                    return_char_alignments=False,
                )
                used_align_device = selected_align_device
            except Exception:
                if selected_align_device != "cpu":
                    align_cpu_fallback_used = True
                    cpu_align_model, cpu_align_metadata = get_align_model_for_device("cpu")
                    aligned_result = whisperx.align(
                        forced_segments,
                        cpu_align_model,
                        cpu_align_metadata,
                        audio_arr,
                        "cpu",
                        return_char_alignments=False,
                    )
                    used_align_device = "cpu"
                else:
                    raise

            word_segments = aligned_result.get("word_segments")
            if word_segments is None:
                word_segments = []
                for s in aligned_result.get("segments", []):
                    word_segments.extend(s.get("words", []))

            valid_words = []
            for w in word_segments:
                try:
                    ws = float(w.get("start"))
                    we = float(w.get("end"))
                except Exception:
                    continue
                if not np.isfinite(ws) or not np.isfinite(we) or we <= ws:
                    continue
                token = str(w.get("word", "")).strip()
                valid_words.append({"start": ws, "end": we, "word": token})

            orig_words = raw_text.split()
            for i, w in enumerate(valid_words):
                w["orig_word"] = orig_words[i] if i < len(orig_words) else w.get("word", "")

            total_words = len(word_tokenize_for_coverage(raw_text))
            aligned_words = len(valid_words)
            coverage = float(aligned_words / max(1, total_words))
            status = "ok" if coverage >= cfg.coverage_threshold else "needs_review"

            aligned_json_path = aligned_words_dir / f"{row_id:06d}.json"
            aligned_json_path.write_text(
                json.dumps(
                    {
                        "row_id": row_id,
                        "audio_path": audio_path,
                        "duration": duration,
                        "raw_text": raw_text,
                        "coverage": coverage,
                        "status": status,
                        "align_device_used": used_align_device,
                        "forced_segments": forced_segments,
                        "aligned_words": valid_words,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            alignment_rows.append(
                {
                    "row_id": row_id,
                    "audio_path": audio_path,
                    "total_words": total_words,
                    "aligned_words": aligned_words,
                    "coverage": coverage,
                    "status": status,
                    "reason": "",
                    "align_device_used": used_align_device,
                    "aligned_json": str(aligned_json_path),
                }
            )

            if status == "ok":
                aligned_ok_cache[row_id] = {
                    "row_id": row_id,
                    "audio_path": audio_path,
                    "duration": duration,
                    "raw_text": raw_text,
                    "aligned_words": valid_words,
                }
        except Exception as e:
            incr_failure("alignment")
            msg = f"{type(e).__name__}: {e}"
            alignment_rows.append(
                {
                    "row_id": row_id,
                    "audio_path": audio_path,
                    "total_words": np.nan,
                    "aligned_words": np.nan,
                    "coverage": 0.0,
                    "status": "error",
                    "reason": msg,
                }
            )
            processing_errors.append(
                {
                    "row_id": row_id,
                    "audio_path": audio_path,
                    "stage": "alignment",
                    "code": "E_ALIGN_FILE",
                    "error": msg,
                    "traceback": traceback.format_exc(),
                }
            )

    alignment_report_df = pd.DataFrame(alignment_rows)
    if cfg.resume and not prior_alignment_df.empty:
        alignment_report_df = pd.concat([prior_alignment_df, alignment_report_df], ignore_index=True)
        alignment_report_df = alignment_report_df.sort_values(by=["row_id"]).drop_duplicates(subset=["row_id"], keep="last")
    alignment_report_df.to_csv(alignment_report_path, index=False)

    segment_candidates: list[dict[str, Any]] = []
    dropped_segments: list[dict[str, Any]] = []
    for row_id, rec in tqdm(aligned_ok_cache.items(), total=len(aligned_ok_cache), desc="Segmentation"):
        aligned_words = rec["aligned_words"]
        raw_text = rec["raw_text"]
        if not aligned_words:
            dropped_segments.append({"row_id": row_id, "segment_id": "", "reason": "no_aligned_words", "duration": 0.0, "text": raw_text})
            continue

        sentence_chunks = allocate_words_to_sentences(aligned_words, raw_text)
        pause_chunks = []
        for c in sentence_chunks:
            parts = split_words_by_pause(c["words"], cfg.pause_gap_threshold)
            if len(parts) == 1:
                pause_chunks.append(make_segment(parts[0], default_text=c["text"]))
            else:
                for p in parts:
                    pause_chunks.append(make_segment(p))

        shaped = []
        for seg in pause_chunks:
            if seg["duration"] > cfg.preferred_max_dur:
                for sp in split_chunk_to_target(seg["words"], cfg.preferred_max_dur):
                    shaped.append(make_segment(sp))
            else:
                shaped.append(seg)

        shaped = sorted(shaped, key=lambda x: (x["start"], x["end"]))
        merged = []
        i = 0
        while i < len(shaped):
            seg = shaped[i]
            if seg["duration"] < cfg.min_dur:
                merged_flag = False
                if i + 1 < len(shaped):
                    comb = combine_segments(seg, shaped[i + 1])
                    if comb["duration"] <= cfg.preferred_max_dur:
                        shaped[i + 1] = comb
                        merged_flag = True
                if not merged_flag and merged:
                    comb = combine_segments(merged[-1], seg)
                    if comb["duration"] <= cfg.preferred_max_dur:
                        merged[-1] = comb
                        merged_flag = True
                if not merged_flag:
                    dropped_segments.append(
                        {
                            "row_id": row_id,
                            "segment_id": "",
                            "reason": "too_short_unmergeable",
                            "duration": seg["duration"],
                            "text": seg["text"],
                        }
                    )
                i += 1
                continue
            merged.append(seg)
            i += 1

        valid = []
        for seg in merged:
            if seg["duration"] < cfg.min_dur:
                dropped_segments.append(
                    {
                        "row_id": row_id,
                        "segment_id": "",
                        "reason": "below_min_duration",
                        "duration": seg["duration"],
                        "text": seg["text"],
                    }
                )
                continue
            if seg["duration"] > cfg.max_dur:
                parts = split_chunk_to_target(seg["words"], cfg.max_dur)
                for hp in parts:
                    hp_seg = make_segment(hp)
                    if hp_seg["duration"] > cfg.max_dur:
                        dropped_segments.append(
                            {
                                "row_id": row_id,
                                "segment_id": "",
                                "reason": "above_hard_max_after_split",
                                "duration": hp_seg["duration"],
                                "text": hp_seg["text"],
                            }
                        )
                    elif hp_seg["duration"] < cfg.min_dur:
                        dropped_segments.append(
                            {
                                "row_id": row_id,
                                "segment_id": "",
                                "reason": "below_min_after_hard_split",
                                "duration": hp_seg["duration"],
                                "text": hp_seg["text"],
                            }
                        )
                    else:
                        valid.append(hp_seg)
                continue
            valid.append(seg)

        for seg_idx, seg in enumerate(valid):
            segment_candidates.append(
                {
                    "row_id": row_id,
                    "audio_path": rec["audio_path"],
                    "file_duration": rec["duration"],
                    "segment_index": seg_idx,
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["duration"],
                    "text": seg["text"],
                }
            )

    dropped_segments_df = pd.DataFrame(dropped_segments)

    metadata_rows: list[dict[str, str]] = []
    existing_filenames: set[str] = set()
    metadata_path = output_dir / "metadata.csv"
    if cfg.resume and metadata_path.exists():
        try:
            prior_meta = pd.read_csv(metadata_path, sep="|", names=["filename", "raw_text", "normalized_text"], header=None)
            for r in prior_meta.itertuples(index=False):
                metadata_rows.append(
                    {"filename": str(r.filename), "raw_text": str(r.raw_text), "normalized_text": str(r.normalized_text)}
                )
                existing_filenames.add(str(r.filename))
        except Exception:
            pass

    export_drop_rows: list[dict[str, Any]] = []
    for rec in tqdm(segment_candidates, total=len(segment_candidates), desc="Exporting"):
        row_id = int(rec["row_id"])
        seg_idx = int(rec["segment_index"])
        audio_path = rec["audio_path"]
        text = str(rec["text"]).strip()
        out_id = f"utt_{row_id:06d}_{seg_idx:03d}"
        out_rel = f"wavs/{out_id}.wav"
        out_wav = wavs_dir / f"{out_id}.wav"

        if cfg.resume and (out_rel in existing_filenames or out_wav.exists()):
            if out_rel not in existing_filenames:
                metadata_rows.append({"filename": out_rel, "raw_text": text, "normalized_text": normalize_whitespace(text)})
                existing_filenames.add(out_rel)
            continue

        if not text:
            export_drop_rows.append(
                {"row_id": row_id, "segment_id": out_id, "reason": "empty_text", "duration": rec["duration"], "text": text}
            )
            continue

        pad = cfg.padding_ms / 1000.0
        start = max(0.0, float(rec["start"]) - pad)
        end = min(float(rec["file_duration"]), float(rec["end"]) + pad)
        if end <= start:
            export_drop_rows.append(
                {"row_id": row_id, "segment_id": out_id, "reason": "invalid_time_window", "duration": 0.0, "text": text}
            )
            continue

        temp_wav = temp_dir / f"tmp_{row_id:06d}_{seg_idx:03d}.wav"
        try:
            ffmpeg_extract_to_wav16k_mono(audio_path, start, end, temp_wav, cfg.target_sr)
            y, sr = sf.read(temp_wav)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > 0.98 and peak > 0:
                y = y * (0.98 / peak)
            y = trim_silence_with_cap(y, sr, librosa, max_trim_ms=cfg.silence_trim_ms, db_threshold=-35.0)
            final_dur = float(len(y) / sr) if sr > 0 else 0.0
            if final_dur < cfg.min_dur:
                export_drop_rows.append(
                    {"row_id": row_id, "segment_id": out_id, "reason": "below_min_duration_after_trim", "duration": final_dur, "text": text}
                )
                continue
            if final_dur > cfg.max_dur:
                export_drop_rows.append(
                    {"row_id": row_id, "segment_id": out_id, "reason": "above_max_duration_after_trim", "duration": final_dur, "text": text}
                )
                continue
            sf.write(out_wav, y, sr, subtype="PCM_16")
            metadata_rows.append({"filename": out_rel, "raw_text": text, "normalized_text": normalize_whitespace(text)})
            existing_filenames.add(out_rel)
        except Exception as e:
            incr_failure("export")
            msg = f"{type(e).__name__}: {e}"
            processing_errors.append(
                {
                    "row_id": row_id,
                    "audio_path": audio_path,
                    "stage": "export",
                    "code": "E_EXPORT_SEGMENT",
                    "error": msg,
                    "traceback": traceback.format_exc(),
                }
            )
            export_drop_rows.append(
                {"row_id": row_id, "segment_id": out_id, "reason": f"export_error: {msg}", "duration": rec["duration"], "text": text}
            )
        finally:
            temp_wav.unlink(missing_ok=True)

    metadata_df = pd.DataFrame(metadata_rows)
    if not metadata_df.empty:
        metadata_df = metadata_df.drop_duplicates(subset=["filename"], keep="first").sort_values(by=["filename"])
    metadata_df.to_csv(metadata_path, sep="|", header=False, index=False, quoting=csv.QUOTE_MINIMAL)

    all_drops_df = (
        pd.concat([dropped_segments_df, pd.DataFrame(export_drop_rows)], ignore_index=True)
        if len(dropped_segments_df) or len(export_drop_rows)
        else pd.DataFrame(columns=["row_id", "segment_id", "reason", "duration", "text"])
    )
    all_drops_df.to_csv(output_dir / "dropped_segments.csv", index=False)
    errors_df = pd.DataFrame(processing_errors)
    errors_df.to_csv(output_dir / "processing_errors.csv", index=False)

    post_qc_records = []
    for wav_path in tqdm(sorted(wavs_dir.glob("*.wav")), desc="Post-QC"):
        y, sr = sf.read(wav_path, always_2d=True)
        n_samples, n_channels = y.shape
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        duration = float(n_samples / sr) if sr > 0 else 0.0
        post_qc_records.append(
            {
                "filename": f"wavs/{wav_path.name}",
                "path": str(wav_path),
                "duration_sec": duration,
                "sample_rate": int(sr),
                "channels": int(n_channels),
                "peak_amplitude": peak,
                "is_clipping": bool(peak >= 0.999),
            }
        )
    pd.DataFrame(post_qc_records).to_csv(output_dir / "qc_post_export.csv", index=False)

    zip_path = shutil.make_archive(str(output_dir), "zip", root_dir=str(output_dir))
    runtime_info["asr_cpu_fallback_used"] = asr_cpu_fallback_used
    runtime_info["align_cpu_fallback_used"] = align_cpu_fallback_used
    write_json(logs_dir / "runtime_info.json", runtime_info)

    failures_summary = {
        "timestamp_utc": utc_now_iso(),
        "mode": mode,
        "counts": {
            "alignment_rows_total": int(len(alignment_report_df)),
            "alignment_ok": int((alignment_report_df["status"] == "ok").sum() if "status" in alignment_report_df.columns else 0),
            "alignment_needs_review": int((alignment_report_df["status"] == "needs_review").sum() if "status" in alignment_report_df.columns else 0),
            "alignment_error": int((alignment_report_df["status"] == "error").sum() if "status" in alignment_report_df.columns else 0),
            "segments_dropped": int(len(all_drops_df)),
            "processing_errors": int(len(errors_df)),
            "exported_wavs": int(len(list(wavs_dir.glob("*.wav")))),
        },
        "failures_by_stage": failures_by_stage,
        "runtime": runtime_info,
    }
    write_json(logs_dir / "failures_summary.json", failures_summary)
    print("\nPipeline completed.")
    print(f"- metadata: {metadata_path}")
    print(f"- output zip: {zip_path}")
    print(f"- runtime info: {logs_dir / 'runtime_info.json'}")
    print(f"- failures summary: {logs_dir / 'failures_summary.json'}")
    return failures_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Darija WhisperX dataset builder (Windows hardened).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true", help="Install dependencies from lockfile.")
    mode.add_argument("--preflight", action="store_true", help="Run preflight checks and exit.")
    mode.add_argument("--smoke-test", action="store_true", help="Run small subset pipeline.")
    mode.add_argument("--run-full", action="store_true", help="Run full pipeline.")

    parser.add_argument("--zip-path", default="./cafe-small-clean.zip")
    parser.add_argument("--extract-root", default="./data")
    parser.add_argument("--output-dir", default="./tts_export")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--min-dur", type=float, default=1.5)
    parser.add_argument("--max-dur", type=float, default=15.0)
    parser.add_argument("--preferred-max-dur", type=float, default=12.0)
    parser.add_argument("--pause-gap-threshold", type=float, default=0.6)
    parser.add_argument("--coverage-threshold", type=float, default=0.70)
    parser.add_argument("--silence-trim-ms", type=int, default=200)
    parser.add_argument("--padding-ms", type=int, default=80)
    parser.add_argument("--whisper-model-size", default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--device-preference", choices=["auto", "xpu", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-disk-gb", type=float, default=20.0)
    parser.add_argument("--min-memory-gb", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lockfile", default=str(DEFAULT_LOCKFILE))
    parser.set_defaults(skip_optional_hf=True)
    parser.add_argument("--skip-optional-hf", dest="skip_optional_hf", action="store_true")
    parser.add_argument("--run-optional-hf", dest="skip_optional_hf", action="store_false")
    return parser


def args_to_config(args: argparse.Namespace) -> Config:
    return Config(
        zip_path=args.zip_path,
        extract_root=args.extract_root,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        preferred_max_dur=args.preferred_max_dur,
        pause_gap_threshold=args.pause_gap_threshold,
        coverage_threshold=args.coverage_threshold,
        silence_trim_ms=args.silence_trim_ms,
        padding_ms=args.padding_ms,
        whisper_model_size=args.whisper_model_size,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        device_preference=args.device_preference,
        min_disk_gb=args.min_disk_gb,
        min_memory_gb=args.min_memory_gb,
        resume=args.resume,
        max_files=args.max_files,
        skip_optional_hf=args.skip_optional_hf,
        seed=args.seed,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    lockfile = Path(args.lockfile)

    mode = "run-full"
    if args.install:
        mode = "install"
    elif args.preflight:
        mode = "preflight"
    elif args.smoke_test:
        mode = "smoke-test"
    elif args.run_full:
        mode = "run-full"

    try:
        if mode == "install":
            run_install(lockfile)
            return 0

        ok, preflight_report = run_preflight(cfg, lockfile, strict_lock=True)
        logs_dir = Path(cfg.output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        write_json(logs_dir / "preflight_report.json", preflight_report)

        for c in preflight_report["checks"]:
            state = "OK" if c["ok"] else "FAIL"
            print(f"[{state}] {c['name']}: {c['detail']}")
            if (not c["ok"]) and c.get("hint"):
                print(f"      hint: {c['hint']}")

        if preflight_report.get("lock_mismatches"):
            print("Lockfile mismatches:")
            for m in preflight_report["lock_mismatches"][:20]:
                print(f"  - {m['package']}: expected {m['expected']}, installed {m['installed']}")

        if mode == "preflight":
            return 0 if ok else 1

        ensure_python_311()
        if not ok:
            raise PipelineError("preflight", "E_PREFLIGHT_FAILED", "Preflight checks failed.")

        summary = run_pipeline(cfg, mode=mode, lockfile=lockfile)
        print("Summary counts:", summary.get("counts", {}))
        return 0
    except PipelineError as e:
        print(f"ERROR [{e.stage}/{e.code}]: {e}")
        return 1
    except Exception as e:
        print(f"ERROR [unhandled]: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
