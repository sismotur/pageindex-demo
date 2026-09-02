#!/usr/bin/env python3
"""
run_store.py — Historized run storage for eval and conversation artifacts.

Layout (all under the gitignored results/ directory):

    results/{dest}/{lang}/
    ├── runs/{YYYY-MM-DDTHH-MM-SSZ}_{model_tag}/
    │   ├── manifest.json            # run metadata (code/data/model versions)
    │   ├── eval.json                # raw per-question results (run_eval.py)
    │   ├── scored.json              # rubric output (score_results.py)
    │   ├── conversations.json       # scripted threads (chat_demo.py)
    │   └── conversations_scored.json
    ├── latest -> runs/{...}         # symlink (or latest.json pointer file)
    ├── baseline.json                # {"run": "<run_id>"} regression reference
    └── history.jsonl                # one aggregate row per scored run

A run directory is created once and never overwritten, so any two runs can
be compared later. The manifest records the code version (git commit + dirty
flag), the data version (index generated_at + sha256), and the model, so a
score change can be attributed to code, data, or model.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"


# ── Naming ─────────────────────────────────────────────────────────────────

def model_tag(model: str) -> str:
    """'openai/gemma-4-E2B-it-MLX-8bit' -> 'gemma-4-E2B-it-MLX-8bit'."""
    return model.split("/")[-1].replace(":", "-")


def make_run_id(model: str, started_at: datetime | None = None) -> str:
    ts = started_at or datetime.now(timezone.utc)
    return f"{ts.strftime('%Y-%m-%dT%H-%M-%SZ')}_{model_tag(model)}"


def new_run_dir(destination: str, lang: str, model: str,
                started_at: datetime | None = None) -> Path:
    """Create and return a fresh, never-before-used run directory."""
    base = RESULTS_DIR / destination / lang / "runs"
    run_id = make_run_id(model, started_at)
    candidate = base / run_id
    n = 2
    while candidate.exists():
        candidate = base / f"{run_id}-{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


# ── Manifest ───────────────────────────────────────────────────────────────

def git_state() -> dict:
    """Current commit + dirty flag; None values when git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def index_fingerprint(index_path: Path, index: dict | None = None) -> dict:
    """Content hash + build metadata of the index file used for the run."""
    fp = {
        "path": _relpath(index_path),
        "sha256_12": hashlib.sha256(
            index_path.read_bytes()).hexdigest()[:12],
    }
    meta = (index or {}).get("meta") or {}
    for key in ("generated_at", "poi_count", "schema_version"):
        if meta.get(key) is not None:
            fp[key] = meta[key]
    return fp


def build_manifest(run_dir: Path, *, kind: str, model: str, lang: str,
                   destination: str, index_path: Path,
                   index: dict | None = None,
                   questions_file: Path | None = None,
                   conversations_file: Path | None = None,
                   started_at: datetime | None = None) -> dict:
    ts = started_at or datetime.now(timezone.utc)
    manifest = {
        "run_id":       run_dir.name,
        "kind":         kind,          # "eval" | "conversations"
        "started_at":   ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model":        model,
        "model_tag":    model_tag(model),
        "lang":         lang,
        "destination":  destination,
        "git":          git_state(),
        "index":        index_fingerprint(index_path, index),
    }
    if questions_file is not None:
        manifest["questions_file"] = _relpath(questions_file)
    if conversations_file is not None:
        manifest["conversations_file"] = _relpath(conversations_file)
    write_manifest(run_dir, manifest)
    return manifest


def write_manifest(run_dir: Path, manifest: dict) -> None:
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def read_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── latest pointer ─────────────────────────────────────────────────────────

def update_latest(run_dir: Path) -> None:
    """Point {dest}/{lang}/latest at this run dir (symlink, JSON fallback)."""
    lang_dir = run_dir.parent.parent
    link = lang_dir / "latest"
    target = Path("runs") / run_dir.name
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    except OSError:
        # Filesystem without symlink support: JSON pointer file instead.
        with open(lang_dir / "latest.json", "w", encoding="utf-8") as f:
            json.dump({"run": run_dir.name}, f)


def find_latest_run(destination: str, lang: str,
                    require: str = "eval.json") -> Path | None:
    """Newest run dir for (destination, lang) containing `require`d file."""
    lang_dir = RESULTS_DIR / destination / lang
    link = lang_dir / "latest"
    if link.is_symlink():
        candidate = lang_dir / os.readlink(link)
        if (candidate / require).exists():
            return candidate
    pointer = lang_dir / "latest.json"
    if pointer.exists():
        with open(pointer, encoding="utf-8") as f:
            candidate = lang_dir / "runs" / json.load(f).get("run", "")
        if (candidate / require).exists():
            return candidate
    runs_dir = lang_dir / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir() if (d / require).exists()),
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


# ── Baseline ───────────────────────────────────────────────────────────────

def read_baseline(destination: str, lang: str) -> Path | None:
    path = RESULTS_DIR / destination / lang / "baseline.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        run_id = json.load(f).get("run", "")
    run_dir = RESULTS_DIR / destination / lang / "runs" / run_id
    return run_dir if run_dir.is_dir() else None


def set_baseline(destination: str, lang: str, run_dir: Path) -> Path:
    path = RESULTS_DIR / destination / lang / "baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"run": run_dir.name}, f, indent=2)
    return path


# ── History ────────────────────────────────────────────────────────────────

def history_path(destination: str, lang: str) -> Path:
    return RESULTS_DIR / destination / lang / "history.jsonl"


def append_history(destination: str, lang: str, row: dict) -> bool:
    """Append one aggregate row; skip when run_id is already recorded."""
    path = history_path(destination, lang)
    run_id = row.get("run_id")
    if path.exists() and run_id:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and json.loads(line).get("run_id") == run_id \
                        and json.loads(line).get("kind") == row.get("kind"):
                    return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def load_history(destination: str, lang: str) -> list[dict]:
    path = history_path(destination, lang)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_lang_dirs() -> list[tuple[str, str, Path]]:
    """All (destination, lang, dir) pairs that hold at least one run."""
    pairs = []
    if not RESULTS_DIR.is_dir():
        return pairs
    for dest_dir in sorted(RESULTS_DIR.iterdir()):
        if not dest_dir.is_dir():
            continue
        for lang_dir in sorted(dest_dir.iterdir()):
            if (lang_dir / "runs").is_dir():
                pairs.append((dest_dir.name, lang_dir.name, lang_dir))
    return pairs


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
