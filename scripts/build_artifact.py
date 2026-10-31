#!/usr/bin/env python3
"""Build the archival research bundle for the LazyQC paper.

The bundle holds the library source, benchmark runners and frozen manifests,
raw results with their summaries, the paper sources, ``uv.lock`` and a
``SHA256SUMS`` file. ``ARTIFACT.json`` records the git commit, whether the
working tree was clean, the environment and every manifest. An archive built
from a dirty tree is marked as such; deposit only a bundle built from a clean,
tagged commit (for example through the Zenodo GitHub integration).

    uv run python scripts/build_artifact.py
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "src",
    "tests",
    "benchmarks",
    "results",
    "paper",
    "docs",
    "scripts",
    "cloud",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "Taskfile.yml",
)
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "node_modules"}
EXCLUDE_SUFFIXES = (".pyc", ".lock.tmp", ".py.lock")


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def files():
    for entry in INCLUDE:
        path = ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            for item in sorted(path.rglob("*")):
                if (
                    item.is_file()
                    and not EXCLUDE_PARTS.intersection(item.relative_to(ROOT).parts)
                    and not item.name.endswith(EXCLUDE_SUFFIXES)
                ):
                    yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--allow-dirty", action="store_true", help="build from a dirty tree")
    args = parser.parse_args()
    status = git("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit(
            "working tree is not clean; commit first or pass --allow-dirty for a draft"
        )
    commit = git("rev-parse", "HEAD")
    selected = list(files())
    sums = []
    for path in selected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {path.relative_to(ROOT)}")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_describe": git("describe", "--always", "--dirty", "--tags"),
        "clean_tree": not status,
        "python": sys.version,
        "platform": platform.platform(),
        "lazyqc_version": importlib.metadata.version("lazyqc"),
        "file_count": len(selected),
        "manifests": sorted(
            str(p.relative_to(ROOT))
            for p in selected
            if p.parent.name == "benchmarks" and p.name.endswith("manifest.json")
        ),
        "note": "Raw repetitions sit next to each summary in results/. "
        "SHA256SUMS covers every file.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = "" if not status else "-dirty"
    name = f"lazyqc-artifact-{stamp}-{commit[:8]}{suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"{name}.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for path in selected:
            archive.add(path, arcname=f"{name}/{path.relative_to(ROOT)}")
        for label, text in (
            ("SHA256SUMS", "\n".join(sums) + "\n"),
            ("ARTIFACT.json", json.dumps(metadata, indent=2) + "\n"),
        ):
            data = text.encode()
            info = tarfile.TarInfo(f"{name}/{label}")
            info.size = len(data)
            info.mtime = int(datetime.now(timezone.utc).timestamp())
            archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target} files={len(selected)} sha256={digest} clean={not status}")


if __name__ == "__main__":
    main()
