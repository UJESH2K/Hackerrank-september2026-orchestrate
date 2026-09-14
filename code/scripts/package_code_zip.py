"""Build the code.zip submission artifact.

    python code/scripts/package_code_zip.py

Zips `code/` into `code.zip` at the repository root, excluding everything the
submission instructions say to leave out: virtualenvs, build artifacts, and
the model-response cache (which also holds no source, only saved API
responses keyed by hash - regenerable, and worthless to a grader). Never
touches `dataset/`; this script only ever reads and writes outside it.

The submission table in problem_statement.md defines code.zip as "Full
runnable solution, prompts/configuration, README, and the required
evaluation/ folder" - so this also bundles the top-level README.md and
docs/APPROACH.md at the zip root alongside code/, since the setup
instructions and the problem-statement mapping live there, not inside code/.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT_PATH = REPO_ROOT / "code.zip"

# Extra top-level files/directories the submission table requires alongside
# code/ (see module docstring). Each entry is (source path, arcname prefix).
EXTRA_INCLUDES = [
    (REPO_ROOT / "README.md", Path("README.md")),
]

# Directory name components that exclude a path (and everything under it).
EXCLUDED_DIR_NAMES = {"__pycache__", ".cache", ".venv", "venv", "node_modules"}
# File suffixes that never belong in a source zip.
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _is_excluded(path: Path, root: Path) -> bool:
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts)


def collect_files() -> list[tuple[Path, Path]]:
    """Returns (source path, archive path) pairs."""
    entries: list[tuple[Path, Path]] = []
    for path in sorted(CODE_DIR.rglob("*")):
        if path.is_file() and not _is_excluded(path, CODE_DIR):
            entries.append((path, Path("code") / path.relative_to(CODE_DIR)))
    for path in sorted(DOCS_DIR.rglob("*")) if DOCS_DIR.exists() else []:
        if path.is_file() and not _is_excluded(path, DOCS_DIR):
            entries.append((path, Path("docs") / path.relative_to(DOCS_DIR)))
    for source, arcname in EXTRA_INCLUDES:
        if source.exists():
            entries.append((source, arcname))
    return entries


def build(output_path: Path = OUTPUT_PATH) -> list[str]:
    entries = collect_files()
    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in entries:
            archive.write(source, arcname=str(arcname))
    return [str(arcname) for _source, arcname in entries]


def main() -> int:
    entries = build()
    print("wrote " + str(OUTPUT_PATH) + " with " + str(len(entries)) + " files")
    for entry in entries:
        print("  " + entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
