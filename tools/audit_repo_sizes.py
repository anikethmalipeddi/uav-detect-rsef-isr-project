from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


LIMIT_BYTES = 20 * 1024 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> List[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
        text=False,
    )
    files: List[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace")
        files.append(root / rel)
    return files


def main() -> int:
    root = repo_root()
    big: List[Tuple[int, Path]] = []
    for path in tracked_files(root):
        if not path.exists():
            continue
        size = path.stat().st_size
        if size > LIMIT_BYTES:
            big.append((size, path))
    if big:
        print("FAIL: tracked files larger than 20MB were found:")
        for size, path in sorted(big, reverse=True):
            rel = path.relative_to(root).as_posix()
            print(f"  {size / (1024 * 1024):8.2f} MB  {rel}")
        return 1
    print("PASS: no tracked files larger than 20MB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

