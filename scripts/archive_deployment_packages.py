"""Create ready-to-extract upload archives from the clean deployment folders."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]
OUTPUT = ROOT / "deployment"


def archive(folder_name: str, compression: int) -> dict[str, object]:
    folder = OUTPUT / folder_name
    target = OUTPUT / f"{folder_name}.zip"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    with zipfile.ZipFile(target, "w", compression=compression, allowZip64=True) as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not any(part in {"node_modules", "dist", "__pycache__"} for part in path.parts):
                bundle.write(path, path.relative_to(OUTPUT))
    with zipfile.ZipFile(target) as bundle:
        bad = bundle.testzip()
        if bad:
            raise RuntimeError(f"Corrupt archive entry: {bad}")
        count = len(bundle.namelist())
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"file": target.name, "bytes": target.stat().st_size, "files": count, "sha256": digest.hexdigest()}


def main() -> None:
    results = [
        archive("huggingface-space", zipfile.ZIP_STORED),
        archive("vercel-frontend", zipfile.ZIP_DEFLATED),
    ]
    (OUTPUT / "archives.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
