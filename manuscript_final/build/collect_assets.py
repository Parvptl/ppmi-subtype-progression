#!/usr/bin/env python3
"""Copy the figures the manuscript uses into manuscript_final/figures/.

Reads build/asset_manifest.json, copies each source file (unchanged) to
figures/<file>, and writes figures/FIGURE_INDEX.csv with the source path,
the SHA-256 of the copy and the caption. Nothing outside manuscript_final/
is written or modified.
"""
from __future__ import annotations
import csv, hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MF = HERE.parent
ROOT = MF.parent
FIG = MF / "figures"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    man = json.loads((HERE / "asset_manifest.json").read_text())
    FIG.mkdir(exist_ok=True)
    rows, missing = [], []
    for f in man["figures"]:
        src = ROOT / f["source"]
        dst = FIG / f["file"]
        if not src.exists():
            missing.append(f["source"])
            continue
        shutil.copyfile(src, dst)
        rows.append({"figure": f["id"], "file": f["file"], "source": f["source"],
                     "sha256": sha256(dst), "caption": f["caption"]})
    with (FIG / "FIGURE_INDEX.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["figure", "file", "source", "sha256", "caption"])
        w.writeheader(); w.writerows(rows)
    print(f"copied {len(rows)} figures to {FIG.relative_to(ROOT)}")
    if missing:
        print("MISSING:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
