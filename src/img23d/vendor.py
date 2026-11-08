"""Adaptação verificável do import de extração CPU do TripoSR no Windows."""

import argparse
from pathlib import Path

from .config import configure


def prepare_triposr(cache):
    path = Path(cache) / "code" / "TripoSR" / "tsr" / "models" / "isosurface.py"
    old = "from torchmcubes import marching_cubes"
    new = "from img23d.backends.marching_cubes import marching_cubes"
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(
            "TripoSR mudou o import de isosuperfície; revise a integração antes de continuar."
        )
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(
        "TripoSR: import torchmcubes adaptado para scikit-image CPU (sem compilação)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache")
    args = parser.parse_args()
    prepare_triposr(configure(args.cache))
