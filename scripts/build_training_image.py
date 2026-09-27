#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the HIVE training image.")
    parser.add_argument("--tag", default="prostate-mri-training:latest")
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional docker-save destination, e.g. docker/training_image.tar.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(root / "docker" / "Dockerfile.training"),
            "-t",
            args.tag,
            str(root),
        ],
        check=True,
    )
    if args.save is not None:
        destination = args.save.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            with temporary.open("wb") as handle:
                subprocess.run(["docker", "save", args.tag], stdout=handle, check=True)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
