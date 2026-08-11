#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.export import package_deployment_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a checksum-verified, inference-ready models directory."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--organ", type=Path, required=True)
    parser.add_argument(
        "--lesion",
        type=Path,
        action="append",
        required=True,
        help="Repeat five times in fold0..fold4 order.",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("models/classifier/model_best.pth.tar"),
        help="Existing classifier is copied unchanged.",
    )
    args = parser.parse_args()
    bundle = package_deployment_bundle(
        destination=args.output,
        organ_model=args.organ,
        lesion_models=args.lesion,
        classifier_model=args.classifier,
    )
    print(f"Deployment bundle: {bundle.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
