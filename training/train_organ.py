from __future__ import annotations

import argparse
from training.engine import train_organ_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train organ (prostate) segmentation model with MONAI.")
    parser.add_argument(
        "-c", "--config", type=str, default="configs/organ.yaml", help="Path to organ training YAML config."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = train_organ_from_config(args.config)
    print("Artifacts:")
    print(f" - best checkpoint: {artifacts.best_ckpt}")
    print(f" - last checkpoint: {artifacts.last_ckpt}")
    print(f" - metrics csv:     {artifacts.metrics_csv}")


if __name__ == "__main__":
    main()

