from __future__ import annotations

import argparse

from training.engine import train_lesion_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one inference-compatible lesion fold."
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("-c", "--config", default="configs/lesion.yaml")
    parser.add_argument(
        "--initial-weights",
        help="Optional .pth/.pth.tar checkpoint used only for initialisation.",
    )
    parser.add_argument("--resume", help="Resume full training state from a checkpoint.")
    parser.add_argument("--output-dir", help="Override output.exp_dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = train_lesion_from_config(
        args.config,
        fold=args.fold,
        pretrained_path=args.initial_weights,
        resume=args.resume,
        output_dir=args.output_dir,
    )
    print(f"Run directory:   {artifacts.run_dir}")
    print(f"Drop-in fold:    {artifacts.best_ckpt}")
    print(f"Last checkpoint: {artifacts.last_ckpt}")
    print(f"Metrics:         {artifacts.metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
