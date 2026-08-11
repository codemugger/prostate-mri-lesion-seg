from __future__ import annotations

import argparse

from training.engine import train_organ_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the binary whole-gland model and export model.ts."
    )
    parser.add_argument("-c", "--config", default="configs/organ.yaml")
    parser.add_argument(
        "--initial-weights",
        help="Optional deployed model.ts or checkpoint used only for initialisation.",
    )
    parser.add_argument("--resume", help="Resume full training state from a checkpoint.")
    parser.add_argument("--output-dir", help="Override output.exp_dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = train_organ_from_config(
        args.config,
        pretrained_path=args.initial_weights,
        resume=args.resume,
        output_dir=args.output_dir,
    )
    print(f"Run directory:       {artifacts.run_dir}")
    print(f"Best checkpoint:     {artifacts.best_ckpt}")
    print(f"Last checkpoint:     {artifacts.last_ckpt}")
    print(f"Drop-in organ model: {artifacts.deploy_model}")
    print(f"Metrics:             {artifacts.metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
