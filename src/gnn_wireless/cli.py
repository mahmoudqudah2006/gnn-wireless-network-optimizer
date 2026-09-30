from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN wireless scheduling baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--graphs", type=int, default=200)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--output", type=Path, default=Path("results/training.json"))
    args = parser.parse_args()

    _, metrics = train_model(args.graphs, args.epochs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
