"""Command-line entry point for downloading data and running the experiment."""

import argparse
from pathlib import Path

from .data import download


def main():
    parser = argparse.ArgumentParser(description="Forecast public BIDMC pulse waveforms on CPU.")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help="Download and verify seven recordings.")
    fetch.add_argument("--data-dir", type=Path, default=Path("data/bidmc"))
    train = sub.add_parser("train", help="Train a GRU and evaluate held-out recordings.")
    train.add_argument("--data-dir", type=Path, default=Path("data/bidmc"))
    train.add_argument("--output-dir", type=Path, default=Path("runs/demo"))
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "download":
        download(args.data_dir)
        print("All seven recordings passed SHA-256 verification.")
    else:
        if args.epochs < 1 or args.threads < 1:
            parser.error("--epochs and --threads must be positive")
        from .train import run

        run(args.data_dir, args.output_dir, args.epochs, args.seed, args.threads)


if __name__ == "__main__":
    main()
