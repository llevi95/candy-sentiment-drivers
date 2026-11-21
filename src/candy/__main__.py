"""Command-line entry point: ``python -m candy``."""

from __future__ import annotations

import argparse
from pathlib import Path

from candy.pipeline import SEED, run

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "candy-data.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the results bundle and figures.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw candy CSV")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output folder")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the bootstrap")
    args = parser.parse_args()

    run(raw_csv=args.raw, out_dir=args.out, seed=args.seed)
    print(f"Wrote results bundle and figures to {args.out}")


if __name__ == "__main__":
    main()
