"""Command-line entry point: ``python -m candy``."""

from __future__ import annotations

import argparse
from pathlib import Path

from candy.pipeline import SEED, run

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "candy-data.csv"
DEFAULT_GUMMY = REPO_ROOT / "data" / "derived" / "gummy.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the results bundle, figures and appendix."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw candy CSV")
    parser.add_argument(
        "--gummy", type=Path, default=DEFAULT_GUMMY, help="approved gummy coding CSV"
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output folder")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the bootstrap")
    args = parser.parse_args()

    run(raw_csv=args.raw, gummy_csv=args.gummy, out_dir=args.out, seed=args.seed)
    print(f"Wrote results bundle, figures and appendix to {args.out}")


if __name__ == "__main__":
    main()
