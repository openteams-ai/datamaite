"""CLI entrypoint for databridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="databridge",
        description="Dataset validation, loading, and conversion.",
    )
    sub = parser.add_subparsers(dest="command")

    val_parser = sub.add_parser("validate", help="Validate a dataset.")
    val_parser.add_argument("path", type=Path, help="Path to the dataset root.")
    val_parser.add_argument("--format", default="hmie", help="Dataset format (default: hmie).")
    val_parser.add_argument("--skip-video-check", action="store_true", help="Skip FMV integrity checks.")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "validate":
        from databridge.validation import validate

        result = validate(args.path, format=args.format, check_video_integrity=not args.skip_video_check)
        print(result.summary())
        return 0 if result.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
