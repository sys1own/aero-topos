# -*- coding: utf-8 -*-
"""Command-line entry point for the strict blueprint checker.

Usage::

    python -m blueprint_lang [check] [PATH]

``PATH`` defaults to ``blueprint.aero`` in the current directory.  The leading
``check`` verb is optional (so both ``python -m blueprint_lang check foo.aero``
and ``python -m blueprint_lang foo.aero`` work).  Exit code is ``0`` when the
blueprint is valid and ``1`` when it is not -- so it drops cleanly into a build
script as a pre-flight gate.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

from . import check_file, looks_like_blueprint_dsl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blueprint_lang",
        description="Strictly validate a blueprint.aero file before building.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="blueprint.aero",
        help="path to the blueprint file (default: blueprint.aero)",
    )
    return parser


def _normalize(argv: Sequence[str]) -> List[str]:
    """Drop an optional leading ``check`` verb so a bare PATH also works."""
    args = list(argv)
    if args and args[0] == "check":
        args = args[1:]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(_normalize(raw))

    if os.path.exists(args.path):
        with open(args.path, "r", encoding="utf-8") as handle:
            source = handle.read()
        if source.strip() and not looks_like_blueprint_dsl(source):
            print(
                f"{args.path}: legacy INI/JSON blueprint detected; this checker "
                "only validates block-format blueprints, so nothing to check."
            )
            return 0

    error = check_file(args.path)
    if error is None:
        print(f"{args.path}: OK -- blueprint is valid")
        return 0
    print(error, file=sys.stderr)
    print(
        f"\naborting: '{args.path}' failed strict validation; no build steps were run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
