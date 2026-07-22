"""Compare two run files from disk and exit non-zero on a hard regression.

Separate from `src.run` because CI compares a fresh run against a *committed*
baseline file, with no database in the picture.

    python -m src.compare_cli --baseline baselines/main.json \
        --candidate data/runs/abc123.json --markdown comment.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare.regression import compare_runs, format_report
from .report.html import markdown_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluation run files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--markdown", default=None, help="write a PR-comment summary here")
    parser.add_argument("--history", nargs="*", default=[], help="older runs, for drift")
    args = parser.parse_args(argv)

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    history = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.history]

    result = compare_runs(baseline, candidate, history=history)
    print(format_report(result))

    if args.markdown:
        Path(args.markdown).write_text(
            markdown_summary(candidate, result), encoding="utf-8"
        )

    if result.has_hard_regression:
        print("\nhard regression — failing the check", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
