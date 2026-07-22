"""Generate the adversarial fixtures the suite's non-happy-path tasks need.

Derived from customer_churn.csv rather than synthesised from scratch, so each
adversarial file differs from the happy-path one in exactly one way. That is what
makes a score difference attributable: if `single_class_target` regresses and
`churn_baseline` does not, the single-class label is the cause, because nothing
else about the two files differs.

    python sample_data/generate.py
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "customer_churn.csv"


def _read() -> tuple[list[str], list[list[str]]]:
    with SOURCE.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _write(name: str, header: list[str], rows: list[list[str]]) -> None:
    path = HERE / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows, {len(header)} columns")


def main() -> None:
    header, rows = _read()
    churn_idx = header.index("churned")

    # empty.csv — valid headers, no rows. The agent should notice before it
    # tries to split a zero-row frame, not after.
    _write("empty.csv", header, [])

    # single_class.csv — every label identical. Nothing to classify; accuracy
    # 1.0 is trivially achievable and completely meaningless.
    single = [list(r) for r in rows]
    for row in single:
        row[churn_idx] = "0"
    _write("single_class.csv", header, single)

    # leaky_churn.csv — a column that trivially encodes the target. A model
    # trained on it scores near-perfectly and is worthless in production.
    leaky_header = header + ["account_closed_date"]
    leaky = [
        r + ["2026-03-14" if r[churn_idx] == "1" else ""] for r in map(list, rows)
    ]
    _write("leaky_churn.csv", leaky_header, leaky)

    # corrupt_rows.csv — every 20th row has the wrong field count and junk in a
    # numeric column. Parsing it strictly raises; the correct behaviour is to
    # drop or repair the bad rows and carry on, not to retry the same read.
    corrupt = []
    for i, row in enumerate(map(list, rows)):
        if i % 20 == 0:
            row[header.index("monthly_charges")] = "N/A"
            row = row[:-2]  # short row: wrong field count
        corrupt.append(row)
    _write("corrupt_rows.csv", header, corrupt)


if __name__ == "__main__":
    main()
