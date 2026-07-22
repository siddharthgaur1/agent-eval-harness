"""Self-contained HTML report.

No template engine and no CDN: a report you can email to someone, open on a
plane, or attach to a CI artifact is worth more than a prettier one that needs a
server. Everything is inlined.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..aggregate import RunAggregate
from ..compare.regression import Comparison
from ..config import settings

_CSS = """
:root { --good:#1a7f37; --warn:#9a6700; --bad:#cf222e; --line:#d0d7de; --muted:#57606a; }
* { box-sizing: border-box; }
body { font: 15px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif;
       margin: 0; padding: 2rem; color: #1f2328; background: #fff; }
h1 { margin: 0 0 .25rem; font-size: 1.5rem; }
h2 { margin: 2rem 0 .75rem; font-size: 1.1rem; border-bottom: 1px solid var(--line);
     padding-bottom: .35rem; }
.sub { color: var(--muted); margin-bottom: 1.5rem; font-size: .9rem; }
.cards { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: .9rem 1.1rem; min-width: 150px; }
.card .v { font-size: 1.7rem; font-weight: 600; }
.card .l { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }
th { font-weight: 600; color: var(--muted); font-size: .78rem; text-transform: uppercase; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { background: #eaeef2; border-radius: 3px; height: 9px; width: 160px; overflow: hidden; }
.bar > i { display: block; height: 100%; }
.good > i, .fill-good { background: var(--good); }
.warn > i, .fill-warn { background: var(--warn); }
.bad  > i, .fill-bad  { background: var(--bad); }
.tag { font-size: .72rem; padding: .1rem .45rem; border-radius: 10px; border: 1px solid var(--line); }
.tag.pass { color: var(--good); border-color: var(--good); }
.tag.fail { color: var(--bad); border-color: var(--bad); }
.tag.noise { color: var(--muted); }
details { margin: .3rem 0; }
summary { cursor: pointer; font-size: .88rem; }
pre { background: #f6f8fa; padding: .7rem; border-radius: 6px; overflow-x: auto;
      font-size: .8rem; white-space: pre-wrap; }
.ev { color: var(--muted); font-size: .8rem; }
.banner { padding: .8rem 1rem; border-radius: 8px; margin: 1rem 0; font-weight: 600; }
.banner.bad { background: #ffebe9; color: var(--bad); }
.banner.good { background: #dafbe1; color: var(--good); }
"""


def _cls(score: float) -> str:
    return "good" if score >= 0.8 else "warn" if score >= 0.6 else "bad"


def _bar(score: float) -> str:
    return (
        f'<div class="bar {_cls(score)}"><i style="width:{max(score, 0) * 100:.0f}%"></i></div>'
    )


def _e(text: Any) -> str:
    return html.escape(str(text))


def render_report(record: dict[str, Any], comparison: Comparison | None = None) -> str:
    """Build the full HTML document for one run."""
    agg = RunAggregate.model_validate(record["aggregate"])
    passing = sum(1 for t in agg.tasks.values() if t.passed)

    parts = [
        f"<style>{_CSS}</style>",
        f"<h1>Agent evaluation — {_e(record['agent_version'])}</h1>",
        f"<div class='sub'>suite <b>{_e(record['suite'])}</b> · run <code>{_e(record['run_id'])}</code>"
        f" · {_e(record.get('repeats', 1))} repeat(s) per task · {_e(record.get('created_at', ''))}</div>",
    ]

    if comparison is not None:
        parts.append(
            f"<div class='banner {'bad' if comparison.has_hard_regression else 'good'}'>"
            + (
                "HARD REGRESSION vs baseline "
                + _e(comparison.baseline_run_id)
                if comparison.has_hard_regression
                else "No hard regression vs baseline " + _e(comparison.baseline_run_id)
            )
            + "</div>"
        )

    parts.append(
        "<div class='cards'>"
        f"<div class='card'><div class='v'>{agg.overall.mean:.3f}</div>"
        f"<div class='l'>overall (±{agg.overall.stdev:.3f})</div></div>"
        f"<div class='card'><div class='v'>{passing}/{len(agg.tasks)}</div>"
        "<div class='l'>tasks passing</div></div>"
        f"<div class='card'><div class='v'>{len(agg.unstable_dimensions)}</div>"
        "<div class='l'>unstable dimensions</div></div>"
        "</div>"
    )

    parts.append(_dimension_table(agg))
    if comparison is not None:
        parts.append(_comparison_section(comparison))
    parts.append(_task_table(agg, record, comparison))
    parts.append(_trajectory_drilldown(record))

    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>Agent eval — {_e(record['agent_version'])}</title>"
        + "".join(parts)
    )


def _dimension_table(agg: RunAggregate) -> str:
    rows = []
    for name, stat in sorted(agg.dimensions.items()):
        flag = "<span class='tag noise'>unstable</span>" if stat.unstable else ""
        rows.append(
            f"<tr><td>{_e(name)}</td><td class='num'>{stat.mean:.3f}</td>"
            f"<td class='num'>±{stat.stdev:.3f}</td><td>{_bar(stat.mean)}</td><td>{flag}</td></tr>"
        )
    return (
        "<h2>Dimensions</h2><table><tr><th>dimension</th><th>mean</th><th>spread</th>"
        "<th></th><th></th></tr>" + "".join(rows) + "</table>"
    )


def _comparison_section(cmp: Comparison) -> str:
    rows = []
    for d in cmp.dimensions:
        tag = {
            "regression": "<span class='tag fail'>regression</span>",
            "improvement": "<span class='tag pass'>improved</span>",
            "within_noise": "<span class='tag noise'>within noise</span>",
        }.get(d.verdict, "")
        rows.append(
            f"<tr><td>{_e(d.dimension)}</td><td class='num'>{d.baseline:.3f}</td>"
            f"<td class='num'>{d.candidate:.3f}</td><td class='num'>{d.delta:+.3f}</td>"
            f"<td class='num'>±{d.noise_band:.3f}</td><td>{tag}</td></tr>"
        )
    drift = ""
    if cmp.drift:
        drift = "<h2>Slow drift</h2><ul>" + "".join(
            f"<li>{_e(d.dimension)}: {d.first:.3f} → {d.last:.3f} "
            f"({d.total_drift:+.3f} over {d.n_runs} runs)</li>"
            for d in cmp.drift
        ) + "</ul>"
    return (
        "<h2>vs baseline</h2><table><tr><th>dimension</th><th>baseline</th><th>candidate</th>"
        "<th>delta</th><th>noise band</th><th></th></tr>"
        + "".join(rows)
        + "</table>"
        + drift
    )


def _task_table(agg: RunAggregate, record: dict, cmp: Comparison | None) -> str:
    deltas = {t.task_id: t for t in (cmp.tasks if cmp else [])}
    rows = []
    for task_id, summary in sorted(agg.tasks.items()):
        d = deltas.get(task_id)
        status = (
            "<span class='tag fail'>regressed</span>"
            if d and d.newly_failing
            else ("<span class='tag pass'>pass</span>" if summary.passed else "<span class='tag fail'>fail</span>")
        )
        delta_cell = f"{d.delta:+.3f}" if d else "—"
        rows.append(
            f"<tr><td>{_e(task_id)}</td><td>{_e(summary.category)}</td>"
            f"<td class='num'>{summary.overall.mean:.3f}</td>"
            f"<td class='num'>±{summary.overall.stdev:.3f}</td>"
            f"<td class='num'>{delta_cell}</td><td>{_bar(summary.overall.mean)}</td>"
            f"<td>{status}</td></tr>"
        )
    return (
        "<h2>Tasks</h2><table><tr><th>task</th><th>category</th><th>score</th><th>spread</th>"
        "<th>Δ</th><th></th><th></th></tr>" + "".join(rows) + "</table>"
    )


def _trajectory_drilldown(record: dict) -> str:
    """Per-execution scorer output, including the cited step indices."""
    blocks = []
    for tr in record.get("task_runs", []):
        scores = tr.get("scores", {})
        overall = (
            sum(s["score"] for s in scores.values()) / len(scores) if scores else 0.0
        )
        rows = "".join(
            f"<tr><td>{_e(name)}</td><td class='num'>{s['score']:.3f}</td>"
            f"<td>{_e(s['reasoning'])}</td>"
            f"<td class='ev'>{_e(s.get('evidence') or '—')}</td></tr>"
            for name, s in sorted(scores.items())
        )
        blocks.append(
            f"<details><summary><b>{_e(tr['task_id'])}</b> · repeat {tr['repeat']} · "
            f"score {overall:.3f} · trajectory <code>{_e(tr.get('trajectory_id', ''))}</code></summary>"
            "<table><tr><th>scorer</th><th>score</th><th>reasoning</th><th>evidence steps</th></tr>"
            + rows
            + "</table></details>"
        )
    return "<h2>Drill-down</h2>" + "".join(blocks)


def write_report(
    record: dict[str, Any], comparison: Comparison | None = None, out_path: Path | None = None
) -> Path:
    """Render and write the report, returning the path."""
    settings.ensure_dirs()
    path = out_path or settings.reports_dir / f"{record['run_id']}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(record, comparison), encoding="utf-8")
    return path


def write_json(record: dict[str, Any], out_path: Path | None = None) -> Path:
    """Machine-readable twin of the HTML report."""
    settings.ensure_dirs()
    path = out_path or settings.reports_dir / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def markdown_summary(record: dict[str, Any], comparison: Comparison | None = None) -> str:
    """Compact summary for a PR comment."""
    agg = RunAggregate.model_validate(record["aggregate"])
    passing = sum(1 for t in agg.tasks.values() if t.passed)
    lines = [
        f"### Agent eval — `{record['agent_version']}` on `{record['suite']}`",
        "",
        f"**Overall {agg.overall.mean:.3f}** (±{agg.overall.stdev:.3f}) · "
        f"{passing}/{len(agg.tasks)} tasks passing · run `{record['run_id']}`",
        "",
        "| dimension | score | spread |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| {n} | {s.mean:.3f} | ±{s.stdev:.3f}{' ⚠︎ unstable' if s.unstable else ''} |"
        for n, s in sorted(agg.dimensions.items())
    ]

    if comparison:
        lines += ["", "#### vs baseline `" + comparison.baseline_run_id + "`", ""]
        if comparison.has_hard_regression:
            lines.append("🔴 **Hard regression.**")
            for d in comparison.regressed_dimensions:
                lines.append(
                    f"- `{d.dimension}` {d.baseline:.3f} → {d.candidate:.3f} "
                    f"({d.delta:+.3f}, noise band ±{d.noise_band:.3f})"
                )
            for t in comparison.newly_failing_tasks:
                lines.append(f"- task `{t.task_id}` was passing, now failing ({t.delta:+.3f})")
        else:
            lines.append("🟢 No hard regression — all deltas within threshold or noise.")
        if comparison.drift:
            lines += ["", "⚠︎ Slow drift:"] + [
                f"- `{d.dimension}` {d.total_drift:+.3f} over {d.n_runs} runs"
                for d in comparison.drift
            ]

    lines += ["", f"<sub>generated {datetime.now().isoformat(timespec='seconds')}</sub>"]
    return "\n".join(lines)
