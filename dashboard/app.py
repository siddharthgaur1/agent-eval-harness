"""Trajectory viewer.

The point of the whole harness is being able to *see* where an agent went wrong.
A scorecard tells you `error_recovery` dropped to 0.2; this tells you it was step
17, the tool was `train_model`, and it was called with the same arguments four
times after the same error.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregate import RunAggregate  # noqa: E402
from src.compare.regression import compare_runs  # noqa: E402
from src.persistence.store import Store  # noqa: E402
from src.trace.schema import StepType, Trajectory  # noqa: E402

st.set_page_config(page_title="Agent Eval Harness", layout="wide")

STEP_ICONS = {
    StepType.PLAN: "🧭",
    StepType.TOOL_CALL: "🔧",
    StepType.TOOL_RESULT: "📤",
    StepType.LLM_MESSAGE: "💬",
    StepType.RETRY: "🔁",
    StepType.ESCALATION: "🙋",
}


def _mean_score(task_run: dict) -> float:
    """Unweighted mean across dimensions for one execution."""
    scores = task_run.get("scores") or {}
    return sum(s["score"] for s in scores.values()) / len(scores) if scores else 0.0


@st.cache_resource
def get_store() -> Store:
    return Store()


store = get_store()

st.sidebar.title("Agent Eval Harness")
view = st.sidebar.radio("View", ["Run scorecard", "Trajectory viewer", "Side-by-side diff"])


def _run_options() -> dict[str, dict]:
    runs = store.list_runs(limit=100)
    return {
        f"{r['agent_version']} · {r['run_id']} · {r['created_at'][:19]}": r for r in runs
    }


# ---------------------------------------------------------------------------
# Run scorecard
# ---------------------------------------------------------------------------
if view == "Run scorecard":
    st.title("Run scorecard")
    options = _run_options()
    if not options:
        st.info("No runs yet. Try: `python -m src.run --suite suites/default.yaml --no-llm`")
        st.stop()

    label = st.selectbox("Run", list(options))
    record = store.get_run(options[label]["run_id"])
    agg = RunAggregate.model_validate(record["aggregate"])

    passing = sum(1 for t in agg.tasks.values() if t.passed)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall", f"{agg.overall.mean:.3f}", f"±{agg.overall.stdev:.3f} spread")
    c2.metric("Tasks passing", f"{passing}/{len(agg.tasks)}")
    c3.metric("Unstable dimensions", len(agg.unstable_dimensions))

    st.subheader("Dimensions")
    st.bar_chart({n: s.mean for n, s in sorted(agg.dimensions.items())})
    st.dataframe(
        [
            {
                "dimension": n,
                "mean": s.mean,
                "spread": s.stdev,
                "min": s.min,
                "max": s.max,
                "unstable": s.unstable,
            }
            for n, s in sorted(agg.dimensions.items())
        ],
        use_container_width=True,
    )

    st.subheader("Tasks")
    st.dataframe(
        [
            {
                "task": t.task_id,
                "category": t.category,
                "score": t.overall.mean,
                "spread": t.overall.stdev,
                "passed": t.passed,
                "unstable": ", ".join(t.unstable_dimensions),
            }
            for t in sorted(agg.tasks.values(), key=lambda x: x.overall.mean)
        ],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Trajectory viewer
# ---------------------------------------------------------------------------
elif view == "Trajectory viewer":
    st.title("Trajectory viewer")

    options = _run_options()
    if not options:
        st.info("No runs recorded yet.")
        st.stop()

    label = st.sidebar.selectbox("Run", list(options))
    record = store.get_run(options[label]["run_id"])
    task_runs = record["task_runs"]

    picked = st.sidebar.selectbox(
        "Task execution",
        range(len(task_runs)),
        format_func=lambda i: (
            f"{task_runs[i]['task_id']} #{task_runs[i]['repeat']} · "
            f"{_mean_score(task_runs[i]):.2f}"
        ),
    )
    task_run = task_runs[picked]
    traj: Trajectory | None = store.get_trajectory(task_run["trajectory_id"])

    if traj is None:
        st.error("Trajectory not found (the task may have crashed before recording).")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Terminal state", traj.terminal_state.value)
    c2.metric("Steps", len(traj.steps))
    c3.metric("Tokens", f"{traj.total_tokens:,}")
    c4.metric("Wall clock", f"{traj.wall_clock_seconds:.1f}s")

    # Every step a scorer cited, so the timeline can point at exactly the evidence.
    cited: dict[int, list[str]] = {}
    for name, result in task_run["scores"].items():
        for idx in result.get("evidence", []):
            cited.setdefault(idx, []).append(f"{name} ({result['score']:.2f})")

    st.subheader("Scores")
    for name, result in sorted(task_run["scores"].items()):
        with st.expander(f"{name} — {result['score']:.3f}", expanded=result["score"] < 0.7):
            st.write(result["reasoning"])
            if result.get("evidence"):
                st.caption("Evidence: steps " + ", ".join(map(str, result["evidence"])))
            if result.get("details"):
                st.json(result["details"], expanded=False)

    st.subheader("Timeline")
    show_failures_only = st.checkbox("Only failures, retries, and cited steps")

    for step in traj.steps:
        is_interesting = step.failed or step.step_type is StepType.RETRY or step.index in cited
        if show_failures_only and not is_interesting:
            continue

        icon = STEP_ICONS.get(step.step_type, "•")
        header = f"{icon} `{step.index}` **{step.step_type.value}**"
        if step.agent_name:
            header += f" · {step.agent_name}"
        if step.tool_name:
            header += f" · `{step.tool_name}`"
        if step.failed:
            header += " · ❌ **FAILED**"
        if step.index in cited:
            header += " · 🔍 cited by " + ", ".join(cited[step.index])

        with st.container(border=True):
            st.markdown(header)
            cols = st.columns([1, 1, 1])
            cols[0].caption(f"{step.tokens} tokens")
            cols[1].caption(f"{step.latency_ms} ms")
            cols[2].caption(step.timestamp.strftime("%H:%M:%S"))
            if step.tool_input:
                st.code(str(step.tool_input)[:1000], language="json")
            if step.error:
                st.error(step.error)
            elif step.tool_output is not None:
                st.text(str(step.tool_output)[:1500])

    st.subheader("Final output")
    st.write(traj.final_output or "_(empty)_")

    calls = store.judge_calls(traj.run_id)
    if calls:
        st.subheader("Judge audit trail")
        for call in calls:
            with st.expander(f"{call['scorer']} · {call['model']} · {call['created_at'][:19]}"):
                st.text_area("prompt", call["prompt"], height=200, key=f"p{call['created_at']}{call['scorer']}")
                st.code(call["response"], language="json")


# ---------------------------------------------------------------------------
# Side-by-side diff
# ---------------------------------------------------------------------------
else:
    st.title("Side-by-side diff")
    options = _run_options()
    if len(options) < 2:
        st.info("Need at least two runs to diff.")
        st.stop()

    labels = list(options)
    left = st.sidebar.selectbox("Baseline", labels, index=min(1, len(labels) - 1))
    right = st.sidebar.selectbox("Candidate", labels, index=0)

    base, cand = store.get_run(options[left]["run_id"]), store.get_run(options[right]["run_id"])
    result = compare_runs(base, cand, history=store.list_runs(base["agent_version"]))

    if result.has_hard_regression:
        st.error("HARD REGRESSION")
    else:
        st.success("No hard regression — all deltas within threshold or noise.")

    st.subheader("Dimensions")
    st.dataframe(
        [
            {
                "dimension": d.dimension,
                "baseline": d.baseline,
                "candidate": d.candidate,
                "delta": d.delta,
                "noise band": d.noise_band,
                "verdict": d.verdict,
            }
            for d in result.dimensions
        ],
        use_container_width=True,
    )

    st.subheader("Tasks")
    st.dataframe(
        [
            {
                "task": t.task_id,
                "baseline": t.baseline,
                "candidate": t.candidate,
                "delta": t.delta,
                "newly failing": t.newly_failing,
            }
            for t in sorted(result.tasks, key=lambda x: x.delta)
        ],
        use_container_width=True,
    )

    if result.drift:
        st.subheader("Slow drift")
        st.dataframe([d.model_dump() for d in result.drift], use_container_width=True)

    st.subheader("Same task, both versions")
    task_ids = sorted({t.task_id for t in result.tasks})
    task_id = st.selectbox("Task", task_ids)
    lcol, rcol = st.columns(2)
    for col, record, title in ((lcol, base, "baseline"), (rcol, cand, "candidate")):
        with col:
            st.caption(f"{title} · {record['agent_version']}")
            runs = [tr for tr in record["task_runs"] if tr["task_id"] == task_id]
            for tr in runs:
                st.markdown(f"**repeat {tr['repeat']}** — {_mean_score(tr):.3f}")
                for name, res in sorted(tr["scores"].items()):
                    st.write(f"`{name}` {res['score']:.2f} — {res['reasoning'][:200]}")
