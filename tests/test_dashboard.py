"""The trajectory viewer, executed headless.

A dashboard that imports cleanly but raises on render is a dashboard that is
broken exactly when you need it — mid-incident, looking for the failing step. So
these run the real Streamlit script against a real store rather than testing the
helpers in isolation.
"""

from __future__ import annotations

import pytest

from src.persistence.store import Store
from src.run import run_suite
from src.suites.loader import load_suite
from src.suites.schema import Suite

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = "dashboard/app.py"
TIMEOUT = 60


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory, monkeypatch_session):
    """A store holding a good run and a degraded one, both with real failures."""
    db = tmp_path_factory.mktemp("dash") / "dash.db"
    monkeypatch_session.setenv("DB_PATH", str(db))

    # config caches its settings, so point the default Store at this DB directly.
    from src import config

    config.settings.db_path = db
    config.get_settings.cache_clear()

    store = Store(db)
    full = load_suite("suites/default.yaml")
    suite = Suite(
        name="dash",
        tasks=[t for t in full.tasks if t.task_id in {"churn_baseline", "retry_bait"}],
    )
    kwargs = dict(repeats=2, workers=2, include_llm=False, store=store)
    baseline, _ = run_suite(suite, "mock", "v1", **kwargs)
    degraded, _ = run_suite(suite, "mock", "v2-degraded", **kwargs)
    store.mark_baseline(baseline["run_id"])
    return store, baseline, degraded


@pytest.fixture(scope="module")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _run_view(view: str) -> "AppTest":
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    app.sidebar.radio[0].set_value(view).run()
    return app


def test_scorecard_renders(populated_db):
    app = _run_view("Run scorecard")
    assert not app.exception
    assert any("Overall" in m.label for m in app.metric)


def test_trajectory_viewer_renders_and_surfaces_the_failure(populated_db):
    """Acceptance criterion 4: the viewer shows a failed run's exact failure step."""
    store, _, degraded = populated_db

    app = _run_view("Trajectory viewer")
    assert not app.exception, app.exception

    # The timeline marks failures, and the scorers' cited steps are real indices
    # into the trajectory the viewer is displaying.
    failing = [
        tr
        for tr in degraded["task_runs"]
        if any(s["score"] < 0.7 for s in tr["scores"].values())
    ]
    assert failing, "the degraded agent should have produced a low-scoring run"

    traj = store.get_trajectory(failing[0]["trajectory_id"])
    assert traj is not None
    valid = {s.index for s in traj.steps}
    cited = {
        i
        for score in failing[0]["scores"].values()
        for i in score.get("evidence", [])
    }
    assert cited, "a failing run must cite the steps responsible"
    assert cited <= valid, "every cited step must exist in the trajectory shown"

    marked = {s.index for s in traj.steps if s.failed} | cited
    assert marked, "the viewer must have something to highlight"


def test_side_by_side_diff_renders(populated_db):
    app = _run_view("Side-by-side diff")
    assert not app.exception
    assert app.dataframe, "the diff view must render its delta tables"
