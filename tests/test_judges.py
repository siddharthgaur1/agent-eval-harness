"""LLM judges with a stubbed client: parsing, retry, and evidence filtering."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.scorers import score_trajectory
from src.scorers.base import LLM_JUDGES
from src.scorers.judge import JudgeError, call_judge


class StubClient:
    """Minimal stand-in for the OpenAI client, returning canned bodies in order."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0) if self.responses else self.responses_exhausted()
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))]
        )

    def responses_exhausted(self):
        raise AssertionError("stub client ran out of responses")


GOOD = json.dumps(
    {"score": 0.8, "reasoning": "the plan was sound and mostly followed", "evidence_steps": [0, 2]}
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff would make this suite take 14 seconds for nothing."""
    monkeypatch.setattr("src.scorers.judge.time.sleep", lambda _s: None)


def test_judge_parses_structured_output():
    client = StubClient([GOOD])
    verdict = call_judge("sys", "user", scorer="t", trajectory_id="x", client=client)
    assert verdict.score == 0.8
    assert verdict.evidence_steps == [0, 2]


def test_judge_uses_temperature_zero_and_json_mode():
    client = StubClient([GOOD])
    call_judge("sys", "user", scorer="t", trajectory_id="x", client=client)
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_judge_retries_past_malformed_json():
    client = StubClient(["not json at all", GOOD])
    verdict = call_judge("sys", "user", scorer="t", trajectory_id="x", client=client)
    assert verdict.score == 0.8
    assert len(client.calls) == 2


def test_judge_retries_past_transport_errors():
    client = StubClient([RuntimeError("429 rate limited"), GOOD])
    assert call_judge("sys", "u", scorer="t", trajectory_id="x", client=client).score == 0.8


def test_judge_gives_up_after_max_retries():
    client = StubClient(["nope"] * 5)
    with pytest.raises(JudgeError, match="failed after"):
        call_judge("sys", "u", scorer="t", trajectory_id="x", client=client)


def test_judge_out_of_range_score_is_rejected_then_retried():
    client = StubClient([json.dumps({"score": 4.2, "reasoning": "r", "evidence_steps": [0]}), GOOD])
    assert call_judge("sys", "u", scorer="t", trajectory_id="x", client=client).score == 0.8


def test_raw_response_is_persisted_for_audit(tmp_path):
    from src.persistence.store import Store

    store = Store(tmp_path / "t.db")
    client = StubClient([GOOD])
    call_judge("sys", "user prompt", scorer="plan_coherence", trajectory_id="traj1",
               store=store, client=client)
    saved = store.judge_calls("traj1")
    assert len(saved) == 1
    assert saved[0]["scorer"] == "plan_coherence"
    assert "user prompt" in saved[0]["prompt"]
    assert json.loads(saved[0]["response"])["score"] == 0.8


def test_judge_evidence_is_filtered_to_real_steps(perfect, task):
    """A judge that hallucinates step 99 must not smuggle it into the evidence."""
    payload = json.dumps(
        {"score": 0.5, "reasoning": "mixed", "evidence_steps": [1, 99, -3]}
    )
    judge = LLM_JUDGES["plan_coherence"]
    client = StubClient([payload])
    result = judge.score(perfect, task, client=client)
    assert result.evidence == [1]
    assert result.details["judge_cited"] == [1, 99, -3]


def test_all_three_judges_are_registered():
    assert set(LLM_JUDGES) == {
        "plan_coherence",
        "trajectory_reasoning",
        "output_faithfulness",
    }


def test_a_broken_judge_does_not_abort_the_run(perfect, task):
    """One flaky API call must not discard a run that took minutes to produce."""
    client = StubClient([RuntimeError("boom")] * 20)
    scores = score_trajectory(perfect, task, include_llm=True, client=client)
    assert scores["task_completion"].score == 1.0  # deterministic scorers survived
    assert scores["plan_coherence"].score == 0.0
    assert scores["plan_coherence"].details["unscored"] is True


def test_faithfulness_prompt_contains_only_observed_tool_results(perfect, task):
    client = StubClient([GOOD])
    LLM_JUDGES["output_faithfulness"].score(perfect, task, client=client)
    prompt = client.calls[0]["messages"][1]["content"]
    assert "TOOL RESULTS THE AGENT OBSERVED" in prompt
    assert "FINAL OUTPUT TO VERIFY" in prompt
