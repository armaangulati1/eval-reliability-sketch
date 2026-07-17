"""Study 2: the careful system abstains where the naive system hallucinates."""

from __future__ import annotations

from eval_reliability.abstention import evaluate_system
from eval_reliability.systems import abstaining_system, overextracting_system


def test_abstaining_system_does_not_hallucinate() -> None:
    summary = evaluate_system(abstaining_system, "abstaining")
    # On unanswerable inputs it abstains every time.
    assert summary.hallucination_rate == 0.0
    assert summary.abstention_rate == 1.0


def test_overextracting_system_hallucinates_on_unanswerable() -> None:
    summary = evaluate_system(overextracting_system, "overextracting")
    # The naive system fabricates on every unanswerable input, never abstaining.
    assert summary.hallucination_rate == 1.0
    assert summary.abstention_rate == 0.0


def test_the_abstention_gap_is_legible() -> None:
    careful = evaluate_system(abstaining_system, "abstaining")
    naive = evaluate_system(overextracting_system, "overextracting")
    # The whole point: the gap between careful and naive is the finding.
    assert naive.hallucination_rate > careful.hallucination_rate


def test_abstaining_system_still_answers_answerable_questions() -> None:
    summary = evaluate_system(abstaining_system, "abstaining")
    # Abstention is only careful if it does not collapse to abstaining on
    # everything. The careful system must still answer what it can, correctly.
    assert summary.answer_accuracy == 1.0
    assert summary.over_abstention_rate == 0.0


def test_rates_are_fractions() -> None:
    for system, name in ((abstaining_system, "abstaining"), (overextracting_system, "naive")):
        summary = evaluate_system(system, name)
        for value in (
            summary.hallucination_rate,
            summary.abstention_rate,
            summary.answer_accuracy,
            summary.over_abstention_rate,
        ):
            assert 0.0 <= value <= 1.0
