"""Per-slice excluded-judge discipline: a judge is re-validated on every slice."""

from __future__ import annotations

import pytest

from eval_reliability.dataset import IN_DISTRIBUTION, SHIFTED_SLICES
from eval_reliability.judge import (
    JudgeExcludedError,
    judge_agreement_for_slice,
    lexical_overlap_judge,
    naive_global_admission,
    validate_judge_on_slice,
)


def test_judge_is_admitted_in_distribution() -> None:
    validation = validate_judge_on_slice(lexical_overlap_judge, IN_DISTRIBUTION)
    assert validation.admitted
    assert validation.agreement == 1.0


def test_judge_is_excluded_on_at_least_one_shifted_slice() -> None:
    excluded = [
        s
        for s in SHIFTED_SLICES
        if not validate_judge_on_slice(lexical_overlap_judge, s).admitted
    ]
    assert excluded, "a decayed judge should be excluded somewhere off-distribution"


def test_admitted_slice_returns_a_number() -> None:
    agreement = judge_agreement_for_slice(lexical_overlap_judge, IN_DISTRIBUTION)
    assert 0.0 <= agreement <= 1.0


def test_excluded_slice_refuses_to_return_a_number() -> None:
    # Find a slice the judge fails, and confirm it raises rather than reporting.
    failing = next(
        s
        for s in SHIFTED_SLICES
        if not validate_judge_on_slice(lexical_overlap_judge, s).admitted
    )
    with pytest.raises(JudgeExcludedError):
        judge_agreement_for_slice(lexical_overlap_judge, failing)


def test_global_validation_would_have_trusted_the_judge_everywhere() -> None:
    # The trap the discipline guards against: validating once in-distribution
    # admits the judge, even though it must be excluded on shifted slices.
    assert naive_global_admission(lexical_overlap_judge)
    per_slice_excluded = [
        s
        for s in SHIFTED_SLICES
        if not validate_judge_on_slice(lexical_overlap_judge, s).admitted
    ]
    assert per_slice_excluded


def test_threshold_is_enforced() -> None:
    # Even a perfect in-distribution judge is excluded if the bar exceeds 1.0.
    validation = validate_judge_on_slice(lexical_overlap_judge, IN_DISTRIBUTION, threshold=1.01)
    assert not validation.admitted
