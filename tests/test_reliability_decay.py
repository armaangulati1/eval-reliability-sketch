"""Study 1: judge agreement decays off-distribution, and the direction holds."""

from __future__ import annotations

from eval_reliability.dataset import IN_DISTRIBUTION, SHIFTED_SLICES
from eval_reliability.judge import lexical_overlap_judge
from eval_reliability.reliability import measure_decay


def test_in_distribution_agreement_is_high() -> None:
    report = measure_decay(lexical_overlap_judge)
    # The judge is validated here; it should agree with every human label.
    assert report.in_distribution_agreement == 1.0


def test_every_shifted_slice_is_no_better_than_in_distribution() -> None:
    report = measure_decay(lexical_overlap_judge)
    baseline = report.in_distribution_agreement
    for slice_name in SHIFTED_SLICES:
        # The core claim: agreement decays (or at best holds) under shift.
        assert report.agreement(slice_name) <= baseline


def test_mean_shifted_agreement_is_strictly_below_baseline() -> None:
    report = measure_decay(lexical_overlap_judge)
    # The decay is real, not a rounding artifact.
    assert report.mean_shifted_agreement < report.in_distribution_agreement


def test_at_least_one_shifted_slice_falls_below_threshold() -> None:
    report = measure_decay(lexical_overlap_judge)
    # If nothing were excluded, the per-slice discipline would be pointless.
    assert report.excluded_slices


def test_in_distribution_slice_is_admitted() -> None:
    report = measure_decay(lexical_overlap_judge)
    assert IN_DISTRIBUTION in report.admitted_slices


def test_agreement_values_are_fractions() -> None:
    report = measure_decay(lexical_overlap_judge)
    for row in report.per_slice:
        assert 0.0 <= row.agreement <= 1.0
