"""Study 1: judge-reliability decay under distribution shift.

Question: an LLM-as-judge is validated on an in-distribution slice and looks
trustworthy. What happens to its agreement with human labels when the inputs
shift (paraphrase, adversarial edge, out of domain)?

This module measures judge-vs-gold agreement per slice and reports the DECAY.
The lesson it makes concrete: a judge you validated in-distribution silently
degrades off-distribution, so eval trust has to be re-established per slice
rather than assumed from a single validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval_reliability.dataset import ALL_SLICES, IN_DISTRIBUTION, SHIFTED_SLICES
from eval_reliability.judge import (
    DEFAULT_AGREEMENT_THRESHOLD,
    Judge,
    validate_judge_on_slice,
)


@dataclass(frozen=True)
class SliceAgreement:
    slice: str
    agreement: float
    admitted: bool


@dataclass(frozen=True)
class DecayReport:
    """Per-slice judge agreement and the resulting admit/exclude decisions."""

    threshold: float
    per_slice: tuple[SliceAgreement, ...]

    def agreement(self, slice_name: str) -> float:
        for row in self.per_slice:
            if row.slice == slice_name:
                return row.agreement
        raise KeyError(slice_name)

    @property
    def in_distribution_agreement(self) -> float:
        return self.agreement(IN_DISTRIBUTION)

    @property
    def mean_shifted_agreement(self) -> float:
        vals = [self.agreement(s) for s in SHIFTED_SLICES]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def admitted_slices(self) -> tuple[str, ...]:
        return tuple(r.slice for r in self.per_slice if r.admitted)

    @property
    def excluded_slices(self) -> tuple[str, ...]:
        return tuple(r.slice for r in self.per_slice if not r.admitted)


def measure_decay(
    judge: Judge,
    threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> DecayReport:
    """Validate a judge on every slice and collect the agreement decay."""
    rows = tuple(
        SliceAgreement(
            slice=slice_name,
            agreement=(v := validate_judge_on_slice(judge, slice_name, threshold)).agreement,
            admitted=v.admitted,
        )
        for slice_name in ALL_SLICES
    )
    return DecayReport(threshold=threshold, per_slice=rows)


def format_decay_report(judge_name: str, report: DecayReport) -> str:
    """Human-readable decay table for the CLI/README demo."""
    lines = [
        f"Judge-reliability decay: {judge_name}",
        "=" * (25 + len(judge_name)),
        f"{'slice':<18}{'agreement':>10}  status",
    ]
    for row in report.per_slice:
        status = "admitted" if row.admitted else "EXCLUDED"
        lines.append(f"{row.slice:<18}{row.agreement:>10.3f}  {status}")
    lines.append("")
    lines.append(
        f"in-distribution {report.in_distribution_agreement:.3f}  ->  "
        f"mean shifted {report.mean_shifted_agreement:.3f}  "
        f"(threshold {report.threshold:.2f})"
    )
    return "\n".join(lines)
