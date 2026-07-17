"""The reference-based judge and the per-slice excluded-judge discipline.

An LLM-as-judge is convenient, but a judge that looks reliable on the slice you
validated it on can silently degrade off-distribution. The core discipline in
this module: a judge is validated PER SLICE, and it is EXCLUDED from any slice
where its agreement with the human labels falls below a threshold, rather than
being trusted globally because it passed once in-distribution.

The judge here is a deterministic stub so the methodology is testable offline
with no API key. It models a real reference-based judge's shallow failure mode:
it rewards surface lexical similarity to the reference rather than semantic
equivalence. A real LLM judge would drop in behind the same `Judge` signature
and face the identical per-slice gate.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from eval_reliability.dataset import (
    IN_DISTRIBUTION,
    JudgeCase,
    judge_cases_for_slice,
)

# A judge sees (candidate, reference) and predicts whether the candidate is
# correct. It returns True for "correct", False for "incorrect".
Judge = Callable[[str, str], bool]

DEFAULT_AGREEMENT_THRESHOLD = 0.9
DEFAULT_OVERLAP_TAU = 0.5

_TOKEN_RE = re.compile(r"\w+")


class JudgeExcludedError(RuntimeError):
    """Raised when a judge that failed slice validation is asked for a number."""


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(candidate: str, reference: str) -> float:
    """Token-set Jaccard overlap between a candidate and a reference answer."""
    a = _tokens(candidate)
    b = _tokens(reference)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def lexical_overlap_judge(candidate: str, reference: str, tau: float = DEFAULT_OVERLAP_TAU) -> bool:
    """A deterministic reference-based judge stub.

    It calls a candidate CORRECT when its lexical overlap with the reference is
    at least `tau`. This is faithful to how a shallow judge fails: it approves
    surface-similar answers and rejects correctly-paraphrased ones.
    """
    return jaccard(candidate, reference) >= tau


@dataclass(frozen=True)
class JudgeValidation:
    """The result of validating a judge on one slice."""

    slice: str
    agreement: float
    threshold: float
    admitted: bool
    disagreements: tuple[str, ...]


def validate_judge_on_slice(
    judge: Judge,
    slice_name: str,
    threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> JudgeValidation:
    """Check a judge's agreement with the human labels on a single slice."""
    cases: tuple[JudgeCase, ...] = judge_cases_for_slice(slice_name)
    disagreements = tuple(c.id for c in cases if judge(c.candidate, c.reference) != c.gold_judgment)
    total = len(cases)
    agreement = (total - len(disagreements)) / total if total else 0.0
    return JudgeValidation(
        slice=slice_name,
        agreement=agreement,
        threshold=threshold,
        admitted=agreement >= threshold,
        disagreements=disagreements,
    )


def judge_agreement_for_slice(
    judge: Judge,
    slice_name: str,
    threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> float:
    """Return a judge's agreement on a slice, but ONLY if it passes validation.

    If the judge fails validation on this slice it raises rather than returning a
    number. That is the whole discipline: an unreliable judge contributes nothing
    on the slice where it is unreliable, even if it passed in-distribution.
    """
    validation = validate_judge_on_slice(judge, slice_name, threshold)
    if not validation.admitted:
        raise JudgeExcludedError(
            f"judge excluded on slice {slice_name!r}: agreement "
            f"{validation.agreement:.3f} < threshold {validation.threshold:.3f} "
            f"(disagreed on {list(validation.disagreements)})"
        )
    return validation.agreement


def summarize_validation(validation: JudgeValidation) -> str:
    verdict = "ADMITTED" if validation.admitted else "EXCLUDED"
    return (
        f"[{verdict}] slice={validation.slice} "
        f"agreement={validation.agreement:.3f} "
        f"threshold={validation.threshold:.3f} "
        f"disagreements={list(validation.disagreements)}"
    )


def naive_global_admission(
    judge: Judge,
    threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> bool:
    """The tempting shortcut: validate a judge ONCE, in-distribution, and trust it.

    This returns True when the judge passes on the in-distribution slice alone.
    The point of the study is that this single check is not enough -- the same
    judge must be re-validated on every shifted slice (see per_slice_admission).
    """
    return validate_judge_on_slice(judge, IN_DISTRIBUTION, threshold).admitted
