"""eval-reliability-sketch: a small, self-authored study of evaluator reliability.

Two linked empirical questions on a synthetic, self-authored dataset:
  1. how an LLM-as-judge's agreement with human labels decays under distribution
     shift, and why eval trust must be re-established per slice; and
  2. whether a system abstains or hallucinates on unanswerable inputs.

All data is synthetic and invented for this demo. Not affiliated with, and not
built on data from, any company. Demo scope, self-authored synthetic data,
offline-reproducible.
"""

from eval_reliability.abstention import (
    AbstentionSummary,
    evaluate_system,
)
from eval_reliability.dataset import (
    ABSTAIN,
    JUDGE_CASES,
    KB,
    QA_ITEMS,
    JudgeCase,
    QAItem,
)
from eval_reliability.judge import (
    JudgeExcludedError,
    JudgeValidation,
    lexical_overlap_judge,
    validate_judge_on_slice,
)
from eval_reliability.reliability import DecayReport, measure_decay
from eval_reliability.systems import abstaining_system, overextracting_system

__all__ = [
    "ABSTAIN",
    "JUDGE_CASES",
    "QA_ITEMS",
    "KB",
    "JudgeCase",
    "QAItem",
    "lexical_overlap_judge",
    "validate_judge_on_slice",
    "JudgeValidation",
    "JudgeExcludedError",
    "measure_decay",
    "DecayReport",
    "abstaining_system",
    "overextracting_system",
    "evaluate_system",
    "AbstentionSummary",
]
