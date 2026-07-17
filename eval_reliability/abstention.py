"""Study 2: abstention vs hallucination on unanswerable inputs.

Question: when a question cannot be answered from what the system knows (an
out-of-domain region, or a near-miss name that only looks answerable), does the
system ABSTAIN or does it FABRICATE an answer?

This module splits the QA set into answerable and unanswerable items and reports,
per system:

  * hallucination_rate  -- on UNANSWERABLE items, the fraction where the system
                           produced an answer instead of abstaining. This is the
                           headline number and it should be low.
  * abstention_rate     -- on UNANSWERABLE items, the fraction abstained.
  * answer_accuracy     -- on ANSWERABLE items, the fraction answered correctly
                           (guards against a system that just abstains on
                           everything to look safe).
  * over_abstention_rate-- on ANSWERABLE items, the fraction wrongly abstained.

The gap between a careful abstaining system and a naive over-extracting one is
the whole point, and it mirrors the real observation that motivated the sketch.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval_reliability.dataset import ABSTAIN, QA_ITEMS, QAItem
from eval_reliability.systems import System


@dataclass(frozen=True)
class AbstentionSummary:
    system_name: str
    unanswerable_total: int
    abstained: int
    hallucinated: int
    answerable_total: int
    answered_correct: int
    over_abstained: int

    @property
    def abstention_rate(self) -> float:
        return self.abstained / self.unanswerable_total if self.unanswerable_total else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated / self.unanswerable_total if self.unanswerable_total else 0.0

    @property
    def answer_accuracy(self) -> float:
        return self.answered_correct / self.answerable_total if self.answerable_total else 0.0

    @property
    def over_abstention_rate(self) -> float:
        return self.over_abstained / self.answerable_total if self.answerable_total else 0.0


def evaluate_system(
    system: System,
    system_name: str,
    items: tuple[QAItem, ...] = QA_ITEMS,
) -> AbstentionSummary:
    """Run a system across the QA set and measure abstention vs hallucination."""
    unanswerable_total = abstained = hallucinated = 0
    answerable_total = answered_correct = over_abstained = 0

    for item in items:
        answer = system(item)
        if item.answerable:
            answerable_total += 1
            if answer == ABSTAIN:
                over_abstained += 1
            elif answer == item.gold_answer:
                answered_correct += 1
        else:
            unanswerable_total += 1
            if answer == ABSTAIN:
                abstained += 1
            else:
                hallucinated += 1

    return AbstentionSummary(
        system_name=system_name,
        unanswerable_total=unanswerable_total,
        abstained=abstained,
        hallucinated=hallucinated,
        answerable_total=answerable_total,
        answered_correct=answered_correct,
        over_abstained=over_abstained,
    )


def format_abstention_report(summary: AbstentionSummary) -> str:
    """Human-readable abstention table for the CLI/README demo."""
    return "\n".join(
        (
            f"Abstention vs hallucination: {summary.system_name}",
            "=" * (29 + len(summary.system_name)),
            f"unanswerable items : {summary.unanswerable_total}",
            f"  hallucination_rate : {summary.hallucination_rate:.3f} "
            f"({summary.hallucinated}/{summary.unanswerable_total})",
            f"  abstention_rate    : {summary.abstention_rate:.3f} "
            f"({summary.abstained}/{summary.unanswerable_total})",
            f"answerable items   : {summary.answerable_total}",
            f"  answer_accuracy    : {summary.answer_accuracy:.3f} "
            f"({summary.answered_correct}/{summary.answerable_total})",
            f"  over_abstention    : {summary.over_abstention_rate:.3f} "
            f"({summary.over_abstained}/{summary.answerable_total})",
        )
    )
