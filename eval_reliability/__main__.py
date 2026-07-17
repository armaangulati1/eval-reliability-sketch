"""Run the study offline:  python -m eval_reliability

Prints both studies:
  1. judge-reliability decay across distribution-shift slices, with the per-slice
     admit/exclude decisions; and
  2. abstention vs hallucination for the careful and the naive systems.

No network, no API key required.
"""

from __future__ import annotations

from eval_reliability.abstention import evaluate_system, format_abstention_report
from eval_reliability.judge import (
    lexical_overlap_judge,
    naive_global_admission,
)
from eval_reliability.reliability import format_decay_report, measure_decay
from eval_reliability.systems import abstaining_system, overextracting_system


def main() -> None:
    print("Study 1 -- judge reliability under distribution shift")
    print("-----------------------------------------------------")
    report = measure_decay(lexical_overlap_judge)
    print(format_decay_report("lexical_overlap_judge", report))
    print()
    admitted = naive_global_admission(lexical_overlap_judge)
    print(
        f"Validated in-distribution only? admitted={admitted}. "
        f"Re-validated per slice, EXCLUDED on: {list(report.excluded_slices)}."
    )
    print("A single in-distribution check would have trusted this judge everywhere.")
    print()
    print()

    print("Study 2 -- abstention vs hallucination on unanswerable inputs")
    print("-------------------------------------------------------------")
    careful = evaluate_system(abstaining_system, "abstaining_system")
    naive = evaluate_system(overextracting_system, "overextracting_system")
    print(format_abstention_report(careful))
    print()
    print(format_abstention_report(naive))
    print()
    print(
        f"Hallucination on unanswerable inputs: "
        f"abstaining_system {careful.hallucination_rate:.3f}  vs  "
        f"overextracting_system {naive.hallucination_rate:.3f}."
    )


if __name__ == "__main__":
    main()
