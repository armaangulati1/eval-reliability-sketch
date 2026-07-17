"""Dataset integrity: the synthetic fixtures and gold labels are well-formed."""

from __future__ import annotations

from eval_reliability.dataset import (
    ABSTAIN,
    ALL_SLICES,
    ATTRIBUTES,
    IN_DISTRIBUTION,
    JUDGE_CASES,
    KB,
    QA_ITEMS,
    SHIFTED_SLICES,
    judge_cases_for_slice,
    qa_items_for_slice,
)


def test_judge_case_ids_are_unique() -> None:
    ids = [c.id for c in JUDGE_CASES]
    assert len(ids) == len(set(ids))


def test_qa_item_ids_are_unique() -> None:
    ids = [i.id for i in QA_ITEMS]
    assert len(ids) == len(set(ids))


def test_judge_cases_cover_every_slice() -> None:
    for slice_name in ALL_SLICES:
        assert judge_cases_for_slice(slice_name), f"{slice_name} has no judge cases"


def test_qa_items_cover_every_slice() -> None:
    for slice_name in ALL_SLICES:
        assert qa_items_for_slice(slice_name), f"{slice_name} has no QA items"


def test_judge_gold_judgments_are_bool() -> None:
    for c in JUDGE_CASES:
        assert isinstance(c.gold_judgment, bool)


def test_in_distribution_judge_slice_has_both_labels() -> None:
    # The validated slice must contain correct AND incorrect candidates, or the
    # judge's in-distribution agreement would be trivial.
    labels = {c.gold_judgment for c in judge_cases_for_slice(IN_DISTRIBUTION)}
    assert labels == {True, False}


def test_answerable_items_are_grounded_in_kb() -> None:
    # An answerable question must reference a real KB region + attribute, and its
    # gold answer must be exactly that KB value.
    for item in QA_ITEMS:
        if item.answerable:
            assert item.region in KB
            assert item.attribute in ATTRIBUTES
            assert KB[item.region][item.attribute] == item.gold_answer


def test_unanswerable_items_are_out_of_kb() -> None:
    # An unanswerable question must NOT reference a real KB region, so ABSTAIN is
    # genuinely the correct answer rather than an arbitrary label.
    for item in QA_ITEMS:
        if not item.answerable:
            assert item.gold_answer == ABSTAIN
            assert item.region not in KB


def test_qa_set_has_both_answerable_and_unanswerable() -> None:
    assert any(i.answerable for i in QA_ITEMS)
    assert any(not i.answerable for i in QA_ITEMS)


def test_shifted_slices_exclude_the_reference_slice() -> None:
    assert IN_DISTRIBUTION not in SHIFTED_SLICES
    assert set(ALL_SLICES) == {IN_DISTRIBUTION, *SHIFTED_SLICES}
