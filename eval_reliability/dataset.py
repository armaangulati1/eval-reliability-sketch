"""Synthetic, self-authored dataset for the reliability study.

Everything here is INVENTED for this demo. There is no real place, no real
record, and no data from any company. The domain is a small, deliberately
fictional almanac of made-up regions so that questions have unambiguous gold
answers and out-of-domain questions are genuinely unanswerable.

The dataset feeds two linked studies:

  * JUDGE_CASES  -- reference-based judge test cases, used to measure how an
    LLM-as-judge stub's agreement with human labels DECAYS as the inputs shift
    away from the slice the judge was validated on (see reliability.py).
  * QA_ITEMS     -- factual questions, some answerable and some deliberately
    unanswerable (out-of-domain or near-miss names), used to measure whether a
    system ABSTAINS or HALLUCINATES on inputs it cannot answer (abstention.py).

Both studies are grounded in the same real observation that motivated this
sketch: a small fine-tuned model, validated in-distribution, silently degraded
off-distribution -- it fabricated answers instead of abstaining. This dataset
reproduces that failure shape deterministically so it can be measured and tested.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Sentinels and slice vocabulary.
# ---------------------------------------------------------------------------

# The answer a careful system should return when it cannot answer.
ABSTAIN = "ABSTAIN"

# The slice a judge is validated on before any distribution shift.
IN_DISTRIBUTION = "in_distribution"

# The deliberately shifted slices. Trust must be re-established on each of these.
SHIFTED_SLICES: tuple[str, ...] = ("paraphrase", "adversarial_edge", "out_of_domain")

ALL_SLICES: tuple[str, ...] = (IN_DISTRIBUTION, *SHIFTED_SLICES)


# ---------------------------------------------------------------------------
# The fictional knowledge base. All entries are invented for this demo.
# ---------------------------------------------------------------------------

# A closed world of made-up regions. A system may answer ONLY from these facts;
# any question about a region not present here is unanswerable by construction.
KB: dict[str, dict[str, str]] = {
    "Velmora": {"export": "saltglass", "settlement": "Ashfen", "founded": "812"},
    "Corrindale": {"export": "copperwood", "settlement": "Brenmoor", "founded": "645"},
    "Tesk Hollow": {"export": "riverpearl", "settlement": "Duncarrow", "founded": "903"},
    "Marrowgate": {"export": "tinbark", "settlement": "Holloway", "founded": "771"},
    "Pellmire": {"export": "glasswheat", "settlement": "Cindral", "founded": "688"},
}

ATTRIBUTES: tuple[str, ...] = ("export", "settlement", "founded")


# ---------------------------------------------------------------------------
# Study 1 data: reference-based judge cases.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeCase:
    """One case for a reference-based judge.

    A judge sees `candidate` and `reference` and predicts whether the candidate
    is correct. `gold_judgment` is the human-authored truth of that prediction.
    Judge quality is agreement between the judge's verdict and `gold_judgment`.
    """

    id: str
    slice: str
    candidate: str
    reference: str
    gold_judgment: bool  # True == the candidate answer is actually correct


# The in-distribution slice: the judge is validated here. Correct candidates are
# phrased like the reference and incorrect candidates are lexically distant, so a
# shallow lexical judge agrees with every human label.
_IN_DISTRIBUTION_CASES: tuple[JudgeCase, ...] = (
    JudgeCase(
        id="id01",
        slice=IN_DISTRIBUTION,
        candidate="The primary export of Velmora is saltglass.",
        reference="The primary export of Velmora is saltglass.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="id02",
        slice=IN_DISTRIBUTION,
        candidate="Corrindale was founded in 645.",
        reference="Corrindale was founded in 645.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="id03",
        slice=IN_DISTRIBUTION,
        candidate="The main settlement of Tesk Hollow is Duncarrow.",
        reference="The main settlement of Tesk Hollow is Duncarrow.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="id04",
        slice=IN_DISTRIBUTION,
        candidate="It trades mostly in riverpearl and old rope.",
        reference="Corrindale was founded in 645.",
        gold_judgment=False,
    ),
    JudgeCase(
        id="id05",
        slice=IN_DISTRIBUTION,
        candidate="No records exist for that request.",
        reference="The main settlement of Velmora is Ashfen.",
        gold_judgment=False,
    ),
)

# Paraphrase slice: correct candidates are reworded, so a shallow lexical judge
# reads low overlap and wrongly calls them incorrect (false negatives).
_PARAPHRASE_CASES: tuple[JudgeCase, ...] = (
    JudgeCase(
        id="pp01",
        slice="paraphrase",
        candidate="Velmora chiefly produces saltglass for trade.",
        reference="The primary export of Velmora is saltglass.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="pp02",
        slice="paraphrase",
        candidate="The founding year for Corrindale comes out to 645.",
        reference="Corrindale was founded in 645.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="pp03",
        slice="paraphrase",
        candidate="Duncarrow serves as the principal town over in Tesk Hollow.",
        reference="The main settlement of Tesk Hollow is Duncarrow.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="pp04",
        slice="paraphrase",
        candidate="They established Marrowgate back around the year 771.",
        reference="Marrowgate was founded in 771.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="pp05",
        slice="paraphrase",
        candidate="The export of Corrindale is copperwood.",
        reference="The export of Corrindale is copperwood.",
        gold_judgment=True,
    ),
)

# Adversarial-edge slice: incorrect candidates are near-copies of the reference
# with one critical change (a swapped number or a negation), so a shallow lexical
# judge reads high overlap and wrongly calls them correct (false positives).
_ADVERSARIAL_CASES: tuple[JudgeCase, ...] = (
    JudgeCase(
        id="ae01",
        slice="adversarial_edge",
        candidate="Velmora was founded in 912.",
        reference="Velmora was founded in 812.",
        gold_judgment=False,
    ),
    JudgeCase(
        id="ae02",
        slice="adversarial_edge",
        candidate="The primary export of Velmora is not saltglass.",
        reference="The primary export of Velmora is saltglass.",
        gold_judgment=False,
    ),
    JudgeCase(
        id="ae03",
        slice="adversarial_edge",
        candidate="Corrindale was founded in 465.",
        reference="Corrindale was founded in 645.",
        gold_judgment=False,
    ),
    JudgeCase(
        id="ae04",
        slice="adversarial_edge",
        candidate="The main settlement of Tesk Hollow is Brenmoor.",
        reference="The main settlement of Tesk Hollow is Duncarrow.",
        gold_judgment=False,
    ),
    JudgeCase(
        id="ae05",
        slice="adversarial_edge",
        candidate="The export of Corrindale is copperwood.",
        reference="The export of Corrindale is copperwood.",
        gold_judgment=True,
    ),
)

# Out-of-domain slice: the reference is an abstention. Correct candidates abstain
# in different words (false negatives for a lexical judge), and hallucinated
# candidates are lexically far from "insufficient information" (agreements, but
# for the wrong reason). Agreement still lands below the in-distribution bar.
_OUT_OF_DOMAIN_CASES: tuple[JudgeCase, ...] = (
    JudgeCase(
        id="od01",
        slice="out_of_domain",
        candidate="I do not have enough data to answer that.",
        reference="There is insufficient information to answer.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="od02",
        slice="out_of_domain",
        candidate="That region is not in my reference set.",
        reference="There is insufficient information to answer.",
        gold_judgment=True,
    ),
    JudgeCase(
        id="od03",
        slice="out_of_domain",
        candidate="Velmora exports saltglass.",
        reference="There is insufficient information to answer.",
        gold_judgment=False,
    ),
    JudgeCase(
        id="od04",
        slice="out_of_domain",
        candidate="The founding year is 700.",
        reference="There is insufficient information to answer.",
        gold_judgment=False,
    ),
    JudgeCase(
        id="od05",
        slice="out_of_domain",
        candidate="There is insufficient information to answer.",
        reference="There is insufficient information to answer.",
        gold_judgment=True,
    ),
)

JUDGE_CASES: tuple[JudgeCase, ...] = (
    *_IN_DISTRIBUTION_CASES,
    *_PARAPHRASE_CASES,
    *_ADVERSARIAL_CASES,
    *_OUT_OF_DOMAIN_CASES,
)


def judge_cases_for_slice(slice_name: str) -> tuple[JudgeCase, ...]:
    """All judge cases belonging to one slice."""
    return tuple(c for c in JUDGE_CASES if c.slice == slice_name)


# ---------------------------------------------------------------------------
# Study 2 data: answerable and unanswerable questions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAItem:
    """One factual question.

    `region` and `attribute` are carried explicitly so the systems are exact and
    deterministic (no free-text parsing). `gold_answer` is the correct answer, or
    the `ABSTAIN` sentinel when the question is unanswerable from the KB.
    """

    id: str
    slice: str
    question: str
    region: str
    attribute: str
    gold_answer: str

    @property
    def answerable(self) -> bool:
        return self.gold_answer != ABSTAIN


QA_ITEMS: tuple[QAItem, ...] = (
    # In-distribution: direct, answerable lookups.
    QAItem(
        id="qa01",
        slice=IN_DISTRIBUTION,
        question="What is the primary export of Velmora?",
        region="Velmora",
        attribute="export",
        gold_answer="saltglass",
    ),
    QAItem(
        id="qa02",
        slice=IN_DISTRIBUTION,
        question="In what year was Corrindale founded?",
        region="Corrindale",
        attribute="founded",
        gold_answer="645",
    ),
    QAItem(
        id="qa03",
        slice=IN_DISTRIBUTION,
        question="What is the main settlement of Tesk Hollow?",
        region="Tesk Hollow",
        attribute="settlement",
        gold_answer="Duncarrow",
    ),
    # Paraphrase: reworded but still answerable.
    QAItem(
        id="qa04",
        slice="paraphrase",
        question="Velmora is best known for exporting what?",
        region="Velmora",
        attribute="export",
        gold_answer="saltglass",
    ),
    QAItem(
        id="qa05",
        slice="paraphrase",
        question="Marrowgate's founding year is what, exactly?",
        region="Marrowgate",
        attribute="founded",
        gold_answer="771",
    ),
    # Out-of-domain: the region is not in the KB, so the question is unanswerable.
    QAItem(
        id="qa06",
        slice="out_of_domain",
        question="What is the primary export of Northreach?",
        region="Northreach",
        attribute="export",
        gold_answer=ABSTAIN,
    ),
    QAItem(
        id="qa07",
        slice="out_of_domain",
        question="In what year was Sable Fen founded?",
        region="Sable Fen",
        attribute="founded",
        gold_answer=ABSTAIN,
    ),
    # Adversarial edge: a near-miss name that lures a fuzzy matcher into
    # fabricating the answer for a real region. The gold answer is still ABSTAIN.
    QAItem(
        id="qa08",
        slice="adversarial_edge",
        question="What is the primary export of Kelmora?",
        region="Kelmora",
        attribute="export",
        gold_answer=ABSTAIN,
    ),
    QAItem(
        id="qa09",
        slice="adversarial_edge",
        question="In what year was Corrindell founded?",
        region="Corrindell",
        attribute="founded",
        gold_answer=ABSTAIN,
    ),
)


def qa_items_for_slice(slice_name: str) -> tuple[QAItem, ...]:
    """All QA items belonging to one slice."""
    return tuple(i for i in QA_ITEMS if i.slice == slice_name)
