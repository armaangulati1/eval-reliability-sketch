# eval-reliability-sketch

A small, self-authored empirical study of two failure modes that quietly break
trust in a model evaluation: an LLM-as-judge that degrades off-distribution, and
a system that fabricates answers instead of abstaining. All data is synthetic and
invented for this demo. Not affiliated with, and not built on data from, any
company. Demo scope, self-authored synthetic data, offline-reproducible.

---

## Why this exists

While fine-tuning a small model on a self-authored gold set, I hit a concrete
failure: the model looked reliable on data that matched what it had seen, and
then, out-of-distribution, it fabricated answers instead of saying it did not
know. That one observation contains two separable questions about **evaluator
reliability**, and this repo studies both on a deterministic synthetic set so
they can be measured, tested, and reproduced.

1. **Judge-reliability decay.** If you validate an LLM-as-judge on one slice of
   data and trust it, what happens to its agreement with human labels when the
   inputs shift? The claim under test: an evaluator you validated in-distribution
   silently degrades off-distribution, so eval trust must be re-established per
   slice rather than assumed from a single check.
2. **Abstention vs hallucination.** On inputs a system cannot answer, does it
   abstain ("insufficient information") or fabricate? The claim under test: a
   careful system abstains where a naive over-extracting one hallucinates, and
   that gap is the thing worth measuring.

Everything runs offline with the standard library. No API key is required.

## The synthetic domain

The data is a deliberately fictional almanac of made-up regions (invented names,
invented exports, invented founding years) in `eval_reliability/dataset.py`.
There is no real place, no real record, and no company or product anywhere in
the repo. A closed knowledge base makes "answerable" and "unanswerable"
unambiguous, and a guard test (`tests/test_no_company_names.py`) enforces that
the artifact stays generic.

## Study 1: judge-reliability decay under distribution shift

A reference-based judge sees a candidate answer and a reference answer and
predicts whether the candidate is correct. The stub judge shipped here
(`lexical_overlap_judge`) is deterministic and models a real judge's shallow
failure mode: it rewards surface lexical overlap with the reference rather than
semantic equivalence. Human-authored judge cases span four slices, and the
judge is validated against the human labels on each.

Run `python -m eval_reliability`:

| slice | judge-vs-gold agreement | admitted at threshold 0.90? |
| --- | --- | --- |
| `in_distribution` | **1.000** | admitted |
| `paraphrase` | **0.200** | EXCLUDED |
| `adversarial_edge` | **0.200** | EXCLUDED |
| `out_of_domain` | **0.600** | EXCLUDED |

In-distribution agreement is 1.000; the mean over the three shifted slices is
0.333. The decay is not random, it is structural:

- **paraphrase**: correct answers are reworded, so a lexical judge reads low
  overlap and wrongly rejects them (false negatives).
- **adversarial_edge**: wrong answers are near-copies of the reference with one
  critical change (a swapped number, a negation), so a lexical judge reads high
  overlap and wrongly accepts them (false positives).
- **out_of_domain**: the reference is an abstention, and a correct abstention
  phrased differently reads as low overlap and is wrongly rejected.

### The per-slice excluded-judge discipline

The tempting shortcut is to validate a judge once, in-distribution, and trust it
everywhere. `naive_global_admission` shows that shortcut would **admit** this
judge (it scores 1.000 in-distribution). The discipline in `judge.py` instead
re-validates the judge on every slice and **excludes** it wherever its agreement
falls below threshold: `judge_agreement_for_slice` raises `JudgeExcludedError`
on a slice the judge fails rather than reporting a number you cannot trust. A
real LLM judge would drop in behind the same `Judge` signature and face the
identical per-slice gate.

## Study 2: abstention vs hallucination on unanswerable inputs

Two systems answer the same questions. Some questions are answerable from the
knowledge base; others are unanswerable, either out-of-domain or a near-miss name
that only looks answerable (for example a region that does not exist but
resembles one that does).

| system | hallucination rate (unanswerable) | abstention rate (unanswerable) | answer accuracy (answerable) |
| --- | --- | --- | --- |
| `abstaining_system` | **0.000** (0/4) | 1.000 (4/4) | 1.000 (5/5) |
| `overextracting_system` | **1.000** (4/4) | 0.000 (0/4) | 1.000 (5/5) |

`abstaining_system` answers only on an exact knowledge-base hit and abstains
otherwise, so it never fabricates and still answers every answerable question
correctly. `overextracting_system` snaps every question to its nearest
knowledge-base neighbor and always answers, so it fabricates on all four
unanswerable inputs. The gap between the two hallucination columns is the finding,
and it is the same failure shape as the fine-tune that motivated the sketch,
reproduced deterministically. Answer accuracy is reported alongside so that
"abstain" cannot be gamed by a system that just refuses everything.

## Run it

```bash
pip install -e ".[dev]"
python -m eval_reliability   # prints both studies + the admit/exclude decisions
ruff check .
pytest -q                    # offline test suite
```

## Optional live judge

`eval_reliability/systems.py` has a documented `llm_system` stub and
`judge.py` accepts any callable with the `Judge` signature. If you wired a real
model to either (via `ANTHROPIC_API_KEY`), its outputs would run through the
exact same abstention metrics and the exact same per-slice exclusion gate as the
deterministic stubs. The live path is intentionally not wired here and is never
required by the tests or CI.

## Layout

```
eval_reliability/
  dataset.py       # synthetic almanac KB + judge cases + QA items (all invented)
  judge.py         # reference-based stub judge + per-slice excluded-judge discipline
  reliability.py   # study 1: judge-agreement decay across slices
  systems.py       # study 2: abstaining vs over-extracting systems + optional LLM stub
  abstention.py    # study 2: abstention-vs-hallucination metrics
  __main__.py      # `python -m eval_reliability` demo runner
tests/             # offline pytest: dataset integrity, decay direction, abstention gap,
                   #   per-slice judge exclusion, no-company-names guard
```

## Scope and honesty

- **Synthetic only.** Every fact, question, and judge case is invented for this
  demo. There is no real place, no real record, and no data from any company.
- **The numbers describe this small self-authored set.** The 1.000 / 0.200 /
  0.600 judge agreements and the 0.000 vs 1.000 hallucination rates are agreement
  between deterministic components and the author's own labels on a tiny invented
  corpus. They never travel bare and they are not a general accuracy claim. A
  larger or independently labeled set could surface different numbers.
- **The stub judge is a model of a failure mode, not a real judge.** It is a
  deterministic lexical-overlap function chosen so the decay is legible and
  testable offline. A real LLM judge would show the same class of decay for
  different reasons; the reusable artifact is the per-slice discipline, not the
  stub.
- **Single study, one design.** The slices, thresholds, and systems are
  hand-authored to make the two effects visible. This demonstrates how I reason
  about evaluator reliability and abstention; it is not a benchmark and not a
  production evaluation system.
- **Not affiliated** with any company or product. A test enforces that no
  company or product name appears anywhere in the repo.
