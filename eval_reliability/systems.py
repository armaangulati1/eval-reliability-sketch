"""Two reference systems under evaluation, plus an optional live hook.

These make the abstention study visible. Both take a `QAItem` and return either
an answer string or the `ABSTAIN` sentinel.

  abstaining_system     -- careful. Answers ONLY when the region is exactly in
                           the knowledge base; otherwise it abstains. This is the
                           behavior a model should have off-distribution.
  overextracting_system -- naive. Fuzzy-matches the region to the nearest KB
                           entry and always returns an answer, so it fabricates
                           on near-miss names and out-of-domain questions and
                           never abstains. This reproduces the real failure that
                           motivated the sketch: a fine-tune that hallucinated
                           out-of-distribution instead of saying "I don't know."

Everything is deterministic and offline. The optional `llm_system` is a
documented stub and is never required by the tests or the CLI.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from difflib import SequenceMatcher

from eval_reliability.dataset import ABSTAIN, KB, QAItem

System = Callable[[QAItem], str]


def abstaining_system(item: QAItem) -> str:
    """Answer only on an exact KB hit; abstain otherwise."""
    facts = KB.get(item.region)
    if facts is None:
        return ABSTAIN
    value = facts.get(item.attribute)
    if value is None:
        return ABSTAIN
    return value


def _nearest_region(name: str) -> str:
    """The KB region whose name is most similar to `name` (deterministic)."""
    return max(KB, key=lambda region: SequenceMatcher(None, name.lower(), region.lower()).ratio())


def overextracting_system(item: QAItem) -> str:
    """Always answer by snapping the region to its nearest KB neighbor.

    This never abstains. On a near-miss name (for example a region that is not in
    the KB but looks like one that is) it confidently returns the neighbor's
    value, which is a fabrication.
    """
    region = item.region if item.region in KB else _nearest_region(item.region)
    return KB[region][item.attribute]


# ---------------------------------------------------------------------------
# Optional live system: opt-in, never required by the offline suite.
# ---------------------------------------------------------------------------


def llm_system_available() -> bool:
    """True only if an API key is present. The offline suite never needs this."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_system(item: QAItem) -> str:  # pragma: no cover
    """A documented, opt-in live-model stub.

    Intentionally NOT wired to a client here. If you wanted to test a real model,
    you would implement the call, instruct it to answer only when it can and to
    return the abstain sentinel otherwise, and then run its outputs through the
    SAME abstention metrics as the deterministic systems.
    """
    raise NotImplementedError(
        "llm_system is a documented stub in this demo. Implement a client call "
        "here to test a real model; its answers would be scored by the identical "
        "abstention metrics used for the deterministic systems."
    )
