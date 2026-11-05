"""
Shared numerical constants and ballot-set values.
"""

from __future__ import annotations

from typing import Final

from votekit.types import Candidate

EMPTY_CANDIDATE_SET: Final[frozenset[Candidate]] = frozenset()
END_OF_RANKING_SET: Final[frozenset[str]] = frozenset({"~"})
EMPTY_VOTER_SET: Final[frozenset[str]] = frozenset()

# NOTE: Reserved candidate set IDs. Real candidate sets are numbered from 0 in order
# of first appearance, so ``cand_sets[cand_set_id - EMPTY_SET_ID]`` is the set for any ID. This
# access pattern makes the dense array representation easier to think about.
EMPTY_SET_ID: Final[int] = -2
END_OF_RANKING_ID: Final[int] = -1

MAX_TOTAL_BALLOT_WEIGHT: Final[int] = 2**50

# Decimal precision for rounded scores and the corresponding absolute comparison tolerance.
FLOAT_DECIMAL_PLACES: Final[int] = 10
TOL_EPSILON: Final[float] = 10.0**-FLOAT_DECIMAL_PLACES
