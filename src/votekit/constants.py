"""
Shared numerical constants and candidate-set values.
"""

from __future__ import annotations

from typing import Final

from votekit.types import Candidate

EMPTY_CANDIDATE_SET: Final[frozenset[Candidate]] = frozenset()

# Decimal precision for rounded scores and the corresponding absolute comparison tolerance.
FLOAT_DECIMAL_PLACES: Final[int] = 10
TOL_EPSILON: Final[float] = 10.0**-FLOAT_DECIMAL_PLACES
