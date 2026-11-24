from collections.abc import Sequence
from typing import (
    Any,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix
from votekit.types import Candidate

RankingRow = NDArray[np.object_]

ScoresRow = NDArray[np.float64]

MatrixT = TypeVar("MatrixT", RankMatrix, ScoreMatrix)

RowT_contra = TypeVar("RowT_contra", bound=NDArray[Any], contravariant=True)

ReturnT_co = TypeVar("ReturnT_co", bound=NDArray[Any] | Sequence[Any], covariant=True)


@runtime_checkable
class ProfileTransform(Protocol[MatrixT]):
    """
    Contract for profile transformations that act on all selected voters at once.

    Any object with a ``transform_matrix`` method can be passed to ``transform()`` and is
    assignable to this protocol. ``transform_matrix`` consumes a matrix of the selected rows, of
    the same kind as the profile's matrix, and the generator of the ``transform()`` call. Each
    row's weight is its number of selected voters, in the profile's weight kind, and rows are
    grouped as in the profile rather than split per voter.

    It must return a valid matrix of the same kind. Nothing else is checked: the result may
    change the total weight, the weights, the voter sets, or the ranking or score columns. A
    candidate in the result's ``candidates`` that the profile lacks is added to the profile's
    candidates. A random transform draws from the generator it receives, so one ``rng`` passed to
    ``transform()`` reproduces both the voter selection and the transform. A transform that
    should treat each unit of weight as an independent voter must split or bin the rows itself.

    Arguments are passed by position, so an implementation may name its parameters freely.
    """

    def transform_matrix(self, matrix: MatrixT, rng: np.random.Generator, /) -> MatrixT: ...


class RowTransformFn(Protocol[RowT_contra, ReturnT_co]):
    """
    Contract for functions that ``map()`` applies to one ballot at a time.

    The function receives one ballot as a writeable 1D NumPy array and the generator of the
    ``map()`` call. A ballot is one frozenset of candidates per ranking position for a ranking,
    or one float per score column for scores. It returns the new ballot as a 1D array or a
    sequence, such as a list or tuple, and may modify its input in place and return it. ``map``
    calls it once per row, or once per voter, always with the same generator, so a random
    function gives each call its own draw. A deterministic function ignores the generator.

    Arguments are passed by position, so a function may name its parameters freely.

    Use ``RankRowTransformFn`` or ``ScoreRowTransformFn`` for one kind of ballot.
    """

    def __call__(self, row: RowT_contra, rng: np.random.Generator, /) -> ReturnT_co: ...


RankRowTransformFn: TypeAlias = RowTransformFn[
    RankingRow, RankingRow | Sequence[frozenset[Candidate] | set[Candidate]]
]
"""A row transform function for rankings: candidate sets in, candidate sets out."""

ScoreRowTransformFn: TypeAlias = RowTransformFn[ScoresRow, ScoresRow | Sequence[float]]
"""A row transform function for scores: scores in, scores out."""
