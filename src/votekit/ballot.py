from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from numbers import Integral, Real
from typing import Self, overload

from votekit.types import Candidate, Ranking, RankingLike, ScoresLike
from votekit.utils import SourceWithAttribute, _validate_candidate_names


def validate_weight(weight: object) -> None:
    """
    Validate a single real, finite, nonnegative ballot weight without converting it.

    Supported floating weights are Python float and NumPy float64. Integers and Fractions are
    checked without conversion to float, so large finite values do not overflow during validation.

    Args:
        weight (object): Proposed ballot weight. Booleans are rejected even though bool is
            a subclass of int.

    Raises:
        TypeError: The weight is a bool or has an unsupported numeric type.
        ValueError: The weight is negative, NaN, or infinite.
    """
    if isinstance(weight, bool) or not isinstance(weight, (float, Integral, Fraction)):
        raise TypeError(
            "Ballot weight must be a float, int, or Fraction. "
            "Floating weights must use Python float or NumPy float64. "
            f"Got {type(weight).__name__}."
        )
    if isinstance(weight, float) and not math.isfinite(weight):
        raise ValueError(f"Ballot weight cannot be NaN or infinity. Got {weight}.")
    if weight < 0:
        raise ValueError(f"Ballot weight cannot be negative. Got {weight}.")


def _to_float_weight(weight: float | int) -> float:
    """
    Retain a validated float or convert an integer without rounding or overflow.

    Args:
        weight (float | int): Supported, finite, nonnegative float or integer already checked
            by ``validate_weight``.

    Returns:
        float: Exactly represented weight as a Python float.

    Raises:
        ValueError: Conversion changes the supplied value or exceeds float64's finite range.
    """
    if isinstance(weight, float):
        # NumPy float64 can pass this guard. Convert to python float.
        return float(weight)

    error_message = (
        "Ballot weight cannot be represented exactly as float64. "
        "Pass an explicit float to allow rounding, or a Fraction for an exact weight."
    )
    try:
        converted = float(weight)
    except OverflowError as error:
        raise ValueError(error_message) from error

    # Compare integers to avoid NumPy rounding the source value during comparison.
    if int(converted) != int(weight):
        raise ValueError(error_message)
    return converted


class Ballot:
    """
    Ballot parent class, contains voter set and assigned weight.

    Args:
        ranking (Sequence[Candidate | Iterable[Candidate]] | None): Candidate ranking.
            Entry i of the sequence is a candidate or iterable of candidates ranked in position i.
            Candidates can be strings, integers, or mix of both.
            Defaults to None. Will be coerced to tuple[frozenset[Candidate], ...].
        weight (float | int | Fraction): Weight assigned to a given ballot. Defaults to 1.0.
            Fractions are preserved. Floating weights must be Python float or NumPy float64;
            integer weights must be exactly representable as float64.
        voter_set (set[str] | frozenset[str]): Set of voters who cast the ballot.
            Defaults to frozenset(). Will be coerced to frozenset.
        scores (Mapping[Candidate, float | int] | Mapping[str, float | int]
            | Mapping[int, float | int] | None): Scores for individual candidates. Defaults to None.
            Values can be input as int or float but will be coerced to float.
            Candidates can be strings, integers, or mix of both.
            Stored internally as a dict[Candidate, float].
            Only retains non-zero scores.

    Attributes:
        ranking (tuple[frozenset[Candidate], ...] | None): Tuple of candidate ranking.
            Entry i of the tuple is a frozenset of candidates ranked in position i.
            Candidates can be strings, integers, or mix of both.
        weight (float | Fraction): Weight assigned to a given ballot.
        voter_set (frozenset[str]): Set of voters who cast the ballot.
        scores (Mapping[Candidate, float | int] | None): Scores for individual candidates.

    Raises:
        TypeError: Only one of ranking or scores can be provided.
        ValueError: Weight is negative, nonfinite, or cannot be represented exactly as float64
            without an explicit Fraction.
    """

    # Memory trick since this is a basic type
    __slots__ = [
        "ranking",
        "weight",
        "voter_set",
        "scores",
        "_frozen",
    ]

    @overload
    def __new__(
        cls,
        *,
        ranking: RankingLike,
        scores: None = None,
        weight: float | int | Fraction = 1.0,
        voter_set: set[str] | frozenset[str] = frozenset(),
        include_zero_score: bool = False,
    ) -> RankBallot: ...

    @overload
    def __new__(
        cls,
        *,
        ranking: None = None,
        scores: ScoresLike,
        weight: float | int | Fraction = 1.0,
        voter_set: set[str] | frozenset[str] = frozenset(),
        include_zero_score: bool = False,
    ) -> ScoreBallot: ...

    @overload
    def __new__(
        cls,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: float | int | Fraction = 1.0,
        voter_set: set[str] | frozenset[str] = frozenset(),
        include_zero_score: bool = False,
    ) -> Self: ...

    def __new__(
        cls,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: float | int | Fraction = 1.0,
        voter_set: set[str] | frozenset[str] = frozenset(),
        include_zero_score: bool = False,
    ) -> Ballot:
        if ranking is not None and scores is not None:
            raise TypeError("Only one of ranking or scores can be provided.")
        elif ranking is not None:
            return object.__new__(RankBallot)
        elif scores is not None:
            return object.__new__(ScoreBallot)

        return super().__new__(cls)

    def __init__(
        self,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: float | int | Fraction = 1.0,
        voter_set: set[str] | frozenset[str] = frozenset(),
    ):
        self.voter_set = frozenset(voter_set) if not isinstance(voter_set, frozenset) else voter_set
        validate_weight(weight)

        self.weight = weight if isinstance(weight, Fraction) else _to_float_weight(weight)
        self._frozen = True

    def __eq__(self, other):
        # Check type
        if not isinstance(other, Ballot):
            return False

        # Check weight
        if self.weight != other.weight:
            return False

        # Check voters
        if self.voter_set != other.voter_set:
            return False

        return True

    def __hash__(self):
        return hash(self.weight) + hash(self.voter_set)

    def __str__(self):
        repr_str = f"Ballot\nWeight: {self.weight}"
        if self.voter_set != frozenset():
            repr_str += f"\nVoter set: {set(self.voter_set)}"
        return repr_str

    __repr__ = __str__

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{type(self).__name__} is frozen")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{type(self).__name__} is frozen")
        object.__delattr__(self, name)


class RankBallot(Ballot):
    """
    Class to handle ballots with rankings. Strips whitespace from candidate names.

    Args:
        ranking (RankingLike): Ranking of candidates, defaults to None.
            RankingLike = Sequence[Candidate | Iterable[Candidate]] | None
            Canidates can be strings, integers, or mix of both.
        weight (int | float | Fraction): Weight of the ballot, defaults to 1.0.
        voter_set (set[str] | frozenset[str]): Voter set of the ballot,
            defaults to frozenset().

    Attributes:
        ranking (Ranking): Ranking of candidates.
            Ranking = tuple[frozenset[Candidate], ...] | None
        weight (float | Fraction): Weight of the ballot.
        voter_set (frozenset[str]): Voter set of the ballot.

    Raises:
        TypeError: Ranking is a sequence of bare or iterable str/int candidates.
        ValueError: Candidate '~' found in ballot ranking.
        ValueError: Weight is negative, nonfinite, or cannot be represented exactly as float64
            without an explicit Fraction.
        UserWarning: '1' and 1 candidates are treated as separate candidates.
    """

    def __init__(
        self,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: int | float | Fraction = 1.0,
        voter_set: set[str] | frozenset[str] = frozenset(),
    ):
        if scores is not None:
            raise TypeError("Only one of ranking or scores can be provided.")
        self.ranking = self._convert_ranking_candidates_to_frozenset_strip_whitespace(ranking)
        self._validate_ranking_candidates(self.ranking)
        super().__init__(weight=weight, voter_set=voter_set)

    def _convert_ranking_candidates_to_frozenset_strip_whitespace(
        self, ranking: RankingLike
    ) -> Ranking:
        if ranking is None:
            return None
        if isinstance(ranking, str):
            raise TypeError(
                f"Received ranking `{ranking}` of type {type(ranking).__name__}. "
                "If you intended this to be a bullet vote, then wrap it in a list."
            )
        if not isinstance(ranking, Sequence):
            raise TypeError(
                "ranking must be a Sequence with a guaranteed order. Received"
                f" {type(ranking).__name__}, which is unordered. Wrap ranking in a list"
                " instead."
            )

        normalized_ranking = []
        for cand_set in ranking:
            if isinstance(cand_set, Candidate):
                normalized_ranking.append(
                    frozenset({cand_set.strip() if isinstance(cand_set, str) else cand_set})
                )
            elif isinstance(cand_set, Iterable):
                normalized_ranking.append(
                    frozenset(c.strip() if isinstance(c, str) else c for c in cand_set)
                )
            else:
                normalized_ranking.append(frozenset({cand_set}))
        return tuple(normalized_ranking)

    def _validate_ranking_candidates(self, ranking: RankingLike):
        if ranking is None:
            return
        candidates = []
        for cand_set in ranking:
            if isinstance(cand_set, (str, int)):
                candidates.append(cand_set)
            elif isinstance(cand_set, Iterable):
                candidates.extend([cand for cand in cand_set])
            else:
                raise TypeError(
                    "Ranking is a sequence of Iterables or bare str/int candidates."
                    f" {cand_set} is invalid."
                )
        _validate_candidate_names(candidates, SourceWithAttribute(self, "ranking"))

    def __eq__(self, other):
        if not isinstance(other, RankBallot):
            return False

        if self.ranking != other.ranking:
            return False

        return super().__eq__(other)

    def __hash__(self):
        return hash(self.ranking) + super().__hash__()

    def __str__(self):
        ranking_str = "RankBallot\n"

        if self.ranking:
            for i, s in enumerate(self.ranking):
                ranking_str += f"{i + 1}.) "
                for c in s:
                    ranking_str += f"{c}, "

                if len(s) > 1:
                    ranking_str += "(tie)"
                ranking_str += "\n"

        ranking_str += f"Weight: {self.weight}"
        if self.voter_set != frozenset():
            ranking_str += f"\nVoter set: {set(self.voter_set)}"
        return ranking_str


class ScoreBallot(Ballot):
    """
    Class to handle ballots with scores. Strips whitespace from candidate names.

    Args:
        scores (ScoresLike): Scores of candidates, defaults to None.
            ScoresLike = Mapping[Candidate, int | float] | Mapping[str, int | float]
            | Mapping[int, int | float] | None
            Candidates can be strings, integers, or mix of both.
        weight (int | float | Fraction): Weight of the ballot, defaults to 1.0.
        voter_set (set[str] | frozenset[str]): Voter set of the ballot,
            defaults to frozenset().

    Attributes:
        scores (dict[Candidate, float] | None): Scores of candidates.
        weight (float | Fraction): Weight of the ballot.
        voter_set (frozenset[str]): Voter set of the ballot.

    Raises:
        ValueError: Candidate '~' found in ballot scores.
        ValueError: Weight is negative, nonfinite, or cannot be represented exactly as float64
            without an explicit Fraction.
        TypeError: Scores must be a mapping of candidates to score values.
        TypeError: Score values must be numeric.
        UserWarning: '1' and 1 candidates are treated as separate candidates.
    """

    def __init__(
        self,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: int | float | Fraction = 1.0,
        voter_set: set[str] | frozenset[str] = frozenset(),
    ):
        if ranking is not None:
            raise TypeError("Only one of ranking or scores can be provided.")
        self._validate_scores_candidates(scores)
        self.scores = self._convert_scores_to_float_strip_whitespace(scores)

        super().__init__(weight=weight, voter_set=voter_set)

    def _convert_scores_to_float_strip_whitespace(
        self, scores: ScoresLike
    ) -> dict[Candidate, float] | None:
        if scores is None:
            return None
        return {
            c.strip() if isinstance(c, str) else c: float(s) for c, s in scores.items() if s != 0
        }

    def _validate_scores_candidates(self, scores: ScoresLike):
        if scores is not None:
            if not isinstance(scores, Mapping):
                raise TypeError(
                    "Scores must be a mapping of candidates to score values. Received"
                    f" {type(scores).__name__}."
                )
            if any(not isinstance(s, Real) for s in scores.values()):
                raise TypeError("Score values must be numeric.")

            _validate_candidate_names(list(scores.keys()), SourceWithAttribute(self, "scores"))

    def __eq__(self, other):
        if not isinstance(other, ScoreBallot):
            return False
        if self.scores != other.scores:
            return False
        return super().__eq__(other)

    def __hash__(self):
        return (
            hash(frozenset(self.scores.items()) if self.scores is not None else self.scores)
            + super().__hash__()
        )

    def __str__(self):
        score_str = "ScoreBallot\n"
        if self.scores:
            for c, score in self.scores.items():
                score_str += f"{c}: {score:.2f}\n"

        score_str += f"Weight: {self.weight}"
        if self.voter_set != frozenset():
            score_str += f"\nVoter set: {set(self.voter_set)}"
        return score_str

    __repr__ = __str__
