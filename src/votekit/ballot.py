from __future__ import annotations

from fractions import Fraction
from numbers import Real
from typing import Iterable, Mapping, Optional, Sequence, Union, overload

from votekit.types import Candidate, Ranking, RankingLike, ScoresLike
from votekit.utils import _validate_candidate_names


class Ballot:
    """
    Ballot parent class, contains voter set and assigned weight.

    Args:
        ranking (Optional[Sequence[Candidate | Iterable[Candidate]]]): Candidate ranking.
            Entry i of the sequence is a candidate or iterable of candidates ranked in position i.
            Candidates can be strings, integers, or mix of both.
            Defaults to None. Will be coerced to tuple[frozenset[Candidate], ...].
        weight (Union[float, int, Fraction]): Weight assigned to a given ballot. Defaults to 1.0.
            Rational weights are preserved for rank ballots; other weights are coerced to float.
        voter_set (Union[set[str], frozenset[str]]): Set of voters who cast the ballot.
            Defaults to frozenset(). Will be coerced to frozenset.
        scores (Optional[Mapping[Candidate, float | int] | Mapping[str, float | int]
            | Mapping[int, float | int]]): Scores for individual candidates. Defaults to None.
            Values can be input as int or float but will be coerced to float.
            Candidates can be strings, integers, or mix of both.
            Stored internally as a dict[Candidate, float].
            Only retains non-zero scores.

    Attributes:
        ranking (Optional[tuple[frozenset[Candidate], ...]]): Tuple of candidate ranking.
            Entry i of the tuple is a frozenset of candidates ranked in position i.
            Candidates can be strings, integers, or mix of both.
        weight (float | Fraction): Weight assigned to a given ballot.
        voter_set (frozenset[str]): Set of voters who cast the ballot.
        scores (Optional[Mapping[Candidate, float | int]): Scores for individual candidates.

    Raises:
        TypeError: Only one of ranking or scores can be provided.
        ValueError: Ballot weight cannot be negative.
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
        weight: Union[float, int, Fraction] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
        include_zero_score: None = None,
    ) -> RankBallot: ...

    @overload
    def __new__(
        cls,
        *,
        ranking: None = None,
        scores: ScoresLike,
        weight: Union[float, int] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
        include_zero_score: bool = False,
    ) -> ScoreBallot: ...

    @overload
    def __new__(
        cls,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: Union[float, int] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
        include_zero_score: bool = False,
    ) -> Ballot: ...

    def __new__(
        cls,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: Union[float, int, Fraction] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
        include_zero_score: bool = False,
    ):
        if ranking is not None and scores is not None:
            raise TypeError("Only one of ranking or scores can be provided.")
        elif ranking is not None:
            return super().__new__(RankBallot)
        elif scores is not None:
            return super().__new__(ScoreBallot)

        return super().__new__(cls)

    def __init__(
        self,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: Union[float, int, Fraction] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
    ):
        self.voter_set = frozenset(voter_set) if not isinstance(voter_set, frozenset) else voter_set

        if weight < 0:
            raise ValueError("Ballot weight cannot be negative.")

        self.weight = (
            weight
            if isinstance(self, RankBallot) and isinstance(weight, Fraction)
            else float(weight)
        )
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
        weight (Union[int, float, Fraction]): Weight of the ballot, defaults to 1.0.
        voter_set (Union[set[str], frozenset[str]]): Voter set of the ballot,
            defaults to frozenset().

    Attributes:
        ranking (Ranking): Ranking of candidates.
            Ranking = tuple[frozenset[Candidate], ...] | None
        weight (float | Fraction): Weight of the ballot.
        voter_set (frozenset[str]): Voter set of the ballot.

    Raises:
        TypeError: Ranking is a sequence of bare or iterable str/int candidates.
        ValueError: Candidate '~' found in ballot ranking.
        ValueError: Ballot weight cannot be negative.
        UserWarning: '1' and 1 candidates are treated as separate candidates.
    """

    def __init__(
        self,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: Union[int, float, Fraction] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
    ):
        if scores is not None:
            raise TypeError("Only one of ranking or scores can be provided.")
        self._validate_ranking_candidates(ranking)
        ranking = self._convert_ranking_candidates_to_frozenset_strip_whitespace(ranking)
        self.ranking = ranking
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
            else:
                normalized_ranking.append(
                    frozenset(c.strip() if isinstance(c, str) else c for c in cand_set)
                )
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
        _validate_candidate_names(candidates, self, "ranking")

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
        weight (Union[int, float]): Weight of the ballot, defaults to 1.0.
        voter_set (Union[set[str], frozenset[str]]): Voter set of the ballot,
            defaults to frozenset().

    Attributes:
        scores (Optional[dict[Candidate, float]]): Scores of candidates.
        weight (float): Weight of the ballot.
        voter_set (frozenset[str]): Voter set of the ballot.

    Raises:
        ValueError: Candidate '~' found in ballot scores.
        ValueError: Ballot weight cannot be negative.
        TypeError: Scores must be a mapping of candidates to score values.
        TypeError: Score values must be numeric.
        UserWarning: '1' and 1 candidates are treated as separate candidates.
    """

    def __init__(
        self,
        *,
        ranking: RankingLike = None,
        scores: ScoresLike = None,
        weight: Union[int, float] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
    ):
        if ranking is not None:
            raise TypeError("Only one of ranking or scores can be provided.")
        self._validate_scores_candidates(scores)
        self.scores = self._convert_scores_to_float_strip_whitespace(scores)

        super().__init__(weight=weight, voter_set=voter_set)

    def _convert_scores_to_float_strip_whitespace(
        self, scores: ScoresLike
    ) -> Optional[dict[Candidate, float]]:
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

            _validate_candidate_names(list(scores.keys()), self, "scores")

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
