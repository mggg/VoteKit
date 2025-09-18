from typing import Optional, Union, TypeAlias, Iterable, Sequence
from numbers import Real


Ranking: TypeAlias = Optional[tuple[frozenset[str], ...]]
RankingLike: TypeAlias = Optional[Sequence[Iterable[str]]]


class Ballot:
    """
    Ballot parent class, contains voter set and assigned weight.

    Args:
        ranking (Optional[Sequence[Iterable[str]]]): Candidate ranking. Entry i of the
            sequence is an iterable of candidates ranked in position i. Defaults to None.
            Will be coerced to tuple[frozenset[str], ...].
        weight (Union[float, int]): Weight assigned to a given ballot. Defaults to 1.0
            Can be input as int or float, and will be coerced to float.
        voter_set (Union[set[str], frozenset[str]]): Set of voters who cast the ballot.
            Defaults to frozenset(). Will be coerced to frozenset.
        scores (Optional[dict[str, Union[int, float]]): Scores for individual candidates.
            Defaults to None. Values can be input as int or float but will be coerced to float.
            Only retains non-zero scores.

    Attributes:
        ranking (Optional[tuple[frozenset[str], ...]]): Tuple of candidate ranking.
            Entry i of the tuple is a
            frozenset of candidates ranked in position i.
        weight (float): Weight assigned to a given ballot.
        voter_set (frozenset[str]): Set of voters who cast the ballot.
        scores (Optional[dict[str, float]]): Scores for individual candidates.

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

    def __new__(
        cls,
        *,
        ranking: Optional[Sequence[Iterable[str]]] = None,
        scores: Optional[dict[str, Union[int, float]]] = None,
        weight: Union[float, int] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
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
        ranking: Optional[Sequence[Iterable[str]]] = None,
        scores: Optional[dict[str, Union[int, float]]] = None,
        weight: Union[float, int] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
    ):

        self.voter_set = (
            frozenset(voter_set) if not isinstance(voter_set, frozenset) else voter_set
        )

        if weight < 0:
            raise ValueError("Ballot weight cannot be negative.")

        # Silently promote weight to float
        self.weight = float(weight)
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
        weight (Union[int, float]): Weight of the ballot, defaults to 1.0.
        voter_set (Union[set[str], frozenset[str]]): Voter set of the ballot,
            defaults to frozenset().

    Attributes:
        ranking (RankingLike): Ranking of candidates.
        weight (float): Weight of the ballot.
        voter_set (frozenset[str]): Voter set of the ballot.

    Raises:
        ValueError: Candidate '~' found in ballot ranking.
        ValueError: Ballot weight cannot be negative.
    """

    def __init__(
        self,
        *,
        ranking: RankingLike = None,
        weight: Union[int, float] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
    ):
        self._validate_ranking_candidates(ranking)
        self.ranking = self._strip_whitespace_ranking_candidates(ranking)
        super().__init__(weight=weight, voter_set=voter_set)

    def _validate_ranking_candidates(self, ranking: RankingLike):
        if ranking is None:
            return
        if any(c == "~" for cand_set in ranking for c in cand_set):
            raise ValueError(
                f"Candidate '~' found in ballot ranking {ranking}."
                " '~' is a reserved character and cannot be used for"
                " candidate names."
            )

    def _strip_whitespace_ranking_candidates(self, ranking: RankingLike) -> Ranking:
        if ranking is None:
            return None

        return tuple([frozenset(c.strip() for c in cand_set) for cand_set in ranking])

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
                ranking_str += f"{i+1}.) "
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
        scores (Optional[dict[str, Union[int, float]]]): Scores of candidates, defaults to None.
        weight (Union[int, float]): Weight of the ballot, defaults to 1.0.
        voter_set (Union[set[str], frozenset[str]]): Voter set of the ballot,
            defaults to frozenset().

    Attributes:
        scores (Optional[dict[str, float]]): Scores of candidates.
        weight (float): Weight of the ballot.
        voter_set (frozenset[str]): Voter set of the ballot.

    Raises:
        ValueError: Candidate '~' found in ballot scores.
        ValueError: Ballot weight cannot be negative.
        TypeError: Score values must be numeric.
    """

    def __init__(
        self,
        *,
        scores: Optional[dict[str, Union[int, float]]] = None,
        weight: Union[int, float] = 1.0,
        voter_set: Union[set[str], frozenset[str]] = frozenset(),
    ):
        self._validate_scores_candidates(scores)
        self.scores = self._convert_scores_to_float_strip_whitespace(scores)

        super().__init__(weight=weight, voter_set=voter_set)

    def _validate_scores_candidates(
        self, scores: Optional[dict[str, Union[int, float]]]
    ):
        if scores is not None:
            if "~" in scores:
                raise ValueError(
                    f"Candidate '~' found in ballot scores {list(scores.keys())}."
                    " '~' is a reserved character and cannot be used for"
                    " candidate names."
                )

    def _convert_scores_to_float_strip_whitespace(
        self, scores: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        if scores is None:
            return None

        if any(not isinstance(s, Real) for s in scores.values()):
            raise TypeError("Score values must be numeric.")

        return {c.strip(): float(s) for c, s in scores.items() if s != 0}

    def __eq__(self, other):
        if not isinstance(other, ScoreBallot):
            return False
        if self.scores != other.scores:
            return False
        return super().__eq__(other)

    def __hash__(self):

        return (
            hash(
                tuple(sorted((c, s) for c, s in self.scores.items()))
                if self.scores is not None
                else self.scores
            )
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
