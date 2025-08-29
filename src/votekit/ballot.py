from typing import Optional, Union, TypeAlias, Iterable, Sequence
from numbers import Real


Ranking: TypeAlias = Optional[tuple[frozenset[str], ...]]
RankingLike: TypeAlias = Optional[Sequence[Iterable[str]]]


class Ballot:
    """
    Ballot class, contains ranking and assigned weight. Note that we trim trailing or
    leading whitespace from candidate names.

    Args:
        ranking (tuple[frozenset[str], ...], optional): Tuple of candidate ranking. Entry i of the
            tuple is a frozenset of candidates ranked in position i. Defaults to None.
        weight (Union[float, int): Weight assigned to a given ballot. Defaults to 1.0
            Can be input as int or float, and will be coerced to float.
        voter_set (set[str], optional): Set of voters who cast the ballot. Defaults to None.
        scores (dict[str, float], optional): Scores for individual candidates. Defaults to None.
            Values can be input as int or float.
            Only retains non-zero scores.

    Attributes:
        ranking (tuple[frozenset[str], ...]): Tuple of candidate ranking. Entry i of the tuple is a
            frozenset of candidates ranked in position i.
        weight (float): Weight assigned to a given ballot. Defaults to 1.
        voter_set (set[str], optional): Set of voters who cast the ballot. Defaults to None.
        scores (dict[str, float], optional): Scores for individual candidates. Defaults to None.
    """

    # Memory trick since this is a basic type
    __slots__ = [
        "ranking",
        "weight",
        "voter_set",
        "scores",
        "_frozen",
    ]

    def __init__(
        self,
        ranking: RankingLike = None,
        weight: Union[float, int] = 1.0,
        voter_set: Optional[Union[set, frozenset]] = None,
        scores: Optional[dict[str, Union[float, int]]] = None,
    ):
        self.ranking = self.strip_whitespace_ranking_candidates(
            self.validate_ranking_candidates(ranking)
        )
        self.voter_set = frozenset() if voter_set is None else frozenset(voter_set)
        self.scores = self.convert_scores_to_float_strip_whitespace(
            self.validate_scores_candidates(scores)
        )

        if weight < 0:
            raise ValueError("Ballot weight cannot be negative.")

        # Silently promote weight to float
        self.weight = float(weight)
        self._frozen = True

    def validate_ranking_candidates(self, ranking: RankingLike) -> RankingLike:
        if ranking is None:
            return None

        if any(c == "~" for cand_set in ranking for c in cand_set):
            raise ValueError(
                f"Candidate '~' found in ballot ranking {ranking}."
                " '~' is a reserved character and cannot be used for"
                " candidate names."
            )

        return ranking

    def strip_whitespace_ranking_candidates(self, ranking: RankingLike) -> Ranking:
        if ranking is None:
            return None

        return tuple([frozenset(c.strip() for c in cand_set) for cand_set in ranking])

    def validate_scores_candidates(
        self, scores: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        if scores is not None:
            if "~" in scores:
                raise ValueError(
                    f"Candidate '~' found in ballot scores {list(scores.keys())}."
                    " '~' is a reserved character and cannot be used for"
                    " candidate names."
                )

        return scores

    def convert_scores_to_float_strip_whitespace(
        self, scores: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        if scores:
            if any(not isinstance(s, Real) for s in scores.values()):
                raise TypeError("Score values must be numeric.")

            return {c.strip(): float(s) for c, s in scores.items() if s != 0}
        else:
            return None

    def __eq__(self, other):
        # Check type
        if not isinstance(other, Ballot):
            return False

        # Check ranking
        if self.ranking != other.ranking:
            return False

        # Check weight
        if self.weight != other.weight:
            return False

        # Check voters
        if self.voter_set != other.voter_set:
            return False

        # Check scores
        if self.scores != other.scores:
            return False
        return True

    def __hash__(self):
        return hash(self.ranking)

    def __str__(self):
        weight_str = f"Weight: {self.weight}"

        if self.ranking:
            ranking_str = "Ranking\n"
            for i, s in enumerate(self.ranking):
                # display number and candidates
                ranking_str += f"{i+1}.) "
                for c in s:
                    ranking_str += f"{c}, "

                # if tie
                if len(s) > 1:
                    ranking_str += "(tie)"
                ranking_str += "\n"
        else:
            ranking_str = ""

        if self.scores:
            score_str = "Scores\n"
            for c, score in self.scores.items():
                score_str += f"{c}: {float(score):.2f}\n"
        else:
            score_str = ""
        return ranking_str + score_str + weight_str

    __repr__ = __str__

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{type(self).__name__} is frozen")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{type(self).__name__} is frozen")
        object.__delattr__(self, name)
