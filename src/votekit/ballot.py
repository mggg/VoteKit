from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, field_validator
from typing import Optional
from dataclasses import field


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Ballot:
    """
    Ballot class, contains ranking and assigned weight. Note that we trim trailing or
    leading whitespace from candidate names.

    Args:
        ranking (tuple[frozenset[str], ...], optional): Tuple of candidate ranking. Entry i of the
            tuple is a frozenset of candidates ranked in position i. Defaults to None.
        weight (float, optional): Weight assigned to a given ballot. Defaults to 1.
            Can be input as int or float.
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

    ranking: Optional[tuple[frozenset[str], ...]] = None
    weight: float = 1.0
    voter_set: set[str] = field(default_factory=set)
    scores: Optional[dict[str, float]] = None

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Ballot weight must cannot be negative.")

        # Silently promote weight to float
        object.__setattr__(self, "weight", float(self.weight))

    @field_validator("ranking", mode="before")
    @classmethod
    def validate_ranking_candidates(
        cls, ranking: Optional[tuple[frozenset[str], ...]]
    ) -> Optional[tuple[frozenset[str], ...]]:
        if ranking is not None:
            if any(c == "~" for cand_set in ranking for c in cand_set):
                raise ValueError(
                    f"Candidate '~' found in ballot ranking {ranking}."
                    " '~' is a reserved character and cannot be used for"
                    " candidate names."
                )

        return ranking

    @field_validator("scores", mode="before")
    @classmethod
    def validate_scores_candidates(
        cls, scores: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        if scores is not None:
            if "~" in scores:
                raise ValueError(
                    f"Candidate '~' found in ballot scores {list(scores.keys())}."
                    " '~' is a reserved character and cannot be used for"
                    " candidate names."
                )

        return scores

    @field_validator("ranking", mode="before")
    @classmethod
    def strip_whitespace_ranking_candidates(
        cls, ranking: Optional[tuple[frozenset[str], ...]]
    ) -> Optional[tuple[frozenset[str], ...]]:
        if ranking is None:
            return None

        return tuple([frozenset(c.strip() for c in cand_set) for cand_set in ranking])

    @field_validator("scores", mode="before")
    @classmethod
    def convert_scores_to_fraction_strip_whitespace(
        cls, scores: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        if scores:
            if any(
                not (isinstance(s, float) or isinstance(s, int))
                for s in scores.values()
            ):
                raise TypeError("Score values must be numeric.")

            return {c.strip(): s for c, s in scores.items() if s != 0}
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
