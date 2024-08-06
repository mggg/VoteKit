from fractions import Fraction
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, field_validator
from typing import Optional, Union


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Ballot:
    """
    Ballot class, contains ranking and assigned weight.

    Args:
        ranking (tuple[frozenset, ...], optional): Tuple of candidate ranking. Entry i of the tuple
            is a frozenset of candidates ranked in position i. Defaults to None.
        weight (Fraction, optional): Weight assigned to a given ballot. Defaults to 1.
            Can be input as int, float, or Fraction but will be converted to Fraction.
        voter_set (set[str], optional): Set of voters who cast the ballot. Defaults to None.
        id (str, optional): Ballot ID. Defaults to None.
        scores (dict[str, Fraction], optional): Scores for individual candidates. Defaults to None.
            Values can be input as int, float, or Fraction but will be converted to Fraction.
            Only retains non-zero scores.

    Attributes:
        ranking (tuple[frozenset, ...]): Tuple of candidate ranking. Entry i of the tuple is a
            frozenset of candidates ranked in position i.
        weight (Fraction): Weight assigned to a given ballot. Defaults to 1.
        voter_set (set[str], optional): Set of voters who cast the ballot. Defaults to None.
        id (str, optional): Ballot ID. Defaults to None.
        scores (dict[str, Fraction], optional): Scores for individual candidates. Defaults to None.
    """

    ranking: Optional[tuple[frozenset, ...]] = None
    weight: Fraction = Fraction(1, 1)
    voter_set: Optional[set[str]] = None
    id: Optional[str] = None
    scores: Optional[dict[str, Fraction]] = None

    @field_validator("weight", mode="before")
    @classmethod
    def convert_weight_to_fraction(cls, weight: Union[float, Fraction]) -> Fraction:
        if not isinstance(weight, Fraction):
            weight = Fraction(weight).limit_denominator()
        return weight

    @field_validator("scores", mode="before")
    @classmethod
    def convert_scores_to_fraction(
        cls, scores: Optional[dict[str, Union[float, Fraction]]]
    ) -> Optional[dict[str, Fraction]]:
        if scores:
            if any(
                not (
                    isinstance(s, float)
                    or isinstance(s, Fraction)
                    or isinstance(s, int)
                )
                for s in scores.values()
            ):
                raise TypeError("Score values must be numeric.")

            return {
                c: Fraction(s).limit_denominator() for c, s in scores.items() if s != 0
            }
        else:
            return None

    def __eq__(self, other):
        # Check type
        if not isinstance(other, Ballot):
            return False

        # Check id
        if self.id is not None:
            if self.id != other.id:
                return False

        # Check ranking
        if self.ranking != other.ranking:
            return False

        # Check weight
        if self.weight != other.weight:
            return False

        # Check voters
        if self.voter_set is not None:
            if self.voter_set != other.voter_set:
                return False

        # Check scores
        if self.scores is not None:
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
