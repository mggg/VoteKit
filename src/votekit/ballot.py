from fractions import Fraction
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from dataclasses import field
from typing import Optional


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Ballot:
    """
    Ballot class, contains ranking and assigned weight.

    Args:
        ranking (tuple[frozenset, ...]): Tuple of candidate ranking. Entry i of the tuple is a
            frozenset of candidates ranked in position i.
        weight (Fraction): Weight assigned to a given ballot. Defaults to 1.
        voter_set (set[str], optional): Set of voters who cast the ballot. Defaults to None.
        id (str, optional): Ballot ID. Defaults to None.

    Attributes:
        ranking (tuple[frozenset, ...]): Tuple of candidate ranking. Entry i of the tuple is a
            frozenset of candidates ranked in position i.
        weight (Fraction): Weight assigned to a given ballot. Defaults to 1.
        voter_set (set[str], optional): Set of voters who cast the ballot. Defaults to None.
        id (str, optional): Ballot ID. Defaults to None.
    """

    ranking: tuple[frozenset, ...] = field(default_factory=tuple)
    weight: Fraction = Fraction(1, 1)
    voter_set: Optional[set[str]] = None
    id: Optional[str] = None

    def __post_init__(self):
        # converts weight to a Fraction if an integer or float
        if not isinstance(self.weight, Fraction):
            object.__setattr__(
                self, "weight", Fraction(self.weight).limit_denominator()
            )

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

        return True

    def __hash__(self):
        return hash(self.ranking)

    def __str__(self):
        weight_str = f"Weight: {self.weight}\n"
        ranking_str = "Ballot\n"

        if self.ranking:
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
            ranking_str += "Empty\n"

        return ranking_str + weight_str

    __repr__ = __str__
