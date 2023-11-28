from fractions import Fraction
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from dataclasses import field
from typing import Optional


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Ballot:
    """
    Ballot class, contains ranking and assigned weight.

    **Attributes**

    `id`
    :   optional ballot id.

    `ranking`
    :   list of candidate ranking. Entry i of the list is a set of candidates ranked in position i.

    `weight`
    :   weight assigned to a given a ballot. Defaults to 1.

    `voter_set`
    :   optional set of voters who cast a given a ballot.
    """

    ranking: list[set] = field(default_factory=list)
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
        return hash(str(self.ranking))

    def __str__(self):
        weight_str = f"Weight: {self.weight}\n"
        ranking_str = "Ballot\n"

        if self.ranking:
            for i, s in enumerate(self.ranking):
                ranking_str += f"{i+1}.) "
                for c in s:
                    ranking_str += f"{c}, "

                if len(s) > 1:
                    ranking_str += "(tie)"
                ranking_str += "\n"
        else:
            ranking_str += "No Ranking\n"

        return ranking_str + weight_str

    __repr__ = __str__
