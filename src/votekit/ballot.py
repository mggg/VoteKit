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

    `voters`
    :   optional list of voters who cast a given a ballot.
    """

    ranking: list[set] = field(default_factory=list)
    weight: Fraction = Fraction(1, 1)
    voters: Optional[set[str]] = None
    id: Optional[str] = None

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
        if self.voters is not None:
            if self.voters != other.voters:
                return False

        return True

    def __hash__(self):
        return hash(str(self.ranking))
