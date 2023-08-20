from pydantic import BaseModel
from typing import Optional
from fractions import Fraction


class Ballot(BaseModel):
    """
    id (optional string): assigned ballot id
    ranking (list): list of candidate ranking
    weight (float/fraction): weight assigned to a given a ballot
    voters (optional list): list of voters who cast a given a ballot
    """

    id: Optional[str] = None
    ranking: list[set]
    weight: Fraction
    voters: Optional[set[str]] = None

    class Config:
        arbitrary_types_allowed = True

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
