from fractions import Fraction
from pydantic import BaseModel
from typing import Optional


class Ballot(BaseModel):
    """
    Data structure to represent a possible cast ballot.
    :param id: optional :class:`str` assigned ballot id
    :param ranking: :class:`list` of candidate ranking
    :param weight: :class:`float`/:class:`fraction` weight assigned to a given a ballot
    :param voters: optional :class:`list` of voters who cast a given a ballot
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
