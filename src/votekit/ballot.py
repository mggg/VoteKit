from pydantic import BaseModel
from typing import Optional
from fractions import Fraction


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
