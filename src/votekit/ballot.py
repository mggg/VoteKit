from fractions import Fraction
from pydantic import BaseModel
from typing import Optional, Union
from multiset import Multiset

class Ballot(BaseModel):
    """
    Ballot class, contains ranking and assigned weight

    **Attributes**

    `id`
    :   optionally assigned ballot id

    `ranking`
    :   list of candidate ranking

    `weight`
    :   weight assigned to a given a ballot

    `voters`
    :   set of voters who cast a given a ballot
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

    def __hash__(self):
        return hash(str(self.ranking))
    

class CumulativeBallot(BaseModel):
    """
    Ballot class for cumulative voting, contains ranking and assigned weight

    **Attributes**

    `id`
    :   optionally assigned ballot id

    `multi_votes`
    :   list of candidates for whom votes were cast, with multiplicity. Will be converted to
    Multiset type. Or you can put in a Multiset type.
    ["A", "A", "B"]

    `weight`
    :   weight assigned to a given a ballot

    `voters`
    :   set of voters who cast a given a ballot
    """

    id: Optional[str] = None
    weight: Fraction = Fraction(1,1)
    voters: Optional[set[str]] = None
    multi_votes: Union[list[str], Multiset] = []

    def __init__(self, id = None, weight = Fraction(1,1), 
                 voters=None, multi_votes = []):
        
        super().__init__(id = id, weight = weight, 
                         voters = voters, multi_votes = multi_votes)

        if isinstance(self.multi_votes, list):
            self.multi_votes = Multiset(self.multi_votes)

    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other):
        # Check type
        if not isinstance(other, CumulativeBallot):
            return False

        # Check id
        if self.id is not None:
            if self.id != other.id:
                return False

        # Check multi_votes
        if self.multi_votes != other.multi_votes:
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
        return hash(str(self.multi_votes))
    
    def __str__(self):
        return f"{self.multi_votes} with weight {self.weight}"
    
    __repr__ = __str__