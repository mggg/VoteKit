from fractions import Fraction
from pydantic import BaseModel
from typing import Optional

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
    

class PointBallot(BaseModel):
    """
    Ballot class for voting methods that rely on point systems, like cumulative, Borda, approval.

    **Attributes**

    `id`
    :   optionally assigned ballot id

    `points`
    :  list of candidates chosen with multiplicity or 
            dictionary whose keys are candidates and values are points given to candidates.

    `weight`
    :   weight assigned to a given a ballot

    `voters`
    :   set of voters who cast a given a ballot
    """

    id: Optional[str] = None
    weight: Fraction = Fraction(1,1)
    voters: Optional[set[str]] = None
    points: dict = {}

    def __init__(self, id = None, weight = Fraction(1,1), voters = None, points = {}):

        # convert list entry to dictionary
        if isinstance(points, list):
            di = {}
            for candidate in points:
                if candidate in di.keys():
                    di[candidate]+= 1
                else:
                    di[candidate] = 1 
            points = di

        super().__init__(id = id, weight = weight, voters = voters, points = points)

        


    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other):
        # Check type
        if not isinstance(other, PointBallot):
            return False

        # Check id
        if self.id is not None:
            if self.id != other.id:
                return False

        # Check points
        if self.points != other.points:
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
        return hash(str(self.points))
    
    def __str__(self):
        return f"{self.points} with weight {self.weight}"
    
    __repr__ = __str__