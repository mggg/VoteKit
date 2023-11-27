from fractions import Fraction
from pydantic import BaseModel
from typing import Optional


class Ballot(BaseModel):
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

    id: Optional[str] = None
    ranking: list[set]
    weight: Fraction = Fraction(1,1)
    voter_set: Optional[set[str]] = None

    def __init__(self, id = None, ranking = [], weight = Fraction(1,1), voter_set = None):

        if not isinstance(weight, Fraction):
            # limit_denominator recovers rational numbers represented as floats
            weight = Fraction(weight).limit_denominator()

        super().__init__(id = id, ranking = ranking, weight = weight, voter_set = voter_set)

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

                if len(s)>1:
                    ranking_str+= "(tie)"
                ranking_str+= "\n"
        else:
            ranking_str += "No Ranking\n"
        
        return ranking_str + weight_str

    __repr__ = __str__
