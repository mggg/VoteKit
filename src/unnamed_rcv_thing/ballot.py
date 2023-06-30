from typing_extensions import override
from pydantic import BaseModel
from typing import Optional
import numpy as np


class Ballot(BaseModel):
    """
    id (optional string): assigned ballot id
    ranking (list): list of candidate ranking
    weight (float/fraction): weight assigned to a given a ballot
    voters (optional list): list of voters who cast a given a ballot
    """

    id: Optional[str] = None
    ranking: list
    weight: float
    voters: Optional[list[str]] = None

    @override
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Ballot):
            return np.array_equal(self.ranking, __value.ranking)
        return False
    
    def __hash__(self):
        return hash(tuple(self.ranking))
    
    # def __init__(self, ranking, weight, id):
    #     """
    #     Args:
    #         voters (list of Voter): _description_
    #         candidate_ranking (list of Candidate): _description_
    #         score (int, optional): assigned weight to the ballot.

    #     self.id = id
    #     self.weight = weight
    #     self.ranking = ranking
    #     # self.voters = voters

    # self.is_spoiled = False

    # if score:
    #     self.score = score
    # else:
    #     self.score = len(voters)

    # TODO: define equality for Ballot and iterable


# Pydantic format

if __name__ == '__main__':
    ballots1 = {Ballot(id=None, ranking=['c', np.nan, np.nan], weight=1.0, voters=['a'])}
    ballots2 = {Ballot(id=None, ranking=['c', np.nan, np.nan], weight=1.0, voters=['a'])}
    print(ballots1 == ballots2)
