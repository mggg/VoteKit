from typing_extensions import override
from unnamed_rcv_thing.ballot import Ballot
from typing import List, Optional, Set
from pydantic import BaseModel, validator
import numpy as np

# from functools import cache


class PreferenceProfile(BaseModel):
    """
    ballots (list of allots): ballots from an election
    candidates (list): list of candidates, can be user defined
    """

    #TODO: ask if ballots should be a set, define an equality function
    ballots: Set[Ballot]
    candidates: Optional[list] = None

    @validator("candidates")
    def cands_must_be_unique(cls, cands):
        if not len(set(cands)) == len(cands):
            raise ValueError("all candidates must be unique")
        return cands

    def get_ballots(self):
        """
        Returns list of ballots
        """
        return self.ballots

    # @cache
    def get_candidates(self):
        """
        Returns list of unique candidates
        """
        if self.candidates is not None:
            return self.candidates

        unique_cands = set()
        for ballot in self.ballots:
            unique_cands.update(ballot.ranking)

        return list(unique_cands)

    @override
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, PreferenceProfile) and self.ballots == __value.ballots

    # class Config:
    #     arbitrary_types_allowed = True

    # def __init__(self, ballots, candidates):
    #     """
    #     Args:
    #         ballots (list of Ballot): a list of ballots in the election
    #         candidates (list of Candidates): a list of candidates in the election
    #     """
    #     self.id = uuid.uuid4()
    #     self.ballots = ballots
    #     self.candidates = candidates
    #     self.ballot_weights = [b.score for b in ballots]

if __name__ == '__main__':
    ballots1 = {Ballot(id=None, ranking=['c', np.nan, np.nan], weight=1.0, voters=['a'])}
    ballots2 = {Ballot(id=None, ranking=['c', np.nan, np.nan], weight=1.0, voters=['a'])}
    prof1 = PreferenceProfile(ballots=ballots1)
    prof2 = PreferenceProfile(ballots=ballots2)
    print(prof1 == prof2)