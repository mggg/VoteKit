from unnamed_rcv_thing.ballot import Ballot
from typing import List
from pydantic import BaseModel

# from functools import cache


class PreferenceProfile(BaseModel):
    """
    ballots (list of allots): ballots from an election
    candidates (list): list of candidates, can be user defined
    """

    ballots: List[Ballot]
    # candidates: Optional[list] = None

    # @validator('candidates')
    # def cands_must_be_unique(cls, cands):
    #     if not len(set(cands)) == len(cands):
    #         raise ValueError('all candidates must be unique')
    #     return cands

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
        unique_cands = set()
        for ballot in self.ballots:
            unique_cands.update(ballot.ranking)

        return list(unique_cands)

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
