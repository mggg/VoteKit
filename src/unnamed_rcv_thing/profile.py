from .ballot import Ballot
from typing import Optional
from pydantic import BaseModel, validator
# from functools import cache


class PreferenceProfile(BaseModel):
    """
    ballots (list of allots): ballots from an election
    candidates (list): list of candidates, can be user defined
    """

    ballots: list[Ballot]
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
