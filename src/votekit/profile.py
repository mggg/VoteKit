from .ballot import Ballot
from typing import Optional
from pydantic import BaseModel, validator
from fractions import Fraction


class PreferenceProfile(BaseModel):
    """
    ballots (list of ballots): ballots from an election
    candidates (list): list of candidates, can be user defined
    """

    ballots: Optional[list] = None
    candidates: Optional[list] = None

    @validator("candidates")
    def cands_must_be_unique(cls, candidates: list) -> list:
        if not len(set(candidates)) == len(candidates):
            raise ValueError("all candidates must be unique")
        return candidates

    def get_ballots(self) -> list[Ballot]:
        """
        Returns list of ballots
        """
        return self.ballots

    # @cache
    def get_candidates(self) -> list:
        """
        Returns list of unique candidates
        """
        if self.candidates is not None:
            return self.candidates

        unique_cands: set = set()
        for ballot in self.ballots:
            unique_cands.update(*ballot.ranking)

        return list(unique_cands)

    # can also cache
    def num_ballots(self):
        """
        Assumes weights correspond to number of ballots given to a ranking
        """
        num_ballots = 0
        for ballot in self.ballots:
            num_ballots += ballot.weight

        return num_ballots

    def to_dict(self) -> dict:
        """
        Converts ballots to dictionary with keys (ranking) and values
        the corresponding total weights
        """
        di: dict = {}
        for ballot in self.ballots:
            if str(ballot.ranking) not in di.keys():
                di[str(ballot.ranking)] = Fraction(0)
            di[str(ballot.ranking)] += ballot.weight

        return di

    class Config:
        arbitrary_types_allowed = True

    def condense_ballots(self):
        class_vector = []
        seen_rankings = []
        for ballot in self.ballots:
            if ballot.ranking not in seen_rankings:
                seen_rankings.append(ballot.ranking)
            class_vector.append(seen_rankings.index(ballot.ranking))

        new_ballot_list = []
        for i, ranking in enumerate(seen_rankings):
            total_weight = 0
            for j in range(len(class_vector)):
                if class_vector[j] == i:
                    total_weight += self.ballots[j].weight
            new_ballot_list.append(
                Ballot(ranking=ranking, weight=Fraction(total_weight))
            )
        self.ballots = new_ballot_list

    def __eq__(self, other):
        if not isinstance(other, PreferenceProfile):
            return False
        self.condense_ballots()
        other.condense_ballots()
        for b in self.ballots:
            if b not in other.ballots:
                return False
        for b in self.ballots:
            if b not in other.ballots:
                return False
        return True
