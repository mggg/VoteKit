from pydantic import BaseModel
from profile import PreferenceProfile
from typing import Optional

#1. "--> int" arrow thing for each function
#2. is add_winners_and_losers necessary? --A: probabably not
#3. do we want anything more?]
#4. RCV cruncher: inspiration for funcitons--> statistics functions/things
# Example of immutable data model for results

class Outcome():
    """
    curr_round (an Int): current round number
    elected (a set of Candidate): candidates who pass a certain threshold to win an election
    eliminated (a set of Candidate): candidates who were eliminated (lost in the election)
    remaining (a set of Candidate): candidates who are still in the running
    rankings (a list of a set of Candidate): ranking of candidates with sets representing ties
    ballots (a list of Dict): a list of ballot types (schedule preferences) and their frequency of times
    winners_votes (a Dict of candidates and list of their ballots): each winner's list of ballots that elected them
    """
    # TODO (not andrew written): replace these with list/set[Candidate] once candidate is well-defined?
    def __init__(self, curr_round: int, elected: list[str] = [], eliminated: list[str]=[], remaining: list[str]=[], winner_votes: dict={}, profile: PreferenceProfile=None,  previous =None):
        self.curr_round = curr_round
        self.elected = elected
        self.eliminated = eliminated
        self.remaining = remaining
        self.profile = profile
        self.winner_vote = winner_votes
        self.previous = previous

    #class Config:
     #   allow_mutation = False

    def get_curr_round(self) -> int:
        return self.curr_round
    
    def get_remaining(self) -> list[str]:
        return self.remaining
    
    def get_elected(self) -> list[str]:
        return self.elected
    
    def get_eliminated(self) -> list[str]:
        return self.eliminated
    
    
    def get_all_winners(self) -> list[str]: # returns in order of first elected - last elected
        if self.previous != None:
            return self.previous.get_all_winners() + self.elected
        else:
            return self.elected
    
    def get_all_eliminated(self) -> list[str]: #returns in order of last eliminated - first eliminated
        if self.previous != None:
            l = self.eliminated.copy()
            l.reverse()
            return l +  self.previous.get_all_eliminated() 
        else:
            l = self.eliminated.copy()
            l.reverse()
            return l
    def get_rankings(self) -> list[str]:
        return self.get_all_winners() + self.remaining +  self.get_all_eliminated()
    
    def get_profile(self) -> PreferenceProfile:
        if self.profile != None:
            return self.profile
        elif self.previous==None:
            raise ValueError(f"No profile found")
        else:
            return self.previous.get_profile()
    
    def get_round_outcome(self,roundNum: int) -> list[str]:
        if self.curr_round == roundNum:
            # current round's eliminated and/or elected
            if self.elected != []:
                return self.elected
            else:
                return self.eliminated
        else:
            self.previous.get_round_outcome(roundNum)
    
  ###############################################################################################
    
    """def add_winners_and_losers(self, winners: set[str], losers: set[str]) -> "Outcome":
        # example method, feel free to delete if not useful
        if not winners.issubset(self.remaining) or not losers.issubset(self.remaining):
            missing = (winners.difference(set(self.remaining)) | (losersdifference(set(self.remaining)))
            raise ValueError(f"Cannot promote winners, {missing} not in remaining")
        return Outcome(
            remaining=set(self.remaining).difference(winners | losers),
            elected=list(set(self.elected) | winners)
            eliminated=list(set(self.eliminated) | losers)
        )
    """

    
    # THIS ASSUMES BALLOTS DON'T CHANGE BY ROUND!
    def difference_remaining_candidates(self, prevOutcome1: PreferenceProfile, prevOutcome2: PreferenceProfile) -> float:
        """
        Returns the fractional difference in number of remaining candidates
        """
        
        # check if from same contest
        if set(prevOutcome1.get_profile().ballots) != set(prevOutcome2.get_profile().ballots):
            raise ValueError("Cannot compare outcomes from different elections")
        
        remaining_diff = len((prevOutcome1.remaining).difference(prevOutcome2.remaining))
        allcandidates = len(prevOutcome1.remaining) + len(prevOutcome1.get_all_winners()) + len(prevOutcome1.get_all_eliminated())
        return remaining_diff / allcandidates
    
    def changed_rankings(self, Outcome) -> dict: # {str candidate : (oldRank, newRank)}
        """
        Returns dict of (key) string candidates who changed ranking from previous round and (value) a tuple of (prevRank, newRank)
        """
        if self.previous == None:
            raise ValueError("This is the first round, cannot compare previous ranking")
        
        else:
            curr_ranking = self.get_rankings()
            prev_ranking = self.previous.get_rankings()
        
            if curr_ranking == prev_ranking:
                return {}
        
            changes = {}
            for index, candidate in enumerate(curr_rankings):
                if candidate != prev_ranking[index]:
                    prev_rank = prev_ranking.index(candidate)
                    changes[candidate] = (prev_rank, index)
            return changes
    
 

    
                    
        
        
        
    
