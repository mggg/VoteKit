from .profile import PreferenceProfile
from .ballot import Ballot
from .models import Outcome
from typing import Callable
from fractions import Fraction
from copy import deepcopy
from election_state import ElectionState
import random
import itertools


# old STV class
"""class STV:
    def __init__(self, profile: PreferenceProfile, transfer: Callable, seats: int):
        self.profile = profile
        self.transfer = transfer
        self.elected: set = set()
        self.eliminated: set = set()
        self.seats = seats
        self.threshold = self.get_threshold()

    # can cache since it will not change throughout rounds
    def get_threshold(self) -> int:
        """
        #Drop qouta
        """
        return int(self.profile.num_ballots() / (self.seats + 1) + 1)

    def next_round(self) -> bool:
        """
        #Determines if the number of seats has been met to call election
        """
        return len(self.elected) != self.seats

    def run_step(self, profile: PreferenceProfile) -> tuple[PreferenceProfile, Outcome]:
        """
        #Simulates one round an STV election
        """
        candidates: list = profile.get_candidates()
        ballots: list = profile.get_ballots()
        fp_votes: dict = compute_votes(candidates, ballots)

        # if number of remaining candidates equals number of remaining seats
        if len(candidates) == self.seats - len(self.elected):
            # TODO: sort remaing candidates by vote share
            self.elected.update(set(candidates))
            return profile, Outcome(
                elected=self.elected,
                eliminated=self.eliminated,
                remaining=set(candidates),
                votes=fp_votes,
            )

        for candidate in candidates:
            if fp_votes[candidate] >= self.threshold:
                self.elected.add(candidate)
                candidates.remove(candidate)
                ballots = self.transfer(candidate, ballots, fp_votes, self.threshold)

        if self.next_round():
            lp_votes = min(fp_votes.values())
            lp_candidates = [
                candidate for candidate, votes in fp_votes.items() if votes == lp_votes
            ]
            # is this how to break ties, can be different based on locality
            lp_cand = random.choice(lp_candidates)
            ballots = remove_cand(lp_cand, ballots)
            candidates.remove(lp_cand)
            self.eliminated.add(lp_cand)

        return PreferenceProfile(ballots=ballots), Outcome(
            elected=self.elected,
            eliminated=self.eliminated,
            remaining=set(candidates),
            votes=fp_votes,
        )

    def run_election(self) -> Outcome:
        """
        #Runs complete STV election
        """
        profile = deepcopy(self.profile)

        if not self.next_round():
            raise ValueError(
                f"Length of elected set equal to number of seats ({self.seats})"
            )

        while self.next_round():
            profile, outcome = self.run_step(profile)

        return outcome
"""
        
class Approval:
    def __init__(self, profile: PreferenceProfile, seats: int):
        self.state = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=profile.get_candidates(),
            profile=profile,
            winner_votes={},
            previous=None
        )
        self.seats = seats


    def run_approval_step(self):
        """
        Simulates a complete Approval Voting election, take N winners with most approval votes
        """
        
        approval_scores = {} # {candidate : [num times ranked 1st, num times ranked 2nd, ...]}
        candidates_ballots = {}  # {candidate : [ballots that ranked candidate at all]}
        
        for ballot in self.state.profile.get_ballots():
            weight = ballot.weight
            for candidate in ballot.ranking:
                candidate = str(candidate)
                
                # populates approval_scores
                if candidate not in approval_scores:
                    approval_scores[candidate] = 0
                approval_scores[candidate] += weight
                
                # populates candidates_ballots (for ElectionState's winner_votes)
                if candidate not in candidates_ballots:
                    candidates_ballots[candidate] = []
                candidates_ballots[candidate].append(ballot) # adds ballot where candidate was ranked

        # Identifies winners (elected) and losers (eliminated)
        sorted_approvals = sorted(approval_scores, key=approval_scores.get, reverse=True)
        winners = sorted_approvals[: self.seats]
        losers = sorted_approvals[self.seats :]    
        
        # Create winner_votes dict for ElectionState object
        winner_ballots = {}
        for candidate in winners:
            winner_ballots[candidate] = candidates_ballots[candidate]
        
        # New final state object
        self.state = ElectionState(
            elected=winners,
            eliminated=losers,
            remaining=[],
            profile=self.state.profile,
            curr_round=(self.state.curr_round + 1),
            winner_votes=winner_ballots,
            previous=self.state
        )
        return self.state
        

    def run_approval_election(self):
        final_state = self.run_approval_step()
        return final_state
    
  
class Range:
    def __init__(self, profile: PreferenceProfile, seats: int):
        self.state = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=profile.get_rangecandidates(),
            profile=profile,
            winner_votes={},
            previous=None
        )
        self.seats = seats


    def run_range_step(self):
        """
        Simulates a complete Range Voting election, take (int seats) winners with most approval votes
        """
        
        range_scores = {} # {candidate : (score_sum, num_times_scored)}
        candidates_ballots = {}  # {candidate : [ballots that ranked candidate at all]}
        
        for ballot in self.state.profile.get_rangeballots():
            weight = ballot.weight
            for candidate in ballot.scoring:
                score = ballot.scoring[candidate]
                candidate = str(candidate)
                
                # populates range_scores
                if candidate not in range_scores:
                    range_scores[candidate] = [0,0]
                range_scores[candidate][0] += (weight * score)
                range_scores[candidate][1] += weight
                
                # populates candidates_ballots (for ElectionState's winner_votes)
                if candidate not in candidates_ballots:
                    candidates_ballots[candidate] = []
                candidates_ballots[candidate].append(ballot) # adds ballot where candidate was ranked

        # Calculates average scores per candidate
        avg_scores = {} # {candidate : average range score}
        for candidate in range_scores:
            avg_scores[candidate] = range_scores[candidate][0] / range_scores[candidate][1]
        
        # Identifies winners (elected) and losers (eliminated)
        sorted_candidates = sorted(avg_scores, key=avg_scores.get, reverse=True)
        winners = sorted_candidates[: self.seats]
        losers = sorted_candidates[self.seats :]    
        
        # Create winner_votes dict for ElectionState object
        winner_ballots = {}
        for candidate in winners:
            winner_ballots[candidate] = candidates_ballots[candidate]
        
        # New final state object
        self.state = ElectionState(
            elected=winners,
            eliminated=losers,
            remaining=[],
            profile=self.state.profile,
            curr_round=(self.state.curr_round + 1),
            winner_votes=winner_ballots,
            previous=self.state
        )
        return self.state
        

    def run_range_election(self):
        final_state = self.run_range_step()
        return final_state
  

class ChamberlinCourant:
    def __init__(self, profile: PreferenceProfile, seats: int, borda_weights: list, score_size: int):
        self.state = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=profile.get_candidates(),
            profile=profile,
            winner_votes={},
            previous=None
        )
        self.seats = seats
        self.borda_weights = borda_weights
        self.score_size = score_size # number of ranked candidates to calculate each committee's Borda score


    def run_cc_step(self):
        """
        Simulates a complete Chamberlin-Courant election step
        """
        
        candidates = self.state.profile.get_candidates()
        committees = list(itertools.combinations(candidates, self.seats))
        borda_scores = {}  # {committee : int borda_score}
        candidate_ballots = {}  # {candidate : [ballots that ranked candidate at all]}
        
        if len(candidates) == 0:
            return self.state
        
        for committee in committees:
            if committee not in borda_scores:
                borda_scores[committee] = 0
            for ballot in self.state.profile.get_ballots():
                frequency = ballot.weight

                # find ballot's (N=score_size) favorite candidates in committees
                rank = 0
                num_scored = 0
                while (num_scored < self.score_size) & ((rank - 1) <= len(self.borda_weights)) & (rank < len(ballot.ranking)):
                    curr_candidate = list(ballot.ranking[rank])[0]
                    if curr_candidate in committee:
                        borda_scores[committee] += (self.borda_weights[rank] * frequency)
                        num_scored += 1
                    
                        # populates candidates_ballots (for ElectionState's winner_votes)
                        if curr_candidate not in candidate_ballots:
                            candidate_ballots[curr_candidate] = []
                        if ballot not in candidate_ballots[curr_candidate]:
                            candidate_ballots[curr_candidate].append(ballot)

                    rank += 1
                
        # Identifies CC winners (elected) and losers (eliminated)
        winners = max(borda_scores, key=borda_scores.get, default=None) #unsorted
        winners = self.sort_by_Borda(winners, candidate_ballots, self.borda_weights, descending_scores=True)
        losers = list(set(candidates) - set(winners)) #unsorted
        losers = self.sort_by_Borda(losers, candidate_ballots, self.borda_weights, descending_scores=False)
        
        # Create winner_votes dict for ElectionState object
        winner_ballots = {}
        for candidate in winners:
            winner_ballots[candidate] = candidate_ballots[candidate]
            
        # New final state object
        self.state = ElectionState(
            elected=winners,
            eliminated=losers,
            remaining=[],
            profile=self.state.profile,
            curr_round=(self.state.curr_round + 1),
            winner_votes=winner_ballots,
            previous=self.state
        )
        return self.state
        

    def run_cc_election(self):
        final_state = self.run_cc_step()
        return final_state
    
    def sort_by_Borda(self, candidates: list, candidate_ballots: dict, borda_weights: list, descending_scores: bool):
        """
        Takes in a list of candidates, returns them sorted by candidates' Borda scores either ascending or descending
        """
        
        cand_borda_scores = {}
        for candidate in candidates:
            cand_borda_scores[candidate] = 0
            for ballot in candidate_ballots[candidate]:
                rankings = [next(iter(cand)) for cand in ballot.ranking]
                rank = rankings.index(candidate)
                if (rank + 1) <= len(borda_weights):
                    cand_borda_scores[candidate] += (ballot.weight * self.borda_weights[rank])
        return sorted(cand_borda_scores, key=cand_borda_scores.get, reverse=descending_scores)




## Election Helper Functions


def compute_votes(candidates: list, ballots: list[Ballot]) -> dict:
    """
    Computes first place votes for all candidates in a preference profile
    """
    votes = {}

    for candidate in candidates:
        weight = Fraction(0)
        for ballot in ballots:
            if ballot.ranking and ballot.ranking[0] == {candidate}:
                weight += ballot.weight
        votes[candidate] = weight

    return votes


def fractional_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    # find the transfer value, add tranfer value to weights of vballots
    # that listed the elected in first place, remove that cand and shift
    # everything up, recomputing first-place votes
    transfer_value = (votes[winner] - threshold) / votes[winner]

    for ballot in ballots:
        if ballot.ranking and ballot.ranking[0] == {winner}:
            ballot.weight = ballot.weight * transfer_value

    transfered = remove_cand(winner, ballots)

    return transfered


def remove_cand(removed_cand: str, ballots: list[Ballot]) -> list[Ballot]:
    """
    Removes candidate from ranking of the ballots
    """
    update = deepcopy(ballots)

    for n, ballot in enumerate(update):
        new_ranking = []
        for candidate in ballot.ranking:
            if candidate != {removed_cand}:
                new_ranking.append(candidate)
        update[n].ranking = new_ranking

    return update

