from votekit.profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.election_state import ElectionState

# from typing import Callable
# import random
from fractions import Fraction


# class STV:
#     def __init__(self, profile: PreferenceProfile, transfer: Callable, seats: int):
#         self.profile = profile
#         self.transfer = transfer
#         self.elected: set = set()
#         self.eliminated: set = set()
#         self.seats = seats
#         self.threshold = self.get_threshold()

#     # can cache since it will not change throughout rounds
#     def get_threshold(self) -> int:
#         """
#         Droop qouta
#         """
#         return int(self.profile.num_ballots() / (self.seats + 1) + 1)

#     # change name of this function and reverse bool
#     def is_complete(self) -> bool:
#         """
#         Determines if the number of seats has been met to call election
#         """
#         return len(self.elected) == self.seats

#     def run_step(self, profile: PreferenceProfile) -> tuple[PreferenceProfile, Outcome]:
#         """
#         Simulates one round an STV election
#         """
#         candidates: list = profile.get_candidates()
#         ballots: list = profile.get_ballots()
#         fp_votes: dict = compute_votes(candidates, ballots)

#         # print('ballots', type(ballots))
#         # print('candidates', type(candidates))

#         # if number of remaining candidates equals number of remaining seats
#         if len(candidates) == self.seats - len(self.elected):
#             # TODO: sort remaing candidates by vote share
#             self.elected.update(set(candidates))
#             return profile, Outcome(
#                 elected=self.elected,
#                 eliminated=self.eliminated,
#                 remaining=set(candidates),
#                 votes=fp_votes,
#             )

#         for candidate in candidates:
#             if fp_votes[candidate] >= self.threshold:
#                 self.elected.add(candidate)
#                 candidates.remove(candidate)
#                 ballots = self.transfer(candidate, ballots, fp_votes, self.threshold)

#         if not self.is_complete():
#             lp_votes = min(fp_votes.values())
#             lp_candidates = [
#                 candidate for candidate, votes in fp_votes.items() if votes == lp_votes
#             ]
#             # is this how to break ties, can be different based on locality
#             lp_cand = random.choice(lp_candidates)
#             ballots = remove_cand(lp_cand, ballots)
#             candidates.remove(lp_cand)
#             # print("loser", lp_cand)
#             self.eliminated.add(lp_cand)

#         return PreferenceProfile(ballots=ballots), Outcome(
#             elected=self.elected,
#             eliminated=self.eliminated,
#             remaining=set(candidates),
#             votes=fp_votes,
#         )


class Borda:
    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        borda_weights: list = None,
        standard: bool = True,
    ):
        self.state = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=profile.get_candidates(),
            profile=profile,
            winner_votes={},
            previous=None,
        )
        self.seats = seats

        self.num_cands = len(self.state.profile.get_candidates())

        if borda_weights is None:
            self.borda_weights = [i for i in range(self.num_cands, 0, -1)]
        else:
            self.borda_weights = borda_weights

        self.standard = standard

    def run_borda_step(self):
        """
        Simulates a complete Borda election
        """
        borda_scores = {}  # {candidate : int borda_score}
        candidates_ballots = {}  # {candidate : [ballots that ranked candidate at all]}

        # Populates dicts: candidate_rank_freq, candidates_ballots
        for ballot in self.state.profile.get_ballots():
            frequency = ballot.weight
            rank = 0
            for candidate in ballot.ranking:
                candidate = str(candidate)
                if candidate not in candidates_ballots:
                    candidates_ballots[candidate] = []
                candidates_ballots[candidate].append(
                    ballot
                )  # adds ballot where candidate was ranked

                # adds Borda score to candidate
                if candidate not in borda_scores:
                    borda_scores[candidate] = 0
                if (rank + 1) <= len(self.borda_weights):
                    borda_scores[candidate] += frequency * self.borda_weights[rank]

                rank += 1

            if self.standard is True:
                if rank < (len(self.borda_weights) + 1):
                    # X find total remaining borda points
                    remaining_points = sum(self.borda_weights[rank:])

                    # Y find unranked candidates by the ballot

                    ballot_ranking = [
                        item for subset in ballot.ranking for item in subset
                    ]
                    remaining_cands = list(
                        set(self.state.profile.get_candidates()) - set(ballot_ranking)
                    )

                    # borda_scores[all remaining candidates] = X / Y
                    for candidate in remaining_cands:
                        candidate = str(set(candidate))
                        if candidate not in borda_scores:
                            borda_scores[candidate] = frequency * (
                                remaining_points / len(remaining_cands)
                            )
                        else:
                            borda_scores[candidate] += frequency * (
                                remaining_points / len(remaining_cands)
                            )

        # Identifies Borda winners (elected) and losers (eliminated)
        sorted_borda = sorted(borda_scores, key=borda_scores.get, reverse=True)
        winners = sorted_borda[: self.seats]
        losers = sorted_borda[self.seats :]

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
            previous=self.state,
        )
        return self.state

    def run_borda_election(self):
        final_state = self.run_borda_step()
        return final_state


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


def seqRCV_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Doesn't transfer votes, useful for Sequential RCV election
    """
    return ballots


def remove_cand(removed_cand: str, ballots: list[Ballot]) -> list[Ballot]:
    """
    Removes candidate from ranking of the ballots
    """
    for n, ballot in enumerate(ballots):
        new_ranking = []
        for candidate in ballot.ranking:
            if candidate != {removed_cand}:
                new_ranking.append(candidate)
        ballots[n].ranking = new_ranking

    return ballots
