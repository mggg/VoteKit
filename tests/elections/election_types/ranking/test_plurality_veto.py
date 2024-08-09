from votekit import Ballot, PreferenceProfile
from votekit.elections import PluralityVeto
import random
import numpy as np


# TODO make tests in line with other election types
def test_plurality_veto():
    random.seed(919717)

    # simple 3 candidate election
    candidates = ["A", "B", "C"]
    # With every possible permutation of candidates, we should
    # see that each candidate wins with probability 1/3
    ballots = [
        Ballot(ranking=[{"A"}, {"B"}, {"C"}]),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}]),
        Ballot(ranking=[{"B"}, {"A"}, {"C"}]),
        Ballot(ranking=[{"B"}, {"C"}, {"A"}]),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}]),
        Ballot(ranking=[{"C"}, {"A"}, {"B"}]),
    ]
    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    # count the number of wins over a set of trials
    winner_counts = {c: 0 for c in candidates}
    trials = 10000
    for t in range(trials):
        election = PluralityVeto(test_profile, 1)
        election.run_election()
        winner = list(election.state.winners()[0])[0]
        winner_counts[winner] += 1

    # check to make sure that the fraction of wins matches the true probability
    assert np.allclose(1 / 3, winner_counts["A"] / trials, atol=1e-2)
