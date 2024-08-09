from votekit import PreferenceProfile, Ballot
from votekit.elections import RandomDictator
import numpy as np
import random


# TODO make tests in line with other election types
def test_random_dictator():
    # set seed for more predictable results
    random.seed(919717)

    # simple 3 candidate election
    candidates = ["A", "B", "C"]
    ballots = [
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=3),
        Ballot(ranking=[{"B"}, {"A"}, {"C"}]),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}]),
    ]
    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    # count the number of wins over a set of trials
    winner_counts = {c: 0 for c in candidates}
    trials = 2000
    for t in range(trials):
        election = RandomDictator(test_profile, 1)
        winner = list(election.get_elected()[0])[0]
        winner_counts[winner] += 1

    # check to make sure that the fraction of wins matches the true probability
    assert np.allclose(3 / 5, winner_counts["A"] / trials, atol=1e-2)
