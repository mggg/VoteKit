from votekit import Ballot, PreferenceProfile
from votekit.elections import BoostedRandomDictator
import random
import numpy as np


# TODO make tests in line with other election types
def test_boosted_random_dictator():
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
    trials = 3000
    for t in range(trials):
        election = BoostedRandomDictator(test_profile, 1)
        winner = list(election.get_elected()[0])[0]
        winner_counts[winner] += 1

    # check to make sure that the fraction of wins matches the true probability
    assert np.allclose(
        1 / 2 * 3 / 5 + 1 / 2 * 9 / 11, winner_counts["A"] / trials, atol=1e-2
    )
