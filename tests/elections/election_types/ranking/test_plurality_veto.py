from votekit import Ballot, PreferenceProfile
from votekit.elections import PluralityVeto
import random
import numpy as np
import itertools
from joblib import Parallel, delayed
import pytest


def run_election_once(test_profile):
    """Run one election and return the winner."""
    election = PluralityVeto(test_profile, 1)
    return list(election.get_elected()[0])[0]


def test_plurality_veto_error():
    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        PluralityVeto(PreferenceProfile(), m=1)


def test_plurality_veto_simple_3_candidates():
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
    trials = 2000

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    # check to make sure that the fraction of wins matches the true probability
    assert np.allclose(1 / 3, winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(1 / 3, winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(1 / 3, winner_counts["C"] / trials, atol=5e-2)


def test_plurality_veto_4_candidates_without_ties():
    random.seed(919717)

    candidates = ["A", "B", "C", "D"]

    full_power = list(
        list(
            list(itertools.permutations(x))
            for x in itertools.combinations(candidates, r)
        )
        for r in range(1, len(candidates) + 1)
    )
    powerset = [x for sublist in full_power for item in sublist for x in item]

    ballots = list(map(lambda x: Ballot(ranking=list(set(y) for y in x)), powerset))
    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)
    trials = 5000

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(1 / 4, winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["C"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["D"] / trials, atol=5e-2)


def run_election_once_with_ties(test_profile):
    """Run one election and return the winner."""
    election = PluralityVeto(test_profile, 1, tiebreak="random")
    return list(election.get_elected()[0])[0]


def test_plurality_veto_4_candidates_with_ties():
    random.seed(919717)

    candidates = ["A", "B", "C", "D"]

    powerset = list(
        itertools.chain.from_iterable(
            itertools.combinations(candidates, r) for r in range(1, len(candidates) + 1)
        )
    )
    ballots = list(map(lambda x: Ballot(ranking=[x]), powerset))

    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    trials = 5000

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once_with_ties)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(1 / 4, winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["C"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["D"] / trials, atol=5e-2)


def test_plurality_veto_4_candidates_large_sample(all_possible_ranked_ballots):
    random.seed(919717)

    candidates = ["A", "B", "C", "D"]

    ballots = all_possible_ranked_ballots(candidates)

    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    trials = 5000

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once_with_ties)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(1 / 4, winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["C"] / trials, atol=5e-2)
    assert np.allclose(1 / 4, winner_counts["D"] / trials, atol=5e-2)
