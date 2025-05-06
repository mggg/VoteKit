from votekit import PreferenceProfile, Ballot
from votekit.elections import RandomDictator
import numpy as np
import random
import itertools
from joblib import Parallel, delayed
from votekit.utils import first_place_votes
import pytest


def run_election_once(test_profile):
    """Run one election and return the winner."""
    election = RandomDictator(test_profile, 1)
    return list(election.get_elected()[0])[0]


def test_random_dictator_error():
    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        RandomDictator(PreferenceProfile(), m=1)


def test_random_dictator_simple():
    # set seed for more predictable results
    random.seed(919717)

    candidates = ["A", "B", "C"]
    ballots = [
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=3),
        Ballot(ranking=[{"B"}, {"A"}, {"C"}]),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}]),
    ]
    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    winner_counts = {c: 0 for c in candidates}
    trials = 600

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    # check to make sure that the fraction of wins matches the true probability
    assert np.allclose(3 / 5, winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(1 / 5, winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(1 / 5, winner_counts["C"] / trials, atol=5e-2)


def test_random_dictator_4_candidates_without_ties():
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

    ballots = list(
        map(
            lambda x: Ballot(
                ranking=list(set(y) for y in x), weight=3 if x[0] == "A" else 1
            ),
            powerset,
        )
    )

    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    fpv = first_place_votes(test_profile)
    tot_fpv = sum(fpv.values())
    fpv = {c: v / tot_fpv for c, v in fpv.items()}

    trials = 500

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(float(fpv["A"]), winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["B"]), winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["C"]), winner_counts["C"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["D"]), winner_counts["D"] / trials, atol=5e-2)


def test_random_dictator_4_candidates_with_ties():
    random.seed(919717)

    candidates = ["A", "B", "C", "D"]

    powerset = list(
        itertools.chain.from_iterable(
            itertools.combinations(candidates, r) for r in range(1, len(candidates) + 1)
        )
    )

    ballots = list(
        map(lambda x: Ballot(ranking=[x], weight=3 if "A" in x[0] else 1), powerset)
    )

    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    trials = 500

    fpv = first_place_votes(test_profile, tie_convention="average")
    tot_fpv = sum(fpv.values())
    fpv = {c: v / tot_fpv for c, v in fpv.items()}

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(float(fpv["A"]), winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["B"]), winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["C"]), winner_counts["C"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["D"]), winner_counts["D"] / trials, atol=5e-2)


def test_random_dictator_4_candidates_large_sample(all_possible_ranked_ballots):
    random.seed(919717)

    candidates = ["A", "B", "C", "D"]

    ballots = all_possible_ranked_ballots(candidates)

    trials = 500

    for i, ballot in enumerate(ballots):
        if "A" in ballot.ranking[0]:
            ballots[i] = Ballot(ranking=ballot.ranking, weight=500)

    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    fpv = first_place_votes(test_profile, tie_convention="average")
    tot_fpv = sum(fpv.values())
    fpv = {c: v / tot_fpv for c, v in fpv.items()}

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(float(fpv["A"]), winner_counts["A"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["B"]), winner_counts["B"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["C"]), winner_counts["C"] / trials, atol=5e-2)
    assert np.allclose(float(fpv["D"]), winner_counts["D"] / trials, atol=5e-2)
