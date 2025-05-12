from votekit import Ballot, PreferenceProfile
from votekit.elections import BoostedRandomDictator
import random
import numpy as np
from joblib import Parallel, delayed
import itertools
from votekit.utils import first_place_votes
import pytest


def run_election_once(test_profile):
    """Run one election and return the winner."""
    election = BoostedRandomDictator(test_profile, 1)
    return list(election.get_elected()[0])[0]


def test_boosted_random_dictator_error():
    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        BoostedRandomDictator(PreferenceProfile(), m=1)


def test_boosted_random_dictator_simple():
    random.seed(919717)

    candidates = ["A", "B", "C"]
    ballots = [
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=3),
        Ballot(ranking=[{"B"}, {"A"}, {"C"}]),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}]),
    ]
    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    winner_counts = {c: 0 for c in candidates}
    trials = 3000

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    # check to make sure that the fraction of wins matches the true probability
    assert np.allclose(
        1 / 2 * 3 / 5 + 1 / 2 * 9 / 11, winner_counts["A"] / trials, atol=2e-2
    )
    assert np.allclose(
        1 / 2 * 1 / 5 + 1 / 2 * 1 / 11, winner_counts["B"] / trials, atol=2e-2
    )
    assert np.allclose(
        1 / 2 * 1 / 5 + 1 / 2 * 1 / 11, winner_counts["C"] / trials, atol=2e-2
    )


def test_boosted_random_dictator_4_candidates_without_ties():
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
    tot_fpv_sq = sum(x**2 for x in fpv.values())
    fpv_sq_dict = {c: v**2 / tot_fpv_sq for c, v in fpv.items()}
    fpv = {c: v / tot_fpv for c, v in fpv.items()}

    trials = 3000

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(
        2 / 3 * float(fpv["A"]) + 1 / 3 * float(fpv_sq_dict["A"]),
        winner_counts["A"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["B"]) + 1 / 3 * float(fpv_sq_dict["B"]),
        winner_counts["B"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["C"]) + 1 / 3 * float(fpv_sq_dict["C"]),
        winner_counts["C"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["D"]) + 1 / 3 * float(fpv_sq_dict["D"]),
        winner_counts["D"] / trials,
        atol=2e-2,
    )


def test_boosted_random_dictator_4_candidates_with_ties():
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

    trials = 3500

    fpv = first_place_votes(test_profile, tie_convention="average")
    tot_fpv = sum(fpv.values())
    tot_fpv_sq = sum(x**2 for x in fpv.values())
    fpv_sq_dict = {c: v**2 / tot_fpv_sq for c, v in fpv.items()}
    fpv = {c: v / tot_fpv for c, v in fpv.items()}

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}
    assert np.allclose(
        2 / 3 * float(fpv["A"]) + 1 / 3 * float(fpv_sq_dict["A"]),
        winner_counts["A"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["B"]) + 1 / 3 * float(fpv_sq_dict["B"]),
        winner_counts["B"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["C"]) + 1 / 3 * float(fpv_sq_dict["C"]),
        winner_counts["C"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["D"]) + 1 / 3 * float(fpv_sq_dict["D"]),
        winner_counts["D"] / trials,
        atol=2e-2,
    )


def test_random_dictator_4_candidates_large_sample(all_possible_ranked_ballots):
    random.seed(919717)

    candidates = ["A", "B", "C", "D"]

    ballots = all_possible_ranked_ballots(candidates)

    trials = 3000

    for i, ballot in enumerate(ballots):
        if "A" in ballot.ranking[0]:
            ballots[i] = Ballot(ranking=ballot.ranking, weight=500)

    test_profile = PreferenceProfile(ballots=ballots, candidates=candidates)

    fpv = first_place_votes(test_profile, tie_convention="average")
    tot_fpv = sum(fpv.values())
    tot_fpv_sq = sum(x**2 for x in fpv.values())
    fpv_sq_dict = {c: v**2 / tot_fpv_sq for c, v in fpv.items()}
    fpv = {c: v / tot_fpv for c, v in fpv.items()}

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(test_profile) for _ in range(trials)
    )

    winner_counts = {c: results.count(c) for c in candidates}

    assert np.allclose(
        2 / 3 * float(fpv["A"]) + 1 / 3 * float(fpv_sq_dict["A"]),
        winner_counts["A"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["B"]) + 1 / 3 * float(fpv_sq_dict["B"]),
        winner_counts["B"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["C"]) + 1 / 3 * float(fpv_sq_dict["C"]),
        winner_counts["C"] / trials,
        atol=2e-2,
    )
    assert np.allclose(
        2 / 3 * float(fpv["D"]) + 1 / 3 * float(fpv_sq_dict["D"]),
        winner_counts["D"] / trials,
        atol=2e-2,
    )
