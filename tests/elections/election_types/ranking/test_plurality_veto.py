import itertools
import random

import numpy as np
import pytest
from joblib import Parallel, delayed
from typing_extensions import Sequence

from votekit.ballot import RankBallot
from votekit.elections import PluralityVeto
from votekit.pref_profile import RankProfile

BALLOT_WEIGHT = 5
TRIALS = 1000
CANDIDATES = ("A", "B", "C", "D")


def make_complete_ballots(candidates: Sequence[str]) -> RankProfile:
    """Creates all complete and partial ballots, each with weight BALLOT_WEIGHT."""
    full_power = list(
        list(
            list(itertools.permutations(x))
            for x in itertools.combinations(candidates, r)
        )
        for r in range(1, len(candidates) + 1)
    )
    powerset = [x for sublist in full_power for item in sublist for x in item]
    ballots = tuple(
        map(
            lambda x: RankBallot(ranking=list(set(y) for y in x), weight=BALLOT_WEIGHT),
            powerset,
        )
    )
    return RankProfile(ballots=ballots, candidates=candidates)


TEST_PROFILE = make_complete_ballots(candidates=CANDIDATES)


def run_election_once(test_profile, tiebreak):
    """Run one election and return the winner."""
    election = PluralityVeto(test_profile, 1, tiebreak=tiebreak)
    return next(iter(election.get_elected()[0]))


def test_plurality_veto_errors():
    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        PluralityVeto(RankProfile(), m=1)

    with pytest.raises(ValueError, match="m must be positive."):
        PluralityVeto(RankProfile(), m=0)

    with pytest.raises(TypeError, match="Ballots must have rankings."):
        b1 = RankBallot(ranking=("A",))
        b2 = RankBallot()
        PluralityVeto(RankProfile(ballots=(b1, b2)), m=1)

    non_int_weight_msg = (
        r"Ballot RankBallot\n1.\) A, \nWeight: 0.5 has non-integer weight."
    )
    with pytest.raises(TypeError, match=non_int_weight_msg):
        b1 = RankBallot(ranking=("A",), weight=1 / 2)
        PluralityVeto(RankProfile(ballots=(b1,)), m=1)


def test_get_profile_does_not_corrupt_state():
    """Calling get_profile (which replays the election) should not corrupt election state."""
    profile = make_complete_ballots(candidates=("A", "B", "C"))
    election = PluralityVeto(profile, m=1, tiebreak="first_place")

    winners_before = election.get_elected()
    states_before = list(election.election_states)

    # replay various rounds via get_profile
    for i in range(len(election.election_states)):
        p = election.get_profile(i)
        assert isinstance(p, RankProfile)

    # also test negative indexing
    election.get_profile(-1)
    election.get_profile(0)

    winners_after = election.get_elected()
    states_after = list(election.election_states)

    assert winners_before == winners_after
    assert len(states_before) == len(states_after)
    for s_before, s_after in zip(states_before, states_after):
        assert s_before.round_number == s_after.round_number
        assert s_before.elected == s_after.elected
        assert s_before.eliminated == s_after.eliminated
        assert s_before.scores == s_after.scores


@pytest.mark.slow
def test_plurality_veto_4_candidates_deterministic_tiebreaking():
    random.seed(919717)

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(TEST_PROFILE, tiebreak="first_place")
        for _ in range(TRIALS)
    )

    winner_counts = {c: results.count(c) for c in CANDIDATES}
    # first_place tiebreak is indecisive, so we break ties alphabetically,
    # so A should almost always win.
    # There are possible voter orders that cause B to win, but they are extremely rare.
    assert np.allclose(1, winner_counts["A"] / TRIALS, atol=4e-2)
    assert np.allclose(0, winner_counts["B"] / TRIALS, atol=4e-2)
    assert np.allclose(0, winner_counts["C"] / TRIALS, atol=4e-2)
    assert np.allclose(0, winner_counts["D"] / TRIALS, atol=8e-2)


@pytest.mark.slow
def test_plurality_veto_4_candidates_random_tiebreaking():
    random.seed(919717)

    # Parallel execution
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_election_once)(TEST_PROFILE, tiebreak="random")
        for _ in range(TRIALS)
    )

    winner_counts = {c: results.count(c) for c in CANDIDATES}
    assert np.allclose(1 / 4, winner_counts["A"] / TRIALS, atol=8e-2)
    assert np.allclose(1 / 4, winner_counts["B"] / TRIALS, atol=8e-2)
    assert np.allclose(1 / 4, winner_counts["C"] / TRIALS, atol=8e-2)
    assert np.allclose(1 / 4, winner_counts["D"] / TRIALS, atol=8e-2)


def test_serial_veto():
    profile = make_complete_ballots(candidates=("A", "B", "C"))
    election = PluralityVeto(
        profile, 1, tiebreak="first_place", elimination_strategy="careful"
    )
    assert election.get_elected()
