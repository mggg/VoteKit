import pickle
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.elections import IRV, STV, SequentialRCV, fractional_transfer, random_transfer
from votekit.graphs.ballot_graph import BallotGraph
from votekit.pref_profile import RankProfile
from votekit.pref_profile.utils import (
    convert_rank_profile_to_score_profile_via_score_vector,
    rank_profile_to_ballot_dict,
    rank_profile_to_ranking_dict,
)
from votekit.utils import ballot_lengths, mentions


def _three_candidate_profile(*weights: object) -> RankProfile:
    if not weights:
        weights = (3, 2, 1)
    rankings = (({"A"}, {"B"}, {"C"}), ({"B"}, {"C"}, {"A"}), ({"C"}, {"A"}, {"B"}))
    return RankProfile(
        ballots=tuple(
            RankBallot(ranking=ranking, weight=cast(Any, weight))
            for ranking, weight in zip(rankings, weights)
        )
    )


def _score_for(election: STV, round_number: int, candidate: str):
    return next(
        score
        for scored_candidate, score in election.election_states[round_number].scores.items()
        if scored_candidate == candidate
    )


def test_exact_stv_preserves_rationals_through_states_and_replay():
    profile = _three_candidate_profile(Fraction(7, 3), Fraction(5, 3), Fraction(1))
    election = STV(profile, n_seats=2, exact=True, rng_seed=2)

    assert all(isinstance(ballot.weight, Fraction) for ballot in profile.ballots)
    assert all(
        isinstance(score, Fraction)
        for state in election.election_states
        for score in state.scores.values()
    )
    for round_number in range(len(election.election_states)):
        replayed_profile, state = election.get_step(round_number)
        assert state == election.election_states[round_number]
        assert all(isinstance(ballot.weight, Fraction) for ballot in replayed_profile.ballots)


def test_exact_stv_does_not_mutate_float_input_profile():
    profile = _three_candidate_profile(3, 2, 1)
    STV(profile, n_seats=2, exact=True, rng_seed=2)

    assert all(isinstance(ballot.weight, float) for ballot in profile.ballots)


@pytest.mark.parametrize("value", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_exact_stv_converts_integral_real_weights(value: object):
    profile = _three_candidate_profile()
    df = profile.df.copy()
    df["Weight"] = pd.Series([value, 2, 1], dtype=object)
    converted = RankProfile(
        df=df,
        candidates=profile.candidates,
        max_ranking_length=profile.max_ranking_length,
    )

    election = STV(converted, n_seats=2, exact=True, rng_seed=2)

    assert all(isinstance(ballot.weight, Fraction) for ballot in election.get_profile(0).ballots)


def test_exact_stv_rejects_non_integral_float_weight():
    profile = _three_candidate_profile()
    df = profile.df.copy()
    df["Weight"] = pd.Series([1.5, 2, 1], dtype=object)
    invalid = RankProfile(
        df=df,
        candidates=profile.candidates,
        max_ranking_length=profile.max_ranking_length,
    )

    with pytest.raises(ValueError, match=r"pass Fraction\(numerator, denominator\)"):
        STV(invalid, n_seats=2, exact=True)


def test_exact_stv_rejects_non_integral_longdouble_without_narrowing():
    weight = np.longdouble(2**53) + np.longdouble("0.5")
    if weight == int(weight):
        return

    profile = _three_candidate_profile()
    df = profile.df.copy()
    df["Weight"] = pd.Series([weight, 2, 1], dtype=object)
    invalid = RankProfile(
        df=df,
        candidates=profile.candidates,
        max_ranking_length=profile.max_ranking_length,
    )

    with pytest.raises(ValueError, match=r"pass Fraction\(numerator, denominator\)"):
        STV(invalid, n_seats=2, exact=True)


@pytest.mark.parametrize("weight", [-1, Fraction(-1, 3)])
def test_exact_stv_rejects_negative_dataframe_weight(weight: object):
    profile = _three_candidate_profile()
    df = profile.df.copy()
    df["Weight"] = pd.Series([weight, 2, 1], dtype=object)
    invalid = RankProfile(
        df=df,
        candidates=profile.candidates,
        max_ranking_length=profile.max_ranking_length,
    )

    with pytest.raises(ValueError, match="Ballot weight cannot be negative"):
        STV(invalid, n_seats=2, exact=True)


@pytest.mark.parametrize(
    ("quota", "total", "expected"),
    [
        ("droop", 2**53 + 1, (2**53 + 1) // 3 + 1),
        ("hare", 2**53 + 3, (2**53 + 3) // 2),
    ],
)
def test_exact_stv_large_threshold_uses_exact_total(quota: str, total: int, expected: int):
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}), weight=Fraction(total - 1)),
            RankBallot(ranking=({"B"}, {"A"}), weight=Fraction(1)),
        )
    )

    election = STV(profile, n_seats=2, quota=cast(Any, quota), exact=True)

    assert election.threshold == expected


def test_default_stv_converts_rational_profile_to_float_path():
    weights = (3002399751580327, 3002399751580331, 3002399751580335)
    rational_profile = _three_candidate_profile(*(Fraction(weight) for weight in weights))
    float_profile = _three_candidate_profile(*(float(weight) for weight in weights))

    rational_election = STV(
        rational_profile,
        n_seats=2,
        simultaneous=False,
        tiebreak="random",
        rng_seed=1,
    )
    float_election = STV(
        float_profile,
        n_seats=2,
        simultaneous=False,
        tiebreak="random",
        rng_seed=1,
    )

    assert rational_election.threshold == float_election.threshold == 3002399751580331
    assert rational_election.get_elected() == float_election.get_elected()
    assert all(
        isinstance(ballot.weight, float) for ballot in rational_election.get_profile(0).ballots
    )


def test_exact_stv_rejects_zero_hare_quota():
    profile = RankProfile(ballots=(RankBallot(ranking=({"A"}, {"B"}), weight=Fraction(1, 3)),))

    with pytest.raises(ValueError, match="positive Hare quota"):
        STV(profile, n_seats=2, quota="hare", exact=True)


def test_default_stv_preserves_zero_hare_quota_behavior():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=(1, 2, 0), weight=2 / 15),
            RankBallot(ranking=(0, 1, 2), weight=8 / 15),
            RankBallot(ranking=(2, 1, 0), weight=5 / 15),
        )
    )

    election = STV(
        profile,
        n_seats=2,
        quota="hare",
        simultaneous=False,
        tiebreak="random",
        rng_seed=6,
    )

    assert election.threshold == 0
    assert election.get_elected() == ({0}, {1})


def test_exact_stv_keeps_zero_first_place_candidate_and_terminal_state():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3)),
            RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=Fraction(1)),
        )
    )
    election = STV(profile, n_seats=2, exact=True, rng_seed=3)

    assert _score_for(election, 0, "B") == Fraction(0)
    assert election.election_states[-1].scores == {}
    assert election.get_profile(-1).ballots == ()


def test_exact_stv_tallies_and_tie_are_exact():
    # Pullled from issue #311
    ballot_a = RankBallot(ranking=[{f"A{i + 1}"} for i in range(5)])
    ballot_b = RankBallot(ranking=({"B1"},))
    profile = RankProfile(ballots=[ballot_a] * 52 + [ballot_b] * 8)

    election = STV(profile, n_seats=5, exact=True)

    assert [_score_for(election, i, f"A{i + 1}") for i in range(5)] == [
        Fraction(52),
        Fraction(41),
        Fraction(30),
        Fraction(19),
        Fraction(8),
    ]
    assert _score_for(election, 4, "A5") == _score_for(election, 4, "B1")
    assert election.get_elected() == ({"A1"}, {"A2"}, {"A3"}, {"A4"}, {"B1"})


def test_exact_stv_distinguishes_scores_below_float_resolution():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}), weight=Fraction(2**54 + 1)),
            RankBallot(ranking=({"B"}, {"A"}), weight=Fraction(2**54)),
        )
    )

    election = STV(profile, exact=True)

    assert _score_for(election, 0, "A") > _score_for(election, 0, "B")
    assert float(_score_for(election, 0, "A")) == float(_score_for(election, 0, "B"))


def test_exact_stv_records_seeded_loser_tie_for_replay():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            RankBallot(ranking=({"B"}, {"A"}, {"C"}), weight=Fraction(1)),
            RankBallot(ranking=({"C"}, {"A"}, {"B"}), weight=Fraction(1)),
        )
    )
    election = STV(profile, exact=True, rng_seed=358)
    recorded_order = tuple(state.eliminated for state in election.election_states)
    rng_state = election._rng.getstate()

    for round_number in range(len(election.election_states)):
        election.get_step(round_number)

    assert election._rng.getstate() == rng_state
    assert tuple(state.eliminated for state in election.election_states) == recorded_order
    assert any(state.tiebreaks for state in election.election_states)


@pytest.mark.parametrize("tiebreak", ["alphabetical", "lexicographic", "alph", "lex"])
def test_exact_stv_accepts_lexicographic_tiebreak_aliases(tiebreak: str):
    election = STV(
        _three_candidate_profile(),
        n_seats=2,
        exact=True,
        tiebreak=cast(Any, tiebreak),
        rng_seed=2,
    )

    assert all(
        isinstance(score, Fraction)
        for state in election.election_states
        for score in state.scores.values()
    )


def test_exact_stv_uses_exact_borda_tiebreak():
    large_weight = 2**54
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"C"}, {"B"}), weight=Fraction(large_weight)),
            RankBallot(ranking=({"B"}, {"C"}, {"A"}), weight=Fraction(large_weight)),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            RankBallot(ranking=({"B"}, {"C"}, {"A"}), weight=Fraction(1)),
        )
    )

    election = STV(
        profile,
        n_seats=2,
        simultaneous=False,
        exact=True,
        tiebreak="borda",
    )

    resolution = next(
        resolved
        for state in election.election_states
        for tied, resolved in state.tiebreaks.items()
        if tied == frozenset({"A", "B"})
    )
    assert resolution[0] == frozenset({"B"})


def test_exact_stv_accepts_first_place_tiebreak():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"C"}, {"B"})),
            RankBallot(ranking=({"B"}, {"A"}, {"C"})),
        )
    )

    election = STV(
        profile,
        n_seats=2,
        simultaneous=False,
        exact=True,
        tiebreak="first_place",
        rng_seed=358,
    )

    assert any(
        tied == frozenset({"A", "B"})
        for state in election.election_states
        for tied in state.tiebreaks
    )


def test_exact_stv_rejects_invalid_tiebreak():
    with pytest.raises(ValueError, match="supports only"):
        STV(
            _three_candidate_profile(),
            n_seats=2,
            exact=True,
            tiebreak=cast(Any, "invalid"),
        )


def test_exact_stv_accepts_custom_fraction_preserving_transfer():
    profile = _three_candidate_profile()
    calls = []

    def custom_transfer(winner, fpv, ballots, threshold):
        calls.append((winner, fpv))
        assert isinstance(fpv, Fraction)
        assert all(isinstance(ballot.weight, Fraction) for ballot in ballots)
        return fractional_transfer(winner, fpv, ballots, threshold)

    election = STV(profile, n_seats=2, exact=True, transfer=custom_transfer)

    assert calls
    assert all(
        isinstance(score, Fraction)
        for state in election.election_states
        for score in state.scores.values()
    )


def test_exact_stv_random_transfer_preserves_fraction_weights():
    profile = _three_candidate_profile(Fraction(3), Fraction(1), Fraction(1))

    election = STV(
        profile,
        n_seats=2,
        exact=True,
        transfer=random_transfer,
        rng_seed=1,
    )

    assert all(
        isinstance(score, Fraction)
        for state in election.election_states
        for score in state.scores.values()
    )
    assert all(
        isinstance(ballot.weight, Fraction)
        for round_number in range(len(election.election_states))
        for ballot in election.get_profile(round_number).ballots
    )


def test_exact_irv_preserves_fraction_weights_and_scores():
    profile = _three_candidate_profile(Fraction(7, 3), Fraction(5, 3), Fraction(1))

    election = IRV(profile, exact=True)

    assert all(
        isinstance(score, Fraction)
        for state in election.election_states
        for score in state.scores.values()
    )
    assert all(
        isinstance(ballot.weight, Fraction)
        for round_number in range(len(election.election_states))
        for ballot in election.get_profile(round_number).ballots
    )


def test_exact_sequential_rcv_preserves_fraction_weights_and_scores():
    profile = _three_candidate_profile(Fraction(7, 3), Fraction(5, 3), Fraction(1))

    election = SequentialRCV(profile, n_seats=2, exact=True)

    assert all(
        isinstance(score, Fraction)
        for state in election.election_states
        for score in state.scores.values()
    )
    assert all(
        isinstance(ballot.weight, Fraction)
        for round_number in range(len(election.election_states))
        for ballot in election.get_profile(round_number).ballots
    )


def test_exact_profile_pickle_csv_and_state_dict_boundaries(tmp_path):
    profile = _three_candidate_profile(Fraction(7, 3), Fraction(5, 3), Fraction(1))
    restored = pickle.loads(pickle.dumps(profile))
    election = STV(profile, n_seats=2, exact=True, rng_seed=2)

    assert all(isinstance(ballot.weight, Fraction) for ballot in restored.ballots)
    assert all(
        isinstance(value, Fraction)
        for value in election.election_states[0].to_dict()["scores"].values()
    )
    with pytest.raises(ValueError, match="does not support rational weights"):
        profile.to_csv(tmp_path / "exact.csv")


def test_shared_rational_profile_float_boundaries():
    profile = RankProfile(ballots=(RankBallot(ranking=({"A"}, {"B"}), weight=Fraction(1, 3)),))

    values = (
        *rank_profile_to_ballot_dict(profile, standardize=True).values(),
        *rank_profile_to_ranking_dict(profile).values(),
        *rank_profile_to_ranking_dict(profile, standardize=True).values(),
        *mentions(profile).values(),
        *ballot_lengths(profile).values(),
    )
    score_profile = convert_rank_profile_to_score_profile_via_score_vector(profile, [1, 0])
    graph = BallotGraph(profile)

    assert all(isinstance(value, float) for value in values)
    assert all(isinstance(weight, float) for weight in score_profile._df["Weight"])
    assert isinstance(score_profile.total_ballot_wt, float)
    assert all(isinstance(ballot.weight, float) for ballot in score_profile.ballots)
    assert isinstance(graph.num_voters, float)
    assert all(isinstance(weight, float) for weight in graph.node_weights.values() if weight)
