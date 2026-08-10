import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import (
    condense_rank_profile,
    remove_repeat_cands_and_condense_rank_profile,
    remove_repeat_cands_rank_profile,
)
from votekit.pref_profile import CleanedRankProfile, RankProfile


def test_remove_repeated_candidates_and_condense():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"A"}, {"B"}, {"C"}]),
            RankBallot(ranking=[{"A", "C"}, {"C"}, frozenset(), {"B"}]),
        ]
    )

    cleaned = remove_repeat_cands_and_condense_rank_profile(profile)

    assert isinstance(cleaned, CleanedRankProfile)
    assert cleaned.parent_profile == profile
    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned.group_ballots()
    assert grouped.ballots == (
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"A", "C"}, {"B"}], weight=1),
    )
    assert cleaned.nonempty_altr_idxs == {0, 1}
    assert cleaned.unaltr_idxs == set()


def test_combined_cleaning_matches_sequential_cleaning():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"A", "B"}, frozenset(), {"C"}])])

    combined = remove_repeat_cands_and_condense_rank_profile(profile)
    sequential = condense_rank_profile(remove_repeat_cands_rank_profile(profile))

    assert combined.ballots == sequential.ballots
    assert combined.candidates == sequential.candidates


def test_trailing_empty_positions_are_considered_unaltered():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}, frozenset(), frozenset()]),
            RankBallot(ranking=[{"A"}, frozenset(), {"B"}, frozenset()]),
        ]
    )

    cleaned = remove_repeat_cands_and_condense_rank_profile(profile)

    assert cleaned.unaltr_idxs == {0}
    assert cleaned.nonempty_altr_idxs == {1}
