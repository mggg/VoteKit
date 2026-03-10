import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import remove_repeat_cands_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile


def test_remove_repeated_candidates():
    ballot = RankBallot(ranking=[{"A"}, {"A"}, {"B"}, {"C"}], weight=1)
    ballot_tuple = (ballot, ballot)
    profile = RankProfile(ballots=ballot_tuple)
    cleaned_profile = remove_repeat_cands_rank_profile(profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert grouped.ballots == (RankBallot(ranking=({"A"}, frozenset(), {"B"}, {"C"}), weight=2),)

    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_repeated_candidates_ties():
    profile = RankProfile(
        ballots=[
            RankBallot(
                ranking=[{"C", "A"}, {"A", "C"}, {"B"}],
            ),
            RankBallot(
                ranking=[{"A", "C"}, {"C", "A"}, {"B"}],
            ),
        ]
    )
    cleaned_profile = remove_repeat_cands_rank_profile(profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert grouped.ballots == (RankBallot(ranking=[{"C", "A"}, frozenset(), {"B"}], weight=2),)

    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1}
    assert cleaned_profile.unaltr_idxs == set()
