import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import (
    condense_rank_profile,
    remove_and_condense_rank_profile,
    remove_cand_rank_profile,
)
from votekit.pref_profile import CleanedRankProfile, RankProfile

profile_no_ties = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A"}, {"B"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
    ]
)

profile_with_ties = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A", "B"}], weight=1),
        RankBallot(ranking=[{"A", "B", "C"}], weight=1 / 2),
        RankBallot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
    ]
)


def test_remove_and_condense():
    cleaned_profile = remove_and_condense_rank_profile("A", profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"B"}], weight=1),
        RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_then_condense_equivalence():
    cleaned_profile_1 = remove_and_condense_rank_profile("A", profile_no_ties)
    cleaned_profile_2 = condense_rank_profile(remove_cand_rank_profile("A", profile_no_ties))

    assert cleaned_profile_1 == cleaned_profile_2


def test_remove_mult_cands():
    cleaned_profile = remove_and_condense_rank_profile(["A", "B"], profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert set(grouped.ballots) == set(
        [
            RankBallot(ranking=[{"C"}], weight=7 / 2),
        ]
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {0}
    assert cleaned_profile.nonempty_altr_idxs == {1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_and_condense_with_ties():

    cleaned_profile = remove_and_condense_rank_profile(["A", "B"], profile_with_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_with_ties

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert set(grouped.ballots) == set(
        [
            RankBallot(ranking=[{"C"}], weight=7 / 2),
        ]
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {0}
    assert cleaned_profile.nonempty_altr_idxs == {1, 2}
    assert cleaned_profile.unaltr_idxs == set()
