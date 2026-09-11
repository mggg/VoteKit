import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import remove_cand_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile


@pytest.fixture
def profile_no_ties():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=1),
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1 / 2),
            RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        ]
    )


@pytest.fixture
def profile_with_ties():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A", "B"}], weight=1),
            RankBallot(ranking=[{"A", "B", "C"}], weight=1 / 2),
            RankBallot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
        ]
    )


def test_remove_cand(profile_no_ties):
    cleaned_profile = remove_cand_rank_profile("A", profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[frozenset(), {"B"}], weight=1),
        RankBallot(ranking=[frozenset(), {"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}, {"B"}, frozenset()], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()
    assert set(cleaned_profile.candidates) == {"B", "C"}


def test_remove_mult_cands(profile_no_ties):
    cleaned_profile = remove_cand_rank_profile(["A", "B"], profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert set(grouped.ballots) == set(
        [
            RankBallot(ranking=[frozenset(), frozenset()], weight=1),
            RankBallot(ranking=[frozenset(), frozenset(), {"C"}], weight=1 / 2),
            RankBallot(ranking=[{"C"}, frozenset(), frozenset()], weight=3),
        ]
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()
    assert set(cleaned_profile.candidates) == {"C"}


def test_remove_cand_with_ties(profile_with_ties):
    cleaned_profile = remove_cand_rank_profile(["A", "B"], profile_with_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_with_ties

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert set(grouped.ballots) == set(
        [
            RankBallot(ranking=[frozenset()], weight=1),
            RankBallot(ranking=[{"C"}], weight=1 / 2),
            RankBallot(ranking=[frozenset(), {"C"}, frozenset()], weight=3),
        ]
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0, 1, 2}
    assert cleaned_profile.unaltr_idxs == set()
    assert set(cleaned_profile.candidates) == {"C"}


def test_remove_cand_maintain_candidates_list(profile_no_ties):
    cleaned_profile = remove_cand_rank_profile(
        "A", profile_no_ties, retain_original_candidate_list=True
    )

    assert cleaned_profile.max_ranking_length == profile_no_ties.max_ranking_length
    assert set(cleaned_profile.candidates) == set(profile_no_ties.candidates)
