import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import remove_ballots_with_cand_rank_profile, remove_cand_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile


@pytest.fixture
def profile_no_ties():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
            RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
            RankBallot(ranking=[{"C"}], weight=3),
            RankBallot(ranking=[{"B", "C", "A"}], weight=3),
        ],
        max_ranking_length=3,
    )


@pytest.fixture
def profile_with_ties():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A", "B"}, {"C"}], weight=1),
            RankBallot(ranking=[{"B", "C"}], weight=1 / 2),
            RankBallot(ranking=[{"C"}], weight=3),
            RankBallot(ranking=[{"B", "C", "A"}], weight=3),
        ],
        max_ranking_length=3,
    )


def test_remove_ballots_with_cand_rank_profile(profile_no_ties):
    cleaned_profile = remove_ballots_with_cand_rank_profile("A", profile_no_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == set()
    assert cleaned_profile.unaltr_idxs == {0, 1, 2, 3}
    assert set(cleaned_profile.df.index) == {1, 2}
    assert set(cleaned_profile.candidates) == {"B", "C"}


def test_remove_ballots_with_cand_rank_profile_with_ties(profile_with_ties):
    cleaned_profile = remove_ballots_with_cand_rank_profile("A", profile_with_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_with_ties
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"B", "C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}], weight=3),
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == set()
    assert cleaned_profile.unaltr_idxs == {0, 1, 2, 3}
    assert set(cleaned_profile.df.index) == {1, 2}
    assert set(cleaned_profile.candidates) == {"B", "C"}


def test_remove_ballots_with_mult_cands(profile_no_ties):
    cleaned_profile = remove_ballots_with_cand_rank_profile(["A", "B"], profile_no_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (RankBallot(ranking=[{"C"}], weight=3),)
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == set()
    assert cleaned_profile.unaltr_idxs == {0, 1, 2, 3}
    assert set(cleaned_profile.df.index) == {2}
    assert set(cleaned_profile.candidates) == {"C"}


def test_remove_ballots_without_retain_original_candidate_list(profile_no_ties):
    cleaned_profile = remove_ballots_with_cand_rank_profile(
        "A", profile_no_ties, retain_original_candidate_list=True
    )
    assert cleaned_profile.candidates == profile_no_ties.candidates


def test_remove_ballots_with_invalid_cand_type(profile_no_ties):
    with pytest.raises(TypeError, match=r"Non-string\/integer candidate\(s\) found in removed"):
        remove_ballots_with_cand_rank_profile([1.0], profile_no_ties)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in removed"):
        remove_ballots_with_cand_rank_profile(True, profile_no_ties)


def test_remove_ballots_with_cand_idempotent(profile_no_ties):
    cleaned_profile = remove_ballots_with_cand_rank_profile("A", profile_no_ties)
    double_cleaned = remove_ballots_with_cand_rank_profile("A", cleaned_profile)

    assert cleaned_profile == double_cleaned


def test_remove_ballots_with_candidate_chaining_remove_cand(profile_no_ties):
    cleaned_profile = remove_ballots_with_cand_rank_profile("A", profile_no_ties)
    double_cleaned = remove_cand_rank_profile("A", cleaned_profile)

    assert cleaned_profile == double_cleaned
    assert double_cleaned.parent_profile == cleaned_profile
    assert (
        cleaned_profile.ballots
        == double_cleaned.ballots
        == (
            RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
            RankBallot(ranking=[{"C"}], weight=3),
        )
    )

    cleaned_profile = remove_cand_rank_profile("A", profile_no_ties)
    double_cleaned = remove_ballots_with_cand_rank_profile("A", cleaned_profile)

    assert cleaned_profile == double_cleaned
    assert double_cleaned.parent_profile == cleaned_profile
    assert (
        cleaned_profile.ballots
        == double_cleaned.ballots
        == (
            RankBallot(ranking=[{}, {"B"}, {"C"}], weight=1),
            RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
            RankBallot(ranking=[{"C"}], weight=3),
            RankBallot(ranking=[{"B", "C"}], weight=3),
        )
    )


def test_remove_ballots_with_different_candidates_chaining(profile_no_ties):
    first_cleaned_profile = remove_ballots_with_cand_rank_profile("A", profile_no_ties)

    assert first_cleaned_profile.parent_profile == profile_no_ties
    assert list(first_cleaned_profile.df.index) == first_cleaned_profile.df_index_column == [1, 2]
    assert first_cleaned_profile.unaltr_idxs == {0, 1, 2, 3}
    assert first_cleaned_profile.ballots == (
        RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}], weight=3),
    )

    second_cleaned_profile = remove_ballots_with_cand_rank_profile("B", first_cleaned_profile)

    assert second_cleaned_profile.parent_profile == first_cleaned_profile
    assert list(second_cleaned_profile.df.index) == second_cleaned_profile.df_index_column == [2]
    assert second_cleaned_profile.unaltr_idxs == {1, 2}
    assert second_cleaned_profile.ballots == (RankBallot(ranking=[{"C"}], weight=3),)

    for cleaned in (first_cleaned_profile, second_cleaned_profile):
        assert cleaned.no_rank_altr_idxs == set()
        assert cleaned.no_wt_altr_idxs == set()
        assert cleaned.nonempty_altr_idxs == set()
