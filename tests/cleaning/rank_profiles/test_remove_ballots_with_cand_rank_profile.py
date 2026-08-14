import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import remove_ballots_with_cand_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile

profile_no_ties = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}], weight=3),
        RankBallot(ranking=[{"B", "C", "A"}], weight=3),
    ],
    max_ranking_length=3,
)

profile_with_ties = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A", "B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"B", "C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}], weight=3),
        RankBallot(ranking=[{"B", "C", "A"}], weight=3),
    ],
    max_ranking_length=3,
)


def test_remove_ballots_with_cand_rank_profile():
    cleaned_profile = remove_ballots_with_cand_rank_profile("A", profile_no_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == {0, 3}
    assert cleaned_profile.no_rank_altr_idxs == {0, 3}
    assert cleaned_profile.nonempty_altr_idxs == set()
    assert cleaned_profile.unaltr_idxs == {1, 2}


def test_remove_ballots_with_cand_rank_profile_with_ties():
    cleaned_profile = remove_ballots_with_cand_rank_profile("A", profile_with_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_with_ties
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"B", "C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}], weight=3),
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_wt_altr_idxs == {0, 3}
    assert cleaned_profile.no_rank_altr_idxs == {0, 3}
    assert cleaned_profile.nonempty_altr_idxs == set()
    assert cleaned_profile.unaltr_idxs == {1, 2}


def test_remove_ballots_with_mult_cands():
    cleaned_profile = remove_ballots_with_cand_rank_profile(["A", "B"], profile_no_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (RankBallot(ranking=[{"C"}], weight=3),)
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == {0, 1, 3}
    assert cleaned_profile.no_rank_altr_idxs == {0, 1, 3}
    assert cleaned_profile.nonempty_altr_idxs == set()
    assert cleaned_profile.unaltr_idxs == {2}


def test_remove_ballots_with_invalid_cand_type():
    with pytest.raises(TypeError, match="Candidates must be strings or integers within removed."):
        remove_ballots_with_cand_rank_profile([1.0], profile_no_ties)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="removed must be a str/int candidate or a list of candidates."
    ):
        remove_ballots_with_cand_rank_profile(True, profile_no_ties)
