import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import condense_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile


@pytest.fixture
def profile():
    return RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, frozenset(), frozenset(), {"B"}, frozenset()), weight=2),
            RankBallot(ranking=({"C"}, frozenset(), frozenset())),
            RankBallot(ranking=(frozenset(),)),
        ]
    )


def test_condense_profile(profile):
    cleaned_profile = condense_rank_profile(profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile

    assert cleaned_profile.ballots == (
        RankBallot(ranking=({"A"}, {"B"}), weight=2),
        RankBallot(ranking=({"C"},)),
    )
    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {2}
    assert cleaned_profile.nonempty_altr_idxs == {0}
    assert cleaned_profile.unaltr_idxs == {1}
    assert profile.candidates == cleaned_profile.candidates
    assert profile.max_ranking_length == cleaned_profile.max_ranking_length


def test_condense_profile_idempotent(profile):
    cleaned_profile = condense_rank_profile(profile)
    double_cleaned = condense_rank_profile(cleaned_profile)

    assert cleaned_profile == double_cleaned


def test_condense_profile_equivalence(profile):
    cleaned = condense_rank_profile(profile)

    assert cleaned.nonempty_altr_idxs == {0}
    assert cleaned.no_rank_altr_idxs == {2}
    assert cleaned.unaltr_idxs == {1}


def test_condense_profile_with_reduced_max_ranking_length(profile):
    cleaned_profile = condense_rank_profile(profile, reduce_max_ranking_length=True)

    assert cleaned_profile.max_ranking_length != profile.max_ranking_length
    assert cleaned_profile.max_ranking_length == 2
