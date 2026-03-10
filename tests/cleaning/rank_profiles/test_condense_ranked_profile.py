from votekit.ballot import RankBallot
from votekit.cleaning import condense_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile


def test_condense_profile():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, frozenset(), frozenset(), {"B"}, frozenset()), weight=2),
            RankBallot(ranking=({"C"}, frozenset(), frozenset())),
            RankBallot(ranking=(frozenset(),)),
        ]
    )
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


def test_condense_profile_idempotent():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, frozenset(), frozenset(), {"B"}, frozenset()), weight=2),
            RankBallot(ranking=({"C"}, frozenset(), frozenset())),
            RankBallot(ranking=(frozenset(),)),
        ]
    )

    cleaned_profile = condense_rank_profile(profile)
    double_cleaned = condense_rank_profile(cleaned_profile)

    assert cleaned_profile == double_cleaned


def test_condense_profile_equivalence():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, frozenset(), frozenset(), {"B"}, frozenset()), weight=2),
            RankBallot(ranking=({"C"}, frozenset(), frozenset())),
            RankBallot(ranking=(frozenset(),)),
        ]
    )

    cleaned = condense_rank_profile(profile)

    assert cleaned.nonempty_altr_idxs == {0}
    assert cleaned.no_rank_altr_idxs == {2}
    assert cleaned.unaltr_idxs == {1}
