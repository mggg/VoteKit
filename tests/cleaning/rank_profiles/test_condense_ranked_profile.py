from votekit.pref_profile import PreferenceProfile, CleanedRankProfile
from votekit.ballot import Ballot
from votekit.cleaning import condense_ranked_profile


def test_condense_profile():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=({"A"}, frozenset(), frozenset(), {"B"}, frozenset()), weight=2
            ),
            Ballot(ranking=({"C"}, frozenset(), frozenset())),
            Ballot(ranking=(frozenset(),)),
        ]
    )
    cleaned_profile = condense_ranked_profile(profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile

    assert cleaned_profile.ballots == (
        Ballot(ranking=({"A"}, {"B"}), weight=2),
        Ballot(ranking=({"C"},)),
    )
    assert cleaned_profile != profile
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {2}
    assert cleaned_profile.nonempty_altr_idxs == {0}
    assert cleaned_profile.unaltr_idxs == {1}


def test_condense_profile_idempotent():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=({"A"}, frozenset(), frozenset(), {"B"}, frozenset()), weight=2
            ),
            Ballot(ranking=({"C"}, frozenset(), frozenset())),
            Ballot(ranking=(frozenset(),)),
        ]
    )

    cleaned_profile = condense_ranked_profile(profile)
    double_cleaned = condense_ranked_profile(cleaned_profile)

    assert cleaned_profile == double_cleaned


def test_condense_profile_equivalence():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=({"A"}, frozenset(), frozenset(), {"B"}, frozenset()), weight=2
            ),
            Ballot(ranking=({"C"}, frozenset(), frozenset())),
            Ballot(ranking=(frozenset(),)),
        ]
    )

    cleaned = condense_ranked_profile(profile)

    assert cleaned.nonempty_altr_idxs == {0}
    assert cleaned.no_rank_altr_idxs == {2}
    assert cleaned.unaltr_idxs == {1}
