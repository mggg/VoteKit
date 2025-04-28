from votekit.pref_profile import PreferenceProfile, CleanedProfile
from votekit.ballot import Ballot
from votekit.cleaning import condense_profile


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

    cleaned_profile = condense_profile(profile)

    assert isinstance(cleaned_profile, CleanedProfile)
    assert cleaned_profile.parent_profile == profile

    assert cleaned_profile.ballots == (
        Ballot(ranking=({"A"}, {"B"}), weight=2),
        Ballot(ranking=({"C"},)),
    )

    assert cleaned_profile != profile
    assert cleaned_profile.no_weight_altr_ballot_indices == set()
    assert cleaned_profile.no_ranking_and_no_scores_altr_ballot_indices == {2}
    assert cleaned_profile.valid_but_altr_ballot_indices == {0}
    assert cleaned_profile.unaltr_ballot_indices == {1}


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

    cleaned_profile = condense_profile(profile)
    double_cleaned = condense_profile(cleaned_profile)

    assert cleaned_profile == double_cleaned
