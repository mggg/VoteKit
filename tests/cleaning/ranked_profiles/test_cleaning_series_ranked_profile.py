from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import condense_ranked_profile, remove_cand_ranked_profile


def test_cleaning_series():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[
                    {"A"},
                ],
                weight=1,
            ),
            Ballot(weight=0),
            Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        ]
    )

    profile_no_A = remove_cand_ranked_profile(["A"], profile)
    condensed = condense_ranked_profile(profile_no_A)
    assert condensed.group_ballots().ballots == (
        Ballot(ranking=[{"C"}, {"B"}], weight=3),
    )
