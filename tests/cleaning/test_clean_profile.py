from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import clean_profile

profile_no_ties = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=1),
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
    ]
)


def test_clean_profile():
    profile = clean_profile(
        profile_no_ties,
        lambda x: Ballot(
            ranking=[c_set for c_set in x.ranking if "A" not in c_set], weight=x.weight
        ),
    )

    assert isinstance(profile, PreferenceProfile)
    assert profile.ballots == (
        Ballot(ranking=[{"B"}], weight=1),
        Ballot(ranking=[{"B"}, {"C"}], weight=1),
        Ballot(ranking=[{"C"}, {"B"}], weight=3),
    )
