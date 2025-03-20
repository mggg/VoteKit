from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import clean_profile

profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=1),
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        Ballot(ranking=({"A"},)),
        Ballot(ranking=({"B"},), weight=0),
    ]
)


def test_clean_profile_with_defaults():
    adj_profile = clean_profile(
        profile,
        lambda x: Ballot(
            ranking=[c_set for c_set in x.ranking if "A" not in c_set], weight=x.weight
        ),
    )

    assert isinstance(adj_profile, PreferenceProfile)
    assert adj_profile.ballots == (
        Ballot(ranking=[{"B"}], weight=1),
        Ballot(ranking=[{"B"}, {"C"}], weight=1),
        Ballot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert adj_profile != profile


def test_clean_profile_change_defaults():
    adj_profile, count = clean_profile(
        profile,
        lambda x: Ballot(
            ranking=[c_set for c_set in x.ranking if "A" not in c_set], weight=x.weight
        ),
        return_adjusted_count=True,
        remove_empty_ballots=False,
        remove_zero_weight_ballots=False,
    )

    assert isinstance(adj_profile, PreferenceProfile)
    assert count == 5
    assert set(adj_profile.ballots) == set(
        (
            Ballot(ranking=[{"B"}], weight=1),
            Ballot(ranking=[{"B"}, {"C"}], weight=1),
            Ballot(ranking=[{"C"}, {"B"}], weight=3),
            Ballot(ranking=tuple()),
            Ballot(ranking=({"B"},), weight=0),
        )
    )
