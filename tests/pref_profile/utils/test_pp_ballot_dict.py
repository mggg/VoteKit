from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.pref_profile.utils import profile_to_ballot_dict


def test_to_ballot_dict():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=3 / 2),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )
    rv = profile_to_ballot_dict(profile, standardize=False)
    assert rv[Ballot(ranking=({"A"}, {"B"}))] == 5 / 2
    assert rv[Ballot(ranking=({"C"}, {"B"}), scores={"A": 4})] == 2 / 1
    assert rv[Ballot(scores={"A": 4})] == 1

    rv = profile_to_ballot_dict(profile, standardize=True)
    assert rv[Ballot(ranking=({"A"}, {"B"}))] == 5 / 11
    assert rv[Ballot(ranking=({"C"}, {"B"}), scores={"A": 4})] == 4 / 11
    assert rv[Ballot(scores={"A": 4})] == 2 / 11
