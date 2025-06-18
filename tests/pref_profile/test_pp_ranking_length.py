from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile


def test_ranking_length_default():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C", "D"})),
            Ballot(ranking=({"A"}, {"B"}), weight=3 / 2),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )

    assert profile.max_ranking_length == 3


def test_ranking_length_no_default():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C", "D"})),
            Ballot(ranking=({"A"}, {"B"}), weight=3 / 2),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        ),
        max_ranking_length=4,
    )

    assert profile.max_ranking_length == 4
