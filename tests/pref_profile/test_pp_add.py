from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile


def test_add_profiles():
    profile_1 = PreferenceProfile(
        ballots=[
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"D"}), id="X29", voter_set={"Chris"}
            ),
            Ballot(),
            Ballot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )

    profile_2 = PreferenceProfile(
        ballots=[
            Ballot(
                weight=2,
                scores={
                    "A": 1,
                    "B": 2,
                },
            ),
            Ballot(scores={"D": 2, "E": 1}, id="X29", voter_set={"Chris"}),
            Ballot(),
            Ballot(weight=0),
        ],
        candidates=["A", "B", "D", "E", "F"],
        max_ranking_length=0,
    )
    summed_profile = profile_1 + profile_2
    true_summed_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"D"}), id="X29", voter_set={"Chris"}
            ),
            Ballot(
                weight=2,
                scores={
                    "A": 1,
                    "B": 2,
                },
            ),
            Ballot(scores={"D": 2, "E": 1}, id="X29", voter_set={"Chris"}),
            Ballot(),
            Ballot(weight=0),
            Ballot(),
            Ballot(weight=0),
        ),
        candidates=["A", "B", "C", "D", "E", "F"],
        max_ranking_length=3,
    )

    assert set(summed_profile.candidates) == set(["A", "B", "C", "D", "E", "F"])
    assert summed_profile.max_ranking_length == 3
    assert summed_profile.contains_rankings
    assert summed_profile.contains_scores
    assert true_summed_profile == summed_profile
