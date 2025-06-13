from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile


def test_pp_group_ballots_ranking():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=1, voter_set={"Chris"}),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=2,
                voter_set={"Moon", "Peter"},
            ),
        ),
        candidates=("A", "B", "C", "D"),
    )

    pp = profile.group_ballots()
    assert pp.ballots == (
        Ballot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=5,
            voter_set={"Chris", "Moon", "Peter"},
        ),
    )
    assert set(pp.candidates) == set(profile.candidates)
    assert profile == pp


def test_group_ballot_scores():
    profile = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
                weight=1,
                voter_set={"Chris"},
            ),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=2,
            ),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
                weight=2,
                voter_set={"Peter", "Moon"},
            ),
        )
    )
    pp = profile.group_ballots()

    assert (
        Ballot(
            ranking=({"A"}, {"B"}, {"C"}),
            scores={"A": 3, "B": 2},
            weight=3,
            voter_set={"Chris", "Moon", "Peter"},
        )
        in pp.ballots
    )
    assert (
        Ballot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=2,
        )
        in pp.ballots
    )

    assert profile == pp
