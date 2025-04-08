from fractions import Fraction
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile


def test_pp_group_ballots_ranking():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1), voter_set={"Chris"}
            ),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=Fraction(2),
                voter_set={"Moon", "Peter"},
            ),
        ),
        candidates=("A", "B", "C", "D"),
    )

    pp = profile.group_ballots()
    assert pp.ballots == (
        Ballot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=Fraction(5),
            voter_set={"Chris", "Moon", "Peter"},
        ),
    )
    assert set(pp.candidates) == set(profile.candidates)


def test_condense_profile_scores():
    profile = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
                weight=Fraction(1),
                voter_set={"Chris"},
            ),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=Fraction(2),
            ),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                scores={"A": 3, "B": 2},
                weight=Fraction(2),
                voter_set={"Peter", "Moon"},
            ),
        )
    )
    pp = profile.group_ballots()

    assert (
        Ballot(
            ranking=({"A"}, {"B"}, {"C"}),
            scores={"A": 3, "B": 2},
            weight=Fraction(3),
            voter_set={"Chris", "Moon", "Peter"},
        )
        in pp.ballots
    )
    assert (
        Ballot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=Fraction(2),
        )
        in pp.ballots
    )
