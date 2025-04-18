from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from fractions import Fraction


def test_profile_equals_rankings():
    profile1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
        )
    )
    profile2 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3)),
        )
    )
    assert profile1 == profile2


def test_profile_equals_mixed():
    profile_1 = PreferenceProfile(
        ballots=[
            Ballot(
                weight=2,
                scores={
                    "A": 1,
                    "B": 2,
                },
            ),
            Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=2,
                scores={
                    "A": 1,
                    "B": 2,
                },
            ),
        ]
    )

    profile_2 = PreferenceProfile(
        ballots=[
            Ballot(
                weight=1,
                scores={
                    "A": 1,
                    "B": 2,
                },
            ),
            Ballot(
                weight=1,
                scores={
                    "A": 1,
                    "B": 2,
                },
            ),
            Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            Ballot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=2,
                scores={
                    "A": 1,
                    "B": 2,
                },
            ),
        ]
    )
    assert profile_1 == profile_2


def test_profile_not_equals_candidates():
    profile1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
        )
    )
    profile2 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3)),
        ),
        candidates=["A", "B", "C", "D", "E"],
    )

    assert profile1 != profile2


def test_profile_not_equals_ballot_length():
    profile1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
        )
    )
    profile2 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3)),
        ),
        max_ranking_length=4,
    )

    assert profile1 != profile2


def test_profile_not_equals_cand_cast():
    profile1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
        )
    )
    profile2 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"F"}, {"C"}, {"B"}), weight=Fraction(2)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3)),
        ),
    )

    assert profile1 != profile2


def test_profile_not_equals_ballot_wt():
    profile1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
        )
    )
    profile2 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(3)),
        ),
    )

    assert profile1 != profile2


def test_profile_not_equals_contains_rankings():
    profile1 = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"E"}, {"C"}, {"B"}), weight=Fraction(2)),
        ),
        contains_rankings=True,
    )
    profile2 = PreferenceProfile(ballots=(Ballot(),), contains_rankings=False)

    assert profile1 != profile2


def test_profile_not_equals_contains_scores():
    profile1 = PreferenceProfile(ballots=(Ballot(),), contains_scores=False)
    profile2 = PreferenceProfile(
        ballots=(Ballot(scores={"A": 1}),), contains_scores=True
    )

    assert profile1 != profile2
