from votekit.utils import mentions, first_place_votes
from votekit.profile import PreferenceProfile
from votekit.ballot import Ballot
from fractions import Fraction

profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(1), voters={"tom"}),
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voters={"andy"}),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voters={"andy"}),
    ]
)


def test_first_place_votes():
    correct = {"A": 5, "B": 0, "C": 0}
    test = first_place_votes(profile)
    assert correct == test


def test_first_place_ties():
    ties = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A", "B"}, {"D"}], weight=Fraction(1), voters={"tom"}),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voters={"andy"}),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voters={"andy"}),
        ]
    )
    correct = {"A": 4.5, "B": 0.5, "C": 0, "D": 0}
    test = first_place_votes(ties)
    assert test == correct


def test_mentions():
    correct = {"A": 5, "B": 5, "C": 4}
    test = mentions(profile)
    assert correct == test


def test_mentions_with_ties():
    ties = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B", "D"}], weight=Fraction(1), voters={"tom"}),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voters={"andy"}),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voters={"andy"}),
        ]
    )
    correct = {"A": 5, "B": 4.5, "C": 4, "D": 0.5}
    test = mentions(ties)
    assert test == correct
