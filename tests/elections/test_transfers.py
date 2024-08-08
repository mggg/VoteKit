from votekit.elections import fractional_transfer, random_transfer
from votekit import Ballot
from fractions import Fraction
import pytest

ranking_error_ballot_list = [
    Ballot(weight=2),
    Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=2),
    Ballot(ranking=({"C"}, {"B"}, {"A"}), weight=1),
]

integer_error_ballot_list = [
    Ballot(weight=Fraction(3, 2)),
    Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=2.3),
    Ballot(ranking=({"C"}, {"B"}, {"A"}), weight=1),
]


def test_fractional_transfer():
    fractional_ballot_list = [
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(2)),
        Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=Fraction(2)),
        Ballot(ranking=({"C"}, {"B"}, {"A"}), weight=Fraction(1)),
    ]
    new_ballots = fractional_transfer("A", 4, fractional_ballot_list, 2)
    assert set(new_ballots) == set(
        (
            Ballot(ranking=({"B"}, {"C"}), weight=Fraction(1)),
            Ballot(ranking=({"C"}, {"B"}), weight=Fraction(2)),
        )
    )


def test_fractional_transfer_error():
    with pytest.raises(TypeError, match="has no ranking."):
        fractional_transfer("A", 4, ranking_error_ballot_list, 2)


def test_random_transfer():
    random_ballot_list = [
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(4)),
        Ballot(ranking=({"C"}, {"B"}, {"A"}), weight=Fraction(1)),
    ]

    new_ballots = random_transfer("A", 4, random_ballot_list, 2)
    print(new_ballots)

    assert set(new_ballots) == set(
        (
            Ballot(ranking=({"B"}, {"C"}), weight=Fraction(2)),
            Ballot(ranking=({"C"}, {"B"}), weight=Fraction(1)),
        )
    )


def test_random_transfer_error():
    with pytest.raises(TypeError, match="has no ranking."):
        random_transfer("A", 4, ranking_error_ballot_list, 2)

    with pytest.raises(TypeError, match="does not have integer weight."):
        random_transfer("A", 4, integer_error_ballot_list, 2)
