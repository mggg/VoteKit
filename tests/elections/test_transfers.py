from fractions import Fraction

import pytest

from votekit.ballot import RankBallot
from votekit.elections import fractional_transfer, random_transfer

ranking_error_ballot_list = [
    RankBallot(weight=2),
    RankBallot(ranking=({"A"}, {"C"}, {"B"}), weight=2),
    RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=1),
]

integer_error_ballot_list = [
    RankBallot(weight=3 / 2),
    RankBallot(ranking=({"A"}, {"C"}, {"B"}), weight=2.3),
    RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=1),
]


def test_fractional_transfer():
    fractional_ballot_list = [
        RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
        RankBallot(ranking=({"A"}, {"C"}, {"B"}), weight=2),
        RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=1),
    ]
    new_ballots = fractional_transfer("A", 4, fractional_ballot_list, 2)
    assert set(new_ballots) == set(
        (
            RankBallot(ranking=({"B"}, {"C"}), weight=1),
            RankBallot(ranking=({"C"}, {"B"}), weight=1),
            RankBallot(ranking=({"C"}, {"B"}), weight=1),
        )
    )


def test_fractional_transfer_preserves_large_rational_weight():
    denominator = 2**80 + 7
    ballot = RankBallot(ranking=({"A"}, {"B"}), weight=Fraction(2**90 + 3, denominator))
    fpv = ballot.weight

    transferred = fractional_transfer("A", fpv, [ballot], 1)

    expected = ballot.weight * (fpv - 1) / fpv
    assert transferred[0].weight == expected
    assert isinstance(transferred[0].weight, Fraction)


def test_fractional_transfer_error():
    with pytest.raises(TypeError, match="has no ranking."):
        fractional_transfer("A", 4, ranking_error_ballot_list, 2)


def test_random_transfer():
    random_ballot_list = [
        RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=4),
        RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=1),
    ]

    new_ballots = random_transfer("A", 4, random_ballot_list, 2)

    assert set(new_ballots) == set(
        (
            RankBallot(ranking=({"B"}, {"C"}), weight=1),
            RankBallot(ranking=({"B"}, {"C"}), weight=1),
            RankBallot(ranking=({"C"}, {"B"}), weight=1),
        )
    )


def test_random_transfer_error():
    with pytest.raises(TypeError, match="has no ranking."):
        random_transfer("A", 4, ranking_error_ballot_list, 2)

    with pytest.raises(TypeError, match="does not have integer weight."):
        random_transfer("A", 4, integer_error_ballot_list, 2)
