from votekit.ballot import Ballot
from votekit.election_types import random_transfer, compute_votes
from fractions import Fraction


def test_rand_transfer_func_mock_data():
    winner = "A"
    ballots = [
        Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=Fraction(2)),
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
    ]
    votes = {"A": 3}
    threshold = 1

    ballots_after_transfer = random_transfer(
        winner=winner, ballots=ballots, votes=votes, threshold=threshold
    )

    counts = compute_votes(candidates=["B", "C"], ballots=ballots_after_transfer)

    assert counts["C"] == Fraction(1) or counts["C"] == Fraction(2)
    assert counts["B"] == 2 - counts["C"]


def test_rand_transfer_assert():
    winner = "A"
    ballots = [
        Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=Fraction(1000)),
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1000)),
    ]
    votes = {"A": 2000}
    threshold = 1000

    ballots_after_transfer = random_transfer(
        winner=winner, ballots=ballots, votes=votes, threshold=threshold
    )
    counts = compute_votes(candidates=["B", "C"], ballots=ballots_after_transfer)

    assert counts["B"] + counts["C"] == 1000
    assert 400 < counts["B"] < 600
