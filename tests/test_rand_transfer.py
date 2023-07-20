from votekit.cvr_loaders import blt
from votekit.ballot import Ballot
from votekit.election_types import (
    STV,
    random_transfer,
    fractional_transfer,
    compute_votes,
)
from fractions import Fraction
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def test_rand_transfer_func_mock_data():
    winner = "A"
    ballots = [
        Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=Fraction(2)),
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
        Ballot(ranking=({"D"},), weight=Fraction(1)),
    ]
    votes = {"A": 3}
    threshold = 1

    print(
        random_transfer(
            winner=winner, ballots=ballots, votes=votes, threshold=threshold
        )
    )


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


def test_rand_transfer_edinburgh():

    pp, seats = blt(DATA_DIR / "edinburgh17/edinburgh17-01.blt")

    rand_transfer_election = STV(pp, random_transfer, seats=seats)

    while not rand_transfer_election.is_complete():
        pp, out = rand_transfer_election.run_step(pp)

    rand_transfer_outcome = out

    # run election with fractional transfer and see if you get the same result
    # (possible that this will fail sometimes but usually should succeed)
    pp, seats = blt(DATA_DIR / "edinburgh17/edinburgh17-01.blt")

    frac_transfer_election = STV(pp, fractional_transfer, seats=seats)

    while not frac_transfer_election.is_complete():
        pp, out = frac_transfer_election.run_step(pp)

    frac_transfer_outcome = out

    assert rand_transfer_outcome.elected == frac_transfer_outcome.elected


# rand_transfer_func_mock_data_test()
# rand_transfer_assert_test()
# rand_transfer_edinburgh_test()
