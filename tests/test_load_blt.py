from votekit.cvr_loaders import blt
from votekit.election_types import STV, fractional_transfer, compute_votes
import pytest

from pathlib import Path
from pandas.errors import EmptyDataError, DataError
from fractions import Fraction

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def test_blt_parse():

    pp, seats = blt(DATA_DIR / "edinburgh17-01_abridged.blt")

    election = STV(pp, fractional_transfer, seats=seats)

    counts = compute_votes(
        candidates=election.profile.candidates, ballots=election.profile.ballots
    )

    assert counts["('John', 'LONGSTAFF', 'Ind')"] == Fraction(0)
    assert counts["('Louise', 'YOUNG', 'LD')"] == Fraction(14 + 1)


def test_empty_file_blt():
    with pytest.raises(EmptyDataError):
        pp, seats = blt(DATA_DIR / "empty.blt")


def test_bad_metadata_blt():
    with pytest.raises(DataError):
        pp, seats = blt(DATA_DIR / "bad_metadata.blt")


def test_incorrect_metadata_blt():
    with pytest.raises(DataError):
        pp, seats = blt(DATA_DIR / "candidate_metadata_conflict.blt")
