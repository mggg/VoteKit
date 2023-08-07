from votekit.cvr_loaders import blt
from votekit.election_types import STV, fractional_transfer
from pathlib import Path
import pytest


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def test_droop_default_parameter():

    pp, seats = blt(DATA_DIR / "edinburgh17-01_abridged.blt")

    election = STV(pp, fractional_transfer, seats=seats)

    droop_quota = int((8 + 14 + 1 + 13 + 1 + 1 + 2) / (4 + 1)) + 1

    assert election.threshold == droop_quota


def test_droop_inputed_parameter():

    pp, seats = blt(DATA_DIR / "edinburgh17-01_abridged.blt")

    election = STV(pp, fractional_transfer, seats=seats, quota="Droop")

    droop_quota = int((8 + 14 + 1 + 13 + 1 + 1 + 2) / (4 + 1)) + 1

    assert election.threshold == droop_quota


def test_quota_misspelled_parameter():

    pp, seats = blt(DATA_DIR / "edinburgh17-01_abridged.blt")

    with pytest.raises(ValueError):
        _ = STV(pp, fractional_transfer, seats=seats, quota="droops")


def test_hare_quota():

    pp, seats = blt(DATA_DIR / "edinburgh17-01_abridged.blt")

    election = STV(pp, fractional_transfer, seats=seats, quota="hare")

    hare_quota = int((8 + 14 + 1 + 13 + 1 + 1 + 2) / 4)

    assert election.threshold == hare_quota
