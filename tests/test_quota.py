from votekit.cvr_loaders import blt
from votekit.election_types import STV, fractional_transfer
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def test_droop_default_parameter():

    pp, seats = blt(DATA_DIR / "edinburgh17/edinburgh17-01.blt")

    election = STV(pp, fractional_transfer, seats=seats)

    while not election.is_complete():
        pp, out = election.run_step(pp)

    # print(out)


def test_droop_inputed_parameter():

    pp, seats = blt(DATA_DIR / "edinburgh17/edinburgh17-01.blt")

    election = STV(pp, fractional_transfer, seats=seats, quota="Droop")

    while not election.is_complete():
        pp, out = election.run_step(pp)

    # print(out)


def test_quota_misspelled_parameter():

    pp, seats = blt(DATA_DIR / "edinburgh17/edinburgh17-01.blt")

    with pytest.raises(ValueError):
        _ = STV(pp, fractional_transfer, seats=seats, quota="droops")

    # print(out)


def test_hare_quota():

    pp, seats = blt(DATA_DIR / "edinburgh17/edinburgh17-01.blt")

    election = STV(pp, fractional_transfer, seats=seats, quota="hare")

    while not election.is_complete():
        pp, out = election.run_step(pp)

    # print(out)
