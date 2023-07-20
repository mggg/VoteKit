from votekit.cvr_loaders import rank_column_csv, blt
from votekit.election_types import STV, fractional_transfer
import os
import pytest

from pathlib import Path
from pandas.errors import EmptyDataError, DataError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def test_mn_clean_ballots():

    mn_profile = rank_column_csv(DATA_DIR / "mn_clean_ballots.csv")

    election = STV(mn_profile, fractional_transfer, seats=1)

    while not election.is_complete():
        mn_profile, out = election.run_step(mn_profile)

    print(out)


def test_blt():

    pp, seats = blt(DATA_DIR / "edinburgh17/edinburgh17-01.blt")

    election = STV(pp, fractional_transfer, seats=seats)

    while not election.is_complete():
        pp, out = election.run_step(pp)

    print(out)


# Runs STV on 17 ward council elections from Edinburgh in 2017.
# Results were checked by hand against reported results from
# https://www.edinburgh.gov.uk/election-results
# /local-government-election-results#:~:text=Ward%201%20%2D%20Almond,Democrats%20%2D%20elected%20at%20stage%20one
# All were found to be correct.
def test_edinburgh_elections():

    dir_path = DATA_DIR / "edinburgh17"
    for file_name in os.listdir(dir_path):
        pp, seats = blt(os.path.join(dir_path, file_name))

        election = STV(pp, fractional_transfer, seats=seats)

        while not election.is_complete():
            pp, out = election.run_step(pp)

        print(file_name + ": \n", out)
        # why are the fractions for the remaining candidates so big?


# todo: change to pytest formatting
def test_empty_file_blt():
    with pytest.raises(EmptyDataError):
        pp, seats = blt(DATA_DIR / "empty.blt")


def test_bad_metadata_blt():
    with pytest.raises(DataError):
        pp, seats = blt(DATA_DIR / "bad_metadata.blt")


def test_incorrect_metadata_blt():
    with pytest.raises(DataError):
        pp, seats = blt(DATA_DIR / "candidate_metadata_conflict.blt")


# mn_clean_ballots_test()

# blt_test()

# edinburgh_elections_test()

# empty_file_blt()

# bad_metadata_blt()
