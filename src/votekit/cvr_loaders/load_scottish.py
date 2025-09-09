import os
import csv
from pathlib import Path
from pandas.errors import EmptyDataError, DataError
from typing import Union

from votekit.pref_profile import RankProfile
from votekit.ballot import RankBallot


def load_scottish(
    fpath: Union[str, os.PathLike, Path],
) -> tuple[PreferenceProfile, int, list[str], dict[str, str], str]:
    """
    Given a file path, loads cast vote record from format used for Scottish election data
    in (this repo)[https://github.com/mggg/scot-elex].

    Args:
        fpath (str): Path to Scottish election csv file.

    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        DataError: If there is missing or incorrect metadata or candidate data.

    Returns:
        tuple:
            A tuple ``(PreferenceProfile, seats, cand_list, cand_to_party, ward)``
            representing the election, the number of seats in the election, the candidate
            names, a dictionary mapping candidates to their party, and the ward. The
            candidate names are also stored in the PreferenceProfile object.
    """

    fpath = str(fpath)

    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"File with path {fpath} cannot be found")
    if os.path.getsize(fpath) == 0:
        raise EmptyDataError(f"CSV at {fpath} is empty.")

    # Convert the ballot rows to ints while leaving the candidates as strings
    def convert_row(row):
        return [int(item) if item.isdigit() else item for item in row]

    data = []
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # This just removes any empty strings that are hanging out since
            # we don't need to preserve columns
            filtered_row = list(filter(lambda x: x != "", row))

            # only save non-empty rows
            if len(filtered_row) > 0:
                data.append(convert_row(filtered_row))

    if len(data[0]) != 2:
        raise DataError(
            "The metadata in the first row should be number of \
                            candidates, seats."
        )

    cand_num, seats = data[0][0], data[0][1]
    ward = data[-1][0]

    num_to_cand = {}
    cand_to_party = {}

    data_cand_num = len([r for r in data if "Candidate" in str(r[0])])
    if data_cand_num != cand_num:
        raise DataError(
            "Incorrect number of candidates in either first row metadata \
                        or in candidate list at end of csv file."
        )

    # record candidate names, which are up until the final row
    for i, line in enumerate(data[len(data) - (cand_num + 1) : -1]):
        if "Candidate" not in line[0]:
            raise DataError(
                f"The number of candidates on line 1 is {cand_num}, which\
                            does not match the metadata."
            )
        cand = line[1]
        party = line[2]

        # candidates are 1 indexed
        num_to_cand[i + 1] = cand
        cand_to_party[cand] = party

    cand_list = list(cand_to_party.keys())

    ballots = [Ballot()] * len(data[1 : len(data) - (cand_num + 1)])

    for i, line in enumerate(data[1 : len(data) - (cand_num + 1)]):
        ballot_weight = line[0]
        cand_ordering = line[1:]
        ranking = tuple([frozenset({num_to_cand[n]}) for n in cand_ordering])

        ballots[i] = Ballot(ranking=ranking, weight=ballot_weight)

    profile = PreferenceProfile(
        ballots=tuple(ballots), candidates=tuple(cand_list)
    ).group_ballots()
    return (profile, seats, cand_list, cand_to_party, ward)
