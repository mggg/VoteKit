from __future__ import annotations


def _validate_csv_ballot_weight(
    ballot_row: list[str],
    row_index: int,
):
    """
    Validate that the ballot weight is formatted correctly.

    Args:
        ballot_row (list[str]): A ballot row.
        row_index (int): The index of the row in the csv, 0-indexed.

    Raises:
        ValueError: If the ballot weight is improperly formatted for VoteKit.
    """
    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    break_idxs = [i for i, string in enumerate(ballot_row) if string == "&"]
    if break_idxs[1] - break_idxs[0] != 2:
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} has a weight"
                " entry that is too long or short. " + boiler_plate
            )
        )

    else:

        try:
            float(ballot_row[break_idxs[0] + 1])

        except ValueError:
            raise ValueError(
                (
                    f"csv file is improperly formatted. Ballot in row {row_index} has a "
                    "weight entry that can't be converted to float "
                    f"{ballot_row[break_idxs[0] + 1]}. " + boiler_plate
                )
            )


def _validate_csv_ballot_voter_set(
    ballot_row: list[str], row_index: int, include_voter_set: bool
):
    """
    Validate that the ballot voter set is formatted correctly.

    Args:
        ballot_row (list[str]): A ballot row.
        row_index (int): The index of the row in the csv, 0-indexed.
        include_voter_set (bool): Whether or not there is a voter set.

    Raises:
        ValueError: If the ballot voter set is improperly formatted for VoteKit.
    """

    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    break_idxs = [i for i, string in enumerate(ballot_row) if string == "&"]

    if not include_voter_set and len(ballot_row[break_idxs[-1] + 1 :]) > 0:
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} has a "
                f"voter set but it should not: {ballot_row[break_idxs[-1] + 1 :]} "
                + boiler_plate
            )
        )


def _validate_csv_ballot_row_break_idxs(ballot_row: list[str], row_index: int):
    """
    Validate that the ballot number of & symbols in each row is correct.

    Args:
        ballot_row (list[str]): A ballot row.
        row_index (int): The index of the row in the csv, 0-indexed.

    Raises:
        ValueError: If the ballot row is improperly formatted for VoteKit.
    """
    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )
    break_idxs = [i for i, string in enumerate(ballot_row) if string == "&"]

    if len(break_idxs) != 2:
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} does not have 2 &"
                f"symbols, it has {len(break_idxs)}." + boiler_plate
            )
        )
