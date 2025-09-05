from __future__ import annotations
from ...ballot import ScoreBallot
from typing import Tuple
from .csv_utils import (
    _validate_csv_ballot_weight,
    _validate_csv_ballot_voter_set,
    _validate_csv_ballot_row_break_idxs,
)

CANDIDATES_HEADER_ROW = 1
CANDIDATES_MAPPING_ROW = CANDIDATES_HEADER_ROW + 1
VOTER_SET_HEADER_ROW = 3
VOTER_SET_VALUE_ROW = VOTER_SET_HEADER_ROW + 1
EQ_SIGN_ROW = 5
COLUMN_NAMES_ROW = 6
DATA_START_ROW = COLUMN_NAMES_ROW + 1


def _parse_profile_data_from_score_csv(
    csv_data: list[list[str]],
) -> Tuple[dict[str, str], bool, list[int]]:
    """
    Parse the profile data from a ScoreProfile csv.

    Args:
        csv_data (list[list[str]]): Data from csv.

    Returns:
        Tuple[dict[str, str],  bool, list[int]]:
            inv_candidate_mapping, max_ranking_length,
            includes_voter_set, break_indices
    """
    candidate_row = [c_tuple.strip("()").split(":") for c_tuple in csv_data[2]]
    inv_candidate_mapping = {prefix: cand for cand, prefix in candidate_row}

    includes_voter_set = csv_data[VOTER_SET_VALUE_ROW][0] == "True"

    ballot_data_column_names = csv_data[COLUMN_NAMES_ROW]
    break_indices = [
        i for i, col_name in enumerate(ballot_data_column_names) if col_name == "&"
    ]

    return (
        inv_candidate_mapping,
        includes_voter_set,
        break_indices,
    )


def _parse_ballot_from_score_csv(
    ballot_row: list[str],
    includes_voter_set: bool,
    break_indices: list[int],
    inv_candidate_mapping: dict[str, str],
) -> ScoreBallot:
    """
    Parse a ballot from a ScoreProfile csv row.

    Args:
        ballot_row (list[str]): Row from the csv file containing ballot data.
        includes_voter_set (bool): Whether or not the csv contains voter sets.
        break_indices (list[int]): Where the columns of the csv change from one data type to
            another.
        inv_candidate_mapping (dict[str, str]): The iverted candidate mapping of prefix
            to the cand.

    Returns:
        ScoreBallot: Ballot formatted from row of csv.
    """
    candidates = list(inv_candidate_mapping.values())
    voter_set = set()

    try:
        weight = float(ballot_row[break_indices[0] + 1])
    except Exception:
        raise RuntimeError(
            f"Invalid weight format in ballot row: {ballot_row[break_indices[0] + 1]}. Weight "
            "must be float."
        )

    scores = {
        c: float(ballot_row[i]) for i, c in enumerate(candidates) if ballot_row[i]
    }

    if includes_voter_set:
        voter_set = set(v.strip() for v in ballot_row[break_indices[-1] + 1 :])

    return ScoreBallot(scores=scores, voter_set=voter_set, weight=weight)


def _validate_score_csv_header_values(header_data: list[list[str]]):
    """
    Validate that the values of the header rows are correct.

    Args:
        header_data (list[list[str]]): The rows of the csv file.

    Raises:
        ValueError: If csv header rows are improperly formatted for VoteKit.
    """
    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    if any(char not in c_tuple for c_tuple in header_data[2] for char in "(:)"):
        raise ValueError(
            (
                f"csv file is improperly formatted. Row {CANDIDATES_MAPPING_ROW} should contain "
                "tuples mapping candidates "
                "to their unique prefixes. For example, (Chris:Ch), (Colleen: Co). "
                f"Not {header_data[2]}. " + boiler_plate
            )
        )

    if len(header_data[VOTER_SET_VALUE_ROW]) != 1 or header_data[VOTER_SET_VALUE_ROW][
        0
    ] not in ["True", "False"]:
        raise ValueError(
            (
                f"csv file is improperly formatted. Row {VOTER_SET_VALUE_ROW} should be 'True' or "
                f"'False', not {header_data[VOTER_SET_VALUE_ROW]}. " + boiler_plate
            )
        )


def _validate_score_csv_header_rows(header_data: list[list[str]]):
    """
    Validate that the names of the header rows are correct.

    Args:
        header_data (list[list[str]]): The rows of the csv file.

    Raises:
        ValueError: If csv headers rows are improperly formatted for VoteKit.
    """
    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    if header_data[0] != ["VoteKit ScoreProfile"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 0 should be 'VoteKit ScoreProfile',"
                f"not {header_data[0]}. " + boiler_plate
            )
        )

    if header_data[CANDIDATES_HEADER_ROW] != ["Candidates"]:
        raise ValueError(
            (
                f"csv file is improperly formatted. Row {CANDIDATES_HEADER_ROW} should be "
                f"'Candidates', not {header_data[CANDIDATES_HEADER_ROW]}. "
                + boiler_plate
            )
        )

    if header_data[VOTER_SET_HEADER_ROW] != ["Includes Voter Set"]:
        raise ValueError(
            (
                f"csv file is improperly formatted. Row {VOTER_SET_HEADER_ROW} should be"
                f" 'Includes Voter Set', not {header_data[VOTER_SET_HEADER_ROW]}. "
                + boiler_plate
            )
        )

    if header_data[EQ_SIGN_ROW] != ["="] * 10:
        raise ValueError(
            (
                f"csv file is improperly formatted. Row {EQ_SIGN_ROW} should be "
                "'=,=,=,=,=,=,=,=,=,=', "
                f"not {header_data[EQ_SIGN_ROW]}. " + boiler_plate
            )
        )


def _validate_score_csv_header(header_data: list[list[str]]):
    """
    Validate that the values of the header rows are correct.

    Args:
        header_data (list[list[str]]): The rows of the header of the csv file.

    Raises:
        ValueError: If csv header is improperly formatted for VoteKit.
    """

    _validate_score_csv_header_rows(header_data)
    _validate_score_csv_header_values(header_data)


def _validate_score_csv_ballot_header_row(
    ballot_rows: list[list[str]],
    candidate_prefixes: tuple[str],
    include_voter_set: bool,
):
    """
    Validate that the ballot header row is formatted correctly.

    Args:
        ballot_rows (list[list[str]]): All ballot rows.
        candidate_prefixes (tuple[str]): The tuple of candidate prefixes.
        max_ranking_length (int): The max ranking length.
        include_voter_set (bool): Whether or not there is a voter set.

    Raises:
        ValueError: If the ballot header is improperly formatted for VoteKit.
    """
    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    header_row = ballot_rows[0]

    break_indices = [i for i, col_name in enumerate(header_row) if col_name == "&"]

    # TODO what is this if condition?
    if len(header_row[: break_indices[0]]) > 0 and any(
        c not in header_row for c in candidate_prefixes
    ):
        raise ValueError(
            (
                f"csv file is improperly formatted. Row {COLUMN_NAMES_ROW} should include all "
                "candidates before the first &. " + boiler_plate
            )
        )

    if "Weight" not in header_row:
        raise ValueError(
            (
                f"csv file is improperly formatted. Row {COLUMN_NAMES_ROW} should include"
                " 'Weight' column. " + boiler_plate
            )
        )

    if (include_voter_set and "Voter Set" not in header_row) or (
        not include_voter_set and "Voter Set" in header_row
    ):
        raise ValueError(
            (
                "csv file is improperly formatted. Includes Voter Set is not set to the correct"
                f" value given the columns in row {COLUMN_NAMES_ROW}: {header_row}. "
                + boiler_plate
            )
        )


def _validate_score_csv_ballot_score(
    ballot_row: list[str],
    row_index: int,
    candidates: tuple[str],
):
    """
    Validate that the ballot scores are formatted correctly.

    Args:
        ballot_row (list[str]): A ballot row.
        row_index (int): The index of the row in the csv, 0-indexed.
        candidates (tuple[str]): The tuple of candidates.

    Raises:
        ValueError: If the ballot scores are improperly formatted for VoteKit.
    """
    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    break_idx = ballot_row.index("&")

    if len(ballot_row[:break_idx]) != len(candidates):
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} is missing "
                f"some scores: {ballot_row[:break_idx]} ." + boiler_plate
            )
        )

    for score in ballot_row[:break_idx]:
        if score == "":
            continue

        try:
            float(score)
        except ValueError:
            raise ValueError(
                (
                    f"csv file is improperly formatted. Ballot in row {row_index} has "
                    f"non-float score value: {score}. " + boiler_plate
                )
            )


def _validate_score_csv_ballot_rows(csv_data: list[list[str]]):
    """
    Validate that each ballot row is formatted correctly.

    Args:
        csv_data (list[list[str]]): The full csv.

    Raises:
        ValueError: If a row of the csv is improperly formatted for VoteKit.
    """
    candidate_row = csv_data[CANDIDATES_MAPPING_ROW]
    include_voter_set_row = csv_data[VOTER_SET_VALUE_ROW]

    candidate_tuples = [c_tuple.strip("()").split(":") for c_tuple in candidate_row]
    candidates, candidate_prefixes = zip(*candidate_tuples)

    include_voter_set = include_voter_set_row[0] == "True"

    _validate_score_csv_ballot_header_row(
        csv_data[COLUMN_NAMES_ROW:], candidate_prefixes, include_voter_set
    )

    error_rows = []

    for i, ballot_row in enumerate(csv_data[DATA_START_ROW:]):
        try:
            _validate_csv_ballot_row_break_idxs(ballot_row, i + DATA_START_ROW)
            _validate_score_csv_ballot_score(
                ballot_row,
                i + DATA_START_ROW,
                candidates,
            )
            _validate_csv_ballot_weight(ballot_row, i + DATA_START_ROW)
            _validate_csv_ballot_voter_set(
                ballot_row, i + DATA_START_ROW, include_voter_set
            )
        except ValueError as e:
            error_rows.append((i + DATA_START_ROW, e))

        if len(error_rows) == 10:
            raise ValueError(
                (
                    f"There are errors on at least ten lines of this csv: {error_rows}."
                    " Did you try to load a csv that was edited by hand and not "
                    "created with the PreferenceProfile.to_csv() method?"
                )
            )
    if len(error_rows) > 0:
        raise ValueError(
            (
                f"There are errors on {len(error_rows)} lines of this csv: {error_rows}."
                " Did you try to load a csv that was edited by hand and not "
                "created with the PreferenceProfile.to_csv() method?"
            )
        )


def _validate_score_csv_format(csv_data: list[list[str]]):
    """
    Validate that the csv is properly formatted for VoteKit.

    Args:
        csv_data (list[list[str]]): The rows of the csv file.

    Raises:
        ValueError: If csv is improperly formatted for VoteKit.
    """
    _validate_score_csv_header(csv_data[:COLUMN_NAMES_ROW])
    _validate_score_csv_ballot_rows(csv_data)
