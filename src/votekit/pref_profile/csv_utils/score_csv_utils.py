from __future__ import annotations
from ...ballot import Ballot
from typing import Tuple
import ast


def _parse_profile_data_from_csv(
    csv_data: list[list[str]],
) -> Tuple[dict[str, str], int, bool, bool, bool, list[int]]:
    """
    Parse the profile data from a PreferenceProfile csv.

    Args:
        csv_data (list[list[str]]): Data from csv.

    Returns:
        Tuple[dict[str, str], int, bool, bool, bool, list[int]]:
            inv_candidate_mapping, max_ranking_length, contains_rankings, contains_scores,
            includes_voter_set, break_indices
    """
    candidate_row = [c_tuple.strip("()").split(":") for c_tuple in csv_data[2]]
    inv_candidate_mapping = {prefix: cand for cand, prefix in candidate_row}

    max_ranking_length = int(csv_data[4][0])
    contains_rankings = max_ranking_length > 0

    includes_voter_set = csv_data[6][0] == "True"

    ballot_data_column_names = csv_data[8]
    contains_scores = list(inv_candidate_mapping.keys())[0] in ballot_data_column_names
    break_indices = [
        i for i, col_name in enumerate(ballot_data_column_names) if col_name == "&"
    ]

    return (
        inv_candidate_mapping,
        max_ranking_length,
        contains_rankings,
        contains_scores,
        includes_voter_set,
        break_indices,
    )


def _parse_ballot_from_csv(
    ballot_row: list[str],
    contains_rankings: bool,
    contains_scores: bool,
    includes_voter_set: bool,
    break_indices: list[int],
    inv_candidate_mapping: dict[str, str],
) -> Ballot:
    """
    Parse a ballot from a PreferenceProfile csv row.

    Args:
        ballot_row (list[str]): Row from the csv file containing ballot data.
        contains_rankings (bool): Whether or not the csv contains rankings.
        contains_scores (bool): Whether or not the csv contains scores.
        includes_voter_set (bool): Whether or not the csv contains voter sets.
        break_indices (list[int]): Where the columns of the csv change from one data type to
            another.
        inv_candidate_mapping (dict[str, str]): The iverted candidate mapping of prefix
            to the cand.

    Returns:
        Ballot: Ballot formatted from row of csv.
    """
    candidates = list(inv_candidate_mapping.values())
    ranking_start = break_indices[0] + 1
    ranking_end = break_indices[1]

    scores = None
    formatted_ranking = None
    voter_set = set()

    try:
        num, denom = ballot_row[break_indices[1] + 1].split("/")
        num = ast.literal_eval(num)
        denom = ast.literal_eval(denom)
    except Exception:
        raise RuntimeError(
            f"Invalid weight format in ballot row: {ballot_row[break_indices[1] + 1]}"
        )

    weight = float(num) / float(denom)

    if contains_scores:
        scores = {
            c: float(ballot_row[i]) for i, c in enumerate(candidates) if ballot_row[i]
        }

    if contains_rankings:
        ranking = ballot_row[ranking_start:ranking_end]
        temp_ranking = [
            cand_set.strip("{}").split(", ") for cand_set in ranking if cand_set != ""
        ]
        formatted_ranking = tuple(
            [
                (
                    frozenset(inv_candidate_mapping[c.strip("'")] for c in cand_set)
                    if cand_set != [""]
                    else frozenset()
                )
                for cand_set in temp_ranking
            ]
        )

    if includes_voter_set:
        voter_set = set(v.strip() for v in ballot_row[break_indices[-1] + 1 :])

    return Ballot(
        ranking=formatted_ranking, scores=scores, voter_set=voter_set, weight=weight
    )


def _validate_csv_header_values(header_data: list[list[str]]):
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
                "csv file is improperly formatted. Row 2 should contain tuples mapping candidates "
                "to their unique prefixes. For example, (Chris:Ch), (Colleen: Co). "
                f"Not {header_data[2]}. " + boiler_plate
            )
        )

    if len(header_data[4]) != 1:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 4 should be a single non-negative integer, "
                f"not {header_data[4]}. " + boiler_plate
            )
        )

    try:
        max_ranking_length = int(header_data[4][0])

        if max_ranking_length < 0:
            raise ValueError(
                (
                    "csv file is improperly formatted. Row 4 should be a single"
                    f" non-negative integer, not {max_ranking_length}. " + boiler_plate
                )
            )
    except ValueError:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 4 should be a single"
                f" non-negative integer, not {header_data[4][0]}. " + boiler_plate
            )
        )

    if len(header_data[6]) != 1 or header_data[6][0] not in ["True", "False"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 6 should be 'True' or 'False',"
                f" not {header_data[6]}. " + boiler_plate
            )
        )


def _validate_csv_header_rows(header_data: list[list[str]]):
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

    if header_data[-1] != ["="] * 10:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 7 should be '=,=,=,=,=,=,=,=,=,=', "
                f"not {header_data[-1]}. " + boiler_plate
            )
        )

    if header_data[0] != ["VoteKit PreferenceProfile"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 0 should be 'VoteKit PreferenceProfile',"
                f"not {header_data[0]}. " + boiler_plate
            )
        )

    if header_data[1] != ["Candidates"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 1 should be 'Candidates', "
                f"not {header_data[1]}. " + boiler_plate
            )
        )

    if header_data[3] != ["Max Ranking Length"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 3 should be 'Max Ranking Length', "
                f"not {header_data[3]}. " + boiler_plate
            )
        )

    if header_data[5] != ["Includes Voter Set"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 5 should be 'Includes Voter Set', "
                f"not {header_data[5]}. " + boiler_plate
            )
        )


def _validate_csv_header(header_data: list[list[str]]):
    """
    Validate that the values of the header rows are correct.

    Args:
        header_data (list[list[str]]): The rows of the header of the csv file.

    Raises:
        ValueError: If csv header is improperly formatted for VoteKit.
    """

    _validate_csv_header_rows(header_data)
    _validate_csv_header_values(header_data)


def _validate_csv_ballot_header_row(
    ballot_rows: list[list[str]],
    candidate_prefixes: tuple[str],
    max_ranking_length: int,
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
    ranking_cols = [f"Ranking_{i+1}" for i in range(max_ranking_length)]

    if max_ranking_length > 0 and any(
        r_col not in header_row for r_col in ranking_cols
    ):
        raise ValueError(
            (
                "csv file is improperly formatted. Row 8 should include 'Ranking_i' for "
                f"i going from 1 to max ranking length, not {header_row}. "
                + boiler_plate
            )
        )
    elif max_ranking_length == 0 and any("Ranking_" in col for col in header_row):
        raise ValueError(
            (
                "csv file is improperly formatted. Row 8 should not include 'Ranking_', "
                f"not {header_row}. " + boiler_plate
            )
        )

    break_indices = [i for i, col_name in enumerate(header_row) if col_name == "&"]

    if len(header_row[: break_indices[0]]) > 0 and any(
        c not in header_row for c in candidate_prefixes
    ):
        raise ValueError(
            (
                "csv file is improperly formatted. Row 8 should include all candidates before "
                "the first &. " + boiler_plate
            )
        )

    if "Weight" not in header_row:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 8 should include 'Weight' column. "
                + boiler_plate
            )
        )

    if (include_voter_set and "Voter Set" not in header_row) or (
        not include_voter_set and "Voter Set" in header_row
    ):
        raise ValueError(
            (
                "csv file is improperly formatted. Includes Voter Set is not set to the correct"
                f" value given the columns in row 8: {header_row}. " + boiler_plate
            )
        )


def _validate_csv_ballot_score(
    ballot_row: list[str], row_index: int, candidates: tuple[str], contains_scores: bool
):
    """
    Validate that the ballot scores are formatted correctly.

    Args:
        ballot_row (list[str]): A ballot row.
        row_index (int): The index of the row in the csv, 0-indexed.
        candidates (tuple[str]): The tuple of candidates.
        contains_scores (bool): Whether or not the profile contains scores.

    Raises:
        ValueError: If the ballot scores are improperly formatted for VoteKit.
    """
    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    break_idx = ballot_row.index("&")

    if not contains_scores and break_idx > 0:
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} has scores "
                f"but it should not: {ballot_row[:break_idx]}. " + boiler_plate
            )
        )

    elif not contains_scores and break_idx == 0:
        return

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


def _validate_csv_ballot_ranking(
    ballot_row: list[str],
    row_index: int,
    candidate_prefixes: tuple[str],
    max_ranking_length: int,
):
    """
    Validate that the ballot ranking is formatted correctly.

    Args:
        ballot_row (list[str]): A ballot row.
        row_index (int): The index of the row in the csv, 0-indexed.
        candidate_prefixes (tuple[str]): The tuple of candidate prefixes.
        max_ranking_length (int): The max ranking length.

    Raises:
        ValueError: If the ballot ranking is improperly formatted for VoteKit.
    """

    boiler_plate = (
        "This usually indicates that you are loading a csv that was not made with "
        "PreferenceProfile.to_csv()."
    )

    break_idxs = [i for i, string in enumerate(ballot_row) if string == "&"]
    if max_ranking_length == 0 and break_idxs[1] - break_idxs[0] > 1:
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} has rankings "
                f"but it should not: {ballot_row[break_idxs[0]:break_idxs[1]]}. "
                + boiler_plate
            )
        )
    elif max_ranking_length == 0:
        return

    for ranking_set in ballot_row[break_idxs[0] + 1 : break_idxs[1]]:
        if ranking_set == "":
            continue

        if ranking_set[0] != "{" or ranking_set[-1] != "}":
            raise ValueError(
                (
                    f"csv file is improperly formatted. Ballot in row {row_index} has "
                    f"improperly formatted ranking set: {ranking_set}. " + boiler_plate
                )
            )

        if ranking_set in ["", "{}"]:
            continue

        cand_set = ranking_set.strip("{}").split(", ")
        for c in cand_set:
            if c.strip("'") not in candidate_prefixes:
                raise ValueError(
                    (
                        f"csv file is improperly formatted. Ballot in row {row_index} has "
                        f"undefined candidate prefix: {c}. " + boiler_plate
                    )
                )


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
    if break_idxs[2] - break_idxs[1] != 2:
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} has a weight"
                " entry that is too long or short. " + boiler_plate
            )
        )

    if "/" not in ballot_row[break_idxs[1] + 1]:
        raise ValueError(
            (
                f"csv file is improperly formatted. Ballot in row {row_index} has a "
                f"weight entry that is not a fraction: {ballot_row[break_idxs[1] + 1]} "
                + boiler_plate
            )
        )

    else:
        num, denom = ballot_row[break_idxs[1] + 1].split("/")

        try:
            int(num)
            int(denom)

        except ValueError:
            raise ValueError(
                (
                    f"csv file is improperly formatted. Ballot in row {row_index} has a "
                    f"weight entry with non-integer numerator or denominator: {num}/{denom}. "
                    + boiler_plate
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


def _validate_csv_ballot_rows(csv_data: list[list[str]]):
    """
    Validate that each ballot row is formatted correctly.

    Args:
        csv_data (list[list[str]]): The full csv.

    Raises:
        ValueError: If a row of the csv is improperly formatted for VoteKit.
    """
    candidate_row = csv_data[2]
    max_ranking_row = csv_data[4]
    include_voter_set_row = csv_data[6]

    candidate_tuples = [c_tuple.strip("()").split(":") for c_tuple in candidate_row]
    candidates, candidate_prefixes = zip(*candidate_tuples)

    max_ranking_length = int(max_ranking_row[0])
    include_voter_set = include_voter_set_row[0] == "True"

    _validate_csv_ballot_header_row(
        csv_data[8:], candidate_prefixes, max_ranking_length, include_voter_set
    )

    contains_scores = any(c in csv_data[8] for c in candidate_prefixes)

    error_rows = []
    for i, ballot_row in enumerate(csv_data[9:]):
        try:
            _validate_csv_ballot_score(ballot_row, i + 9, candidates, contains_scores)
            _validate_csv_ballot_ranking(
                ballot_row, i + 9, candidate_prefixes, max_ranking_length
            )
            _validate_csv_ballot_weight(ballot_row, i + 9)
            _validate_csv_ballot_voter_set(ballot_row, i + 9, include_voter_set)
        except ValueError as e:
            error_rows.append((i + 9, e))

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


def _validate_csv_format(csv_data: list[list[str]]):
    """
    Validate that the csv is properly formatted for VoteKit.

    Args:
        csv_data (list[list[str]]): The rows of the csv file.

    Raises:
        ValueError: If csv is improperly formatted for VoteKit.
    """
    _validate_csv_header(csv_data[:8])
    _validate_csv_ballot_rows(csv_data)
