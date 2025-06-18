from __future__ import annotations
import csv
import pandas as pd
from ..ballot import Ballot
from .utils import convert_row_to_ballot
import numpy as np
from typing import Optional, Tuple
import warnings
import pickle
from .profile_error import ProfileError
from functools import cached_property
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


class PreferenceProfile:
    """
    PreferenceProfile class, contains ballots and candidates for a given election.
    This is a frozen class, so you need to create a new PreferenceProfile any time
    you want to edit the ballots, candidates, etc.

    Args:
        ballots (tuple[Ballot], optional): Tuple of ``Ballot`` objects. Defaults to empty tuple.
        candidates (tuple[str], optional): Tuple of candidate strings. Defaults to empty tuple.
            If empty, computes this from any candidate listed on a ballot with positive weight.
        max_ranking_length (int, optional): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election. Defaults to longest observed
            ballot.
        contains_rankings (bool, optional): Whether or not the profile contains ballots with
            rankings. If no boolean is provided, then the appropriate boolean value will be
            interpreted from the input preference profile (i.e. if some ballot in the profile has
            a ranking, then this will be set  to `True`).
        contains_scores (bool, optional): Whether or not the profile contains ballots with
            scores. If no boolean is provided, then the appropriate boolean value will be
            interpreted from the input preference profile (i.e. if some ballot in the profile has
            a ranking, then this will be set  to `True`).

    Parameters:
        ballots (tuple[Ballot]): Tuple of ``Ballot`` objects.
        candidates (tuple[str]): Tuple of candidate strings.
        max_ranking_length (int): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election.
        df (pandas.DataFrame): Data frame view of the ballots.
        candidates_cast (tuple[str]): Tuple of candidates who appear on any ballot with positive
            weight, either in the ranking or in the score dictionary.
        total_ballot_wt (float): Sum of ballot weights.
        num_ballots (int): Length of ballot list.
        contains_rankings (bool): Whether or not the profile contains ballots with
            rankings.
        contains_scores (bool): Whether or not the profile contains ballots with
            scores.

    Raises:
        ProfileError: a data frame and ballot list are passed to the init method.
        ProfileError: contains_rankings is set to False but a ballot contains a ranking.
        ProfileError: contains_rankings is set to True but no ballot contains a ranking.
        ProfileError: contains_scores is set to False but a ballot contains a score.
        ProfileError: contains_scores is set to True but no ballot contains a score.
        ProfileError: max_ranking_length is set but a ballot ranking excedes the length.
        ProfileError: a candidate is found on a ballot that is not listed on a provided
            candidate list.
        ProfileError: candidates must be unique.
        ProfileError: candidates must not have names matching ranking columns.

    """

    _is_frozen: bool = False

    def __init__(
        self,
        *,
        ballots: tuple[Ballot, ...] = tuple(),
        candidates: tuple[str, ...] = tuple(),
        max_ranking_length: int = 0,
        df: pd.DataFrame = pd.DataFrame(),
        contains_rankings: Optional[bool] = None,
        contains_scores: Optional[bool] = None,
    ):
        self.candidates = candidates
        self.max_ranking_length = max_ranking_length
        self.contains_rankings = contains_rankings
        self.contains_scores = contains_scores

        if not df.equals(pd.DataFrame()) and ballots != tuple():
            raise ProfileError(
                "Cannot pass a dataframe and a ballot list to profile init method. Must pick one."
            )

        elif df.equals(pd.DataFrame()):
            (
                self.df,
                self.contains_rankings,
                self.contains_scores,
                self.candidates_cast,
            ) = self._init_from_ballots(ballots)

            if self.candidates == tuple():
                self.candidates = self.candidates_cast

        else:
            self.df, self.candidates_cast = self._init_from_df(df)

        self.max_ranking_length = self._find_max_ranking_length()
        self.total_ballot_wt = self._find_total_ballot_wt()
        self.num_ballots = self._find_num_ballots()

        self._validate_candidates()

        self._is_frozen = True

    def __update_ballot_scores_data(
        self,
        ballot_data: dict[str, list],
        idx: int,
        ballot: Ballot,
        candidates_cast: list[str],
        num_ballots: int,
    ) -> None:
        """
        Update the score data from a ballot.

        Args:
            ballot_data (dict[str, list]): Dictionary storing ballot data.
            idx (int): Index of ballot.
            ballot (Ballot): Ballot.
            candidates_cast (list[str]): List of candidates who have received votes.
            num_ballots (int): Total number of ballots.
        """
        if ballot.scores is None:
            return

        if self.contains_scores is False:
            raise ProfileError(
                (
                    f"Ballot {ballot} has scores {ballot.scores} but contains_scores is "
                    "set to False."
                )
            )

        for c, score in ballot.scores.items():
            if ballot.weight > 0 and c not in candidates_cast:
                candidates_cast.append(c)

            if c not in ballot_data:
                if self.candidates:
                    raise ProfileError(
                        f"Candidate {c} found in ballot {ballot} but not in "
                        f"candidate list {self.candidates}."
                    )
                ballot_data[c] = [np.nan] * num_ballots
            ballot_data[c][idx] = score

    def __update_ballot_rankings_data(
        self,
        ballot_data: dict[str, list],
        idx: int,
        ballot: Ballot,
        candidates_cast: list[str],
        num_ballots: int,
    ) -> None:
        """
        Update the ranking data from a ballot.

        Args:
            ballot_data (dict[str, list]): Dictionary storing ballot data.
            idx (int): Index of ballot.
            ballot (Ballot): Ballot.
            candidates_cast (list[str]): List of candidates who have received votes.
            num_ballots (int): Total number of ballots.
        """

        if ballot.ranking is None:
            return
        if self.contains_rankings is False:
            raise ProfileError(
                (
                    f"Ballot {ballot} has ranking {ballot.ranking} but contains_rankings is"
                    " set to False."
                )
            )

        for j, cand_set in enumerate(ballot.ranking):
            for c in cand_set:
                if self.candidates != tuple():
                    if c not in self.candidates:
                        raise ProfileError(
                            f"Candidate {c} found in ballot {ballot} but not in "
                            f"candidate list {self.candidates}."
                        )
                if ballot.weight > 0 and c not in candidates_cast:
                    candidates_cast.append(c)
            if f"Ranking_{j+1}" not in ballot_data:
                if self.max_ranking_length > 0:
                    raise ProfileError(
                        f"Max ballot length {self.max_ranking_length} given but "
                        "ballot {b} has length at least {j+1}."
                    )
                ballot_data[f"Ranking_{j+1}"] = [frozenset("~")] * num_ballots

            ballot_data[f"Ranking_{j+1}"][idx] = cand_set

    def __update_ballot_data_attrs(
        self,
        ballot_data: dict[str, list],
        idx: int,
        ballot: Ballot,
        candidates_cast: list[str],
        num_ballots: int,
    ) -> None:
        """
        Update all ballot data from a ballot.

        Args:
            ballot_data (dict[str, list]): Dictionary storing ballot data.
            idx (int): Index of ballot.
            ballot (Ballot): Ballot.
            candidates_cast (list[str]): List of candidates who have received votes.
            num_ballots (int): Total number of ballots.
        """
        ballot_data["Weight"][idx] = ballot.weight

        if ballot.voter_set != set():
            ballot_data["Voter Set"][idx] = ballot.voter_set

        if ballot.scores is not None:
            self.__update_ballot_scores_data(
                ballot_data=ballot_data,
                idx=idx,
                ballot=ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

        if ballot.ranking is not None:
            self.__update_ballot_rankings_data(
                ballot_data=ballot_data,
                idx=idx,
                ballot=ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

    def __init_ballot_data(
        self, ballots: tuple[Ballot, ...]
    ) -> Tuple[int, dict[str, list]]:
        """
        Create the ballot data objects.

        Args:
            ballots (tuple[Ballot,...]): Tuple of ballots.

        Returns:
            Tuple[int, dict[str, list]]: num_ballots, ballot_data

        """
        num_ballots = len(ballots)

        ballot_data: dict[str, list] = {
            "Weight": [np.nan] * num_ballots,
            "Voter Set": [set()] * num_ballots,
        }

        if self.candidates != tuple():
            ballot_data.update({c: [np.nan] * num_ballots for c in self.candidates})

        if self.max_ranking_length > 0:
            ballot_data.update(
                {
                    f"Ranking_{i+1}": [frozenset("~")] * num_ballots
                    for i in range(self.max_ranking_length)
                }
            )
        return num_ballots, ballot_data

    def __init_formatted_df(
        self,
        ballot_data: dict[str, list],
        contains_scores_indicator: bool,
        candidates_cast: list[str],
    ) -> pd.DataFrame:
        """
        Create a pandas dataframe from the ballot data.

        Args:
            ballot_data (dict[str, list]): Dictionary storing ballot data.
            contains_scores_indicator (bool): Whether or not the profile contains ballots
                with scores.
            candidates_cast (list[str]): List of candidates who received votes.

        Returns:
            pd.DataFrame: Dataframe of profile.
        """
        df = pd.DataFrame(ballot_data)
        temp_col_order = [c for c in df.columns if "Ranking_" in c] + [
            "Voter Set",
            "Weight",
        ]

        if self.candidates != tuple() and contains_scores_indicator:
            col_order = list(self.candidates) + temp_col_order
        elif contains_scores_indicator:
            remaining_cands = set(candidates_cast) - set(df.columns)
            empty_df_cols = np.full((len(df), len(remaining_cands)), np.nan)
            df[list(remaining_cands)] = empty_df_cols

            col_order = (
                sorted([c for c in df.columns if c not in temp_col_order])
                + temp_col_order
            )
        else:
            col_order = temp_col_order
        df = df[col_order]
        df.index.name = "Ballot Index"
        return df

    def _init_from_ballots(
        self, ballots: tuple[Ballot, ...]
    ) -> tuple[pd.DataFrame, bool, bool, tuple[str, ...]]:
        """
        Create the pandas dataframe representation of the profile.

        Args:
            ballots (tuple[Ballot,...]): Tuple of ballots.

        Returns:
            tuple[pd.DataFrame, bool, bool]: df, contains_rankings_indicator,
                contains_scores_indicator

        """
        # `ballot_data` sends {Weight, Voter Set} keys to a list to be
        # indexed in the same order as the output df containing information
        # for each ballot. So ballot_data[<weight>][<index>] is the weight value for
        # the ballot at index <index> in the df.
        num_ballots, ballot_data = self.__init_ballot_data(ballots)

        candidates_cast: list[str] = []
        contains_rankings_indicator = False
        contains_scores_indicator = False

        for i, b in enumerate(ballots):
            contains_scores_indicator = contains_scores_indicator or (
                b.scores is not None
            )
            contains_rankings_indicator = contains_rankings_indicator or (
                b.ranking is not None
            )

            self.__update_ballot_data_attrs(
                ballot_data=ballot_data,
                idx=i,
                ballot=b,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

        df = self.__init_formatted_df(
            ballot_data=ballot_data,
            contains_scores_indicator=contains_scores_indicator,
            candidates_cast=candidates_cast,
        )

        if self.contains_rankings is True and contains_rankings_indicator is False:
            raise ProfileError(
                "contains_rankings is True but we found no ballots with rankings."
            )

        if self.contains_scores is True and contains_scores_indicator is False:
            raise ProfileError(
                "contains_scores is True but we found no ballots with scores."
            )

        return (
            df,
            contains_rankings_indicator,
            contains_scores_indicator,
            tuple(candidates_cast),
        )

    def __validate_init_df_params(self, df: pd.DataFrame) -> None:
        """
        Validate that the correct params were passed to the init method when constructing
        from a dataframe.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Raises:
            ValueError: One of contains_rankings and contains_scores must be True.
            ValueError: If contains_rankings is True, max_ranking_length must be provided.
            ValueError: Candidates must be provided.
        """
        boiler_plate = (
            "When providing a dataframe and no ballot list to the init method, "
        )
        if len(df) == 0:
            return

        if self.contains_rankings is None and self.contains_scores is None:
            raise ValueError(
                boiler_plate
                + "one of contains_rankings and contains_scores must be True."
            )

        elif self.contains_rankings is True and self.max_ranking_length == 0:
            raise ValueError(
                boiler_plate + "if contains_rankings is True, max_ranking_length must"
                " be provided and be non-zero."
            )

        if self.candidates == tuple():
            raise ValueError(boiler_plate + "candidates must be provided.")

    def __validate_init_df(self, df: pd.DataFrame) -> None:
        """
        Validate that the df passed to the init method is of valid type.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Raises:
            ValueError: Candidate column is missing.
            ValueError: Ranking column is missing.
            ValueError: Weight column is missing.
            ValueError: Voter set column is missing.
            ValueError: Index column is misformatted.

        """
        if "Weight" not in df.columns:
            raise ValueError(f"Weight column not in dataframe: {df.columns}")
        if "Voter Set" not in df.columns:
            raise ValueError(f"Voter Set column not in dataframe: {df.columns}")
        if df.index.name != "Ballot Index":
            raise ValueError(f"Index not named 'Ballot Index': {df.index.name}")
        if self.contains_scores:
            if any(c not in df.columns for c in self.candidates):
                for c in self.candidates:
                    if c not in df.columns:
                        raise ValueError(
                            f"Candidate column '{c}' not in dataframe: {df.columns}"
                        )
        if self.contains_rankings:
            if any(
                f"Ranking_{i+1}" not in df.columns
                for i in range(self.max_ranking_length)
            ):
                for i in range(self.max_ranking_length):
                    if f"Ranking_{i+1}" not in df.columns:
                        raise ValueError(
                            f"Ranking column 'Ranking_{i+1}' not in dataframe: {df.columns}"
                        )

    def __find_candidates_cast_from_init_df(self, df: pd.DataFrame) -> tuple[str, ...]:
        """
        Compute which candidates received votes from the df and set the candidates_cast and
        candidates attr.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Returns:
            tuple[str]: Candidates cast.
        """

        mask = df["Weight"] > 0

        candidates_cast: set[str] = set()

        if self.contains_scores:
            positive = df.loc[mask, list(self.candidates)].gt(0).any()
            # .any() applies along the columns, so we get a boolean series where the
            # value is True the candidate has any positive score the column
            candidates_cast |= set(positive[positive].index)

        if self.contains_rankings:
            ranking_cols = [c for c in df.columns if c.startswith("Ranking_")]
            sets = df.loc[mask, ranking_cols].to_numpy().ravel()
            candidates_cast |= set().union(*sets)

        candidates_cast.discard("~")
        return tuple(candidates_cast)

    def _init_from_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """
        Validate the dataframe and determine the candidates cast.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Returns
            tuple[pd.DataFrame, tuple[str]]: df, candidates_cast
        """
        self.__validate_init_df_params(df)
        self.__validate_init_df(df)
        candidates_cast = self.__find_candidates_cast_from_init_df(df)

        if len(df) == 0:
            self.contains_rankings, self.contains_scores, self.max_ranking_length = (
                False,
                False,
                0,
            )

        return df, candidates_cast

    def _find_max_ranking_length(self) -> int:
        """
        Compute and set the maximum ranking length of the profile.

        Returns:
            int: Max ranking length.

        """
        if self.max_ranking_length == 0 and self.contains_rankings is True:
            return len([c for c in self.df.columns if "Ranking_" in c])

        return self.max_ranking_length

    def _find_num_ballots(self) -> int:
        """
        Compute and set the number of ballots.

        Returns:
            int: num ballots
        """
        return len(self.df)

    def _find_total_ballot_wt(self) -> float:
        """
        Compute and set the total ballot weight.

        Returns:
            float: total ballot weight.
        """
        total_weight = 0
        try:
            total_weight = self.df["Weight"].sum()
        except KeyError:
            pass
        return total_weight

    def _validate_candidates(self) -> None:
        """
        Ensure that the candidate names are not equal to the ranking column names, that they are
        unique, and strips whitespace from candidates.

        Raises:
            ProfileError: Candidate names must not be the same as "Ranking_i".
            ProfileError: Candidate names must be unique.
        """
        for cand in self.candidates:
            if any(f"Ranking_{i}" == cand for i in range(len(self.candidates))):
                raise ProfileError(
                    (
                        f"Candidate {cand} must not share name with"
                        " ranking columns: Ranking_i."
                    )
                )
        if not len(set(self.candidates)) == len(self.candidates):
            raise ProfileError("All candidates must be unique.")

        if not set(self.candidates_cast).issubset(self.candidates):
            raise ValueError(
                "Candidates cast are not a subset of candidates list. The following "
                " candidates are in candidates_cast but not candidates: "
                f"{set(self.candidates_cast)-set(self.candidates)}."
            )

        self.candidates = tuple([c.strip() for c in self.candidates])
        self.candidates_cast = tuple([c.strip() for c in self.candidates_cast])

    @cached_property
    def ballots(self: PreferenceProfile) -> tuple[Ballot, ...]:
        """
        Compute the ballot tuple as a cached property.
        """
        computed_ballots = [Ballot()] * len(self.df)
        for i, (_, b_row) in enumerate(self.df.iterrows()):
            computed_ballots[i] = convert_row_to_ballot(
                b_row, self.candidates, self.max_ranking_length
            )
        return tuple(computed_ballots)

    def __setattr__(self, name, value):
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Cannot modify frozen instance: tried to set '{name}'"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Cannot delete attribute '{name}' from frozen instance"
            )
        super().__delattr__(name)

    def __add__(self, other):
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        if isinstance(other, PreferenceProfile):
            ballots = self.ballots + other.ballots
            max_ranking_length = max(
                [self.max_ranking_length, other.max_ranking_length]
            )
            candidates = set(self.candidates).union(other.candidates)
            return PreferenceProfile(
                ballots=ballots,
                max_ranking_length=max_ranking_length,
                candidates=candidates,
            )

        raise TypeError(
            "Unsupported operand type. Must be an instance of PreferenceProfile."
        )

    def group_ballots(self) -> PreferenceProfile:
        """
        Groups ballots by rankings and scores and updates weights. Retains voter sets, but
        loses ballot ids.

        Returns:
            PreferenceProfile: A PreferenceProfile object with grouped ballot list.
        """
        empty_df = pd.DataFrame(columns=["Voter Set", "Weight"], dtype=np.float64)
        empty_df.index.name = "Ballot Index"

        if len(self.df) == 0:
            return PreferenceProfile(
                candidates=self.candidates,
                max_ranking_length=self.max_ranking_length,
            )

        non_group_cols = ["Weight", "Voter Set"]
        ranking_cols = [c for c in self.df.columns if "Ranking_" in c]
        cand_cols = [
            c for c in self.df.columns if c not in non_group_cols + ranking_cols
        ]

        group_df = self.df.groupby(cand_cols + ranking_cols, dropna=False)
        new_df = group_df.aggregate(
            {
                "Weight": "sum",
                "Voter Set": (lambda sets: set().union(*sets)),
            }
        ).reset_index()

        new_df.index.name = "Ballot Index"

        return PreferenceProfile(
            df=new_df,
            candidates=self.candidates,
            max_ranking_length=self.max_ranking_length,
            contains_rankings=self.contains_rankings,
            contains_scores=self.contains_scores,
        )

    def __eq__(self, other):
        if not isinstance(other, PreferenceProfile):
            return False
        if set(self.candidates) != set(other.candidates):
            return False
        if set(self.candidates_cast) != set(other.candidates_cast):
            return False
        if self.total_ballot_wt != other.total_ballot_wt:
            return False
        if self.max_ranking_length != other.max_ranking_length:
            return False
        if self.contains_rankings != other.contains_rankings:
            return False
        if self.contains_scores != other.contains_scores:
            return False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp_1 = self.group_ballots()
            pp_2 = other.group_ballots()
        for b in pp_1.ballots:
            if b not in pp_2.ballots:
                return False
        for b in pp_2.ballots:
            if b not in pp_1.ballots:
                return False
        return True

    def __str__(self) -> str:

        repr_str = f"Profile contains rankings: {self.contains_rankings}\n"
        if self.contains_rankings:
            repr_str += f"Maximum ranking length: {self.max_ranking_length}\n"

        repr_str += (
            f"Profile contains scores: {self.contains_scores}\n"
            f"Candidates: {self.candidates}\n"
            f"Candidates who received votes: {self.candidates_cast}\n"
            f"Total number of Ballot objects: {self.num_ballots}\n"
            f"Total weight of Ballot objects: {self.total_ballot_wt}\n"
        )

        return repr_str

    __repr__ = __str__

    def __to_csv_header(
        self, candidate_mapping: dict[str, str], include_voter_set: bool
    ) -> list[list]:
        """
        Construct the header rows for the PrefProfile a custom CSV format.

        Args:
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
        """
        header = [
            ["VoteKit PreferenceProfile"],
            ["Candidates"],
            [f"({c}:{cand_label})" for c, cand_label in candidate_mapping.items()],
        ]
        header += [["Max Ranking Length"], [str(self.max_ranking_length)]]
        header += [["Includes Voter Set"], [str(include_voter_set)]]
        header += [["="] * 10]

        return header

    def __to_csv_score_list(self, ballot: Ballot) -> list:
        """
        Create the list of score data for a ballot in the profile.

        Args:
            ballot (Ballot): Ballot.

        """
        if ballot.scores is not None:
            return [float(ballot.scores.get(c, 0)) for c in self.candidates]

        return [""] * len(self.candidates)

    def __to_csv_ranking_list(
        self, ballot: Ballot, candidate_mapping: dict[str, str]
    ) -> list:
        """
        Create the list of ranking data for a ballot in the profile.

        Args:
            ballot (Ballot): Ballot.
            candidate_mapping (dict[str, int]): Mapping candidate names to integers.

        """
        if ballot.ranking is not None:
            ranking_list = [
                (
                    set([candidate_mapping[c] for c in cand_set])
                    if cand_set != frozenset()
                    else "{}"
                )
                for cand_set in ballot.ranking
            ]
            if len(ranking_list) != self.max_ranking_length:
                ranking_list += [""] * (self.max_ranking_length - len(ranking_list))

            return ranking_list

        return [""] * self.max_ranking_length

    def __to_csv_ballot_row(
        self, ballot: Ballot, include_voter_set: bool, candidate_mapping: dict[str, str]
    ) -> list[list]:
        """
        Create the row for a ballot in the profile.

        Args:
            ballot (Ballot): Ballot.
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            candidate_mapping (dict[str, int]): Mapping candidate names to integers.

        """
        row = []
        if self.contains_scores:
            row += self.__to_csv_score_list(ballot)
        row += ["&"]

        if self.contains_rankings:
            row += self.__to_csv_ranking_list(ballot, candidate_mapping)
        row += ["&"]

        n, d = ballot.weight.as_integer_ratio()
        row += [f"{n}/{d}", "&"]

        if include_voter_set:
            row += [v for v in sorted(ballot.voter_set)]

        return row

    def __to_csv_data_column_names(
        self, include_voter_set: bool, candidate_mapping: dict[str, str]
    ) -> list:
        """
        Create the data column header.

        Args:
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            candidate_mapping (dict[str, str]): Maps candidate names to prefixes.
        """
        data_col_names = []
        if self.contains_scores:
            data_col_names += [
                f"{cand_label}" for cand_label in candidate_mapping.values()
            ]
        data_col_names += ["&"]
        if self.contains_rankings:
            data_col_names += [f"Ranking_{i+1}" for i in range(self.max_ranking_length)]
        data_col_names += ["&", "Weight", "&"]

        if include_voter_set:
            data_col_names += ["Voter Set"]

        return data_col_names

    def to_csv(self, fpath: str, include_voter_set: bool = False):
        """
        Saves PreferenceProfile to a custom CSV format.

        Args:
            fpath (str): Path to the saved csv.
            include_voter_set (bool, optional): Whether or not to include the voter set of each
                ballot. Defaults to False.
        Raises:
            ProfileError: Cannot write a profile with no ballots to a csv.
            ValueError: File path must be provided.
        """
        if fpath == "":
            raise ValueError("File path must be provided.")

        if len(self.ballots) == 0:
            raise ProfileError("Cannot write a profile with no ballots to a csv.")

        prefix_idx = 1
        candidate_mapping = {c: c[:prefix_idx] for c in self.candidates}
        while len(set(candidate_mapping.values())) < len(candidate_mapping.values()):
            prefix_idx += 1
            candidate_mapping = {c: c[:prefix_idx] for c in self.candidates}

        header = self.__to_csv_header(candidate_mapping, include_voter_set)
        data_col_names = self.__to_csv_data_column_names(
            include_voter_set, candidate_mapping
        )
        ballot_rows = [
            self.__to_csv_ballot_row(b, include_voter_set, candidate_mapping)
            for b in self.ballots
        ]
        rows = header + [data_col_names] + ballot_rows

        with open(
            fpath,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

    @classmethod
    def from_csv(cls, fpath: str) -> PreferenceProfile:
        """
        Creates a PreferenceProfile from a csv, formatted from the ``to_csv`` method.

        Args:
            fpath (str): Path to csv.

        Raises:
            ValueError: If csv is improperly formatted for VoteKit.
            ProfileError: If read profile has no rankings or scores.

        """
        with open(fpath, "r") as file:
            reader = csv.reader(file)
            csv_data = list(reader)

        _validate_csv_format(csv_data)

        (
            inv_candidate_mapping,
            max_ranking_length,
            contains_rankings,
            contains_scores,
            includes_voter_set,
            break_indices,
        ) = _parse_profile_data_from_csv(csv_data)

        if not contains_rankings and not contains_scores:
            raise ProfileError(
                "The profile read from the csv does not contain rankings or scores."
            )

        ballots = [
            _parse_ballot_from_csv(
                row,
                contains_rankings,
                contains_scores,
                includes_voter_set,
                break_indices,
                inv_candidate_mapping,
            )
            for row in csv_data[9:]
        ]

        return cls(
            ballots=tuple(ballots),
            candidates=tuple(inv_candidate_mapping.values()),
            max_ranking_length=max_ranking_length,
            contains_rankings=contains_rankings,
            contains_scores=contains_scores,
        )

    def to_pickle(self, fpath: str):
        """
        Saves profile to pickle file.

        Args:
            fpath (str): File path to save profile to.

        Raises:
            ValueError: File path must be provided.
        """
        if fpath == "":
            raise ValueError("File path must be provided.")
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fpath: str) -> PreferenceProfile:
        """
        Reads profile from pickle file.

        Args:
            fpath (str): File path to profile.
        """

        with open(fpath, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, PreferenceProfile)
        return data
