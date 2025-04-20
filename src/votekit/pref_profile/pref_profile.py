from __future__ import annotations
import csv
from fractions import Fraction
import pandas as pd
from pydantic import ConfigDict, field_validator, model_validator
from ..ballot import Ballot
from .utils import _df_to_ballot_tuple
from pydantic.dataclasses import dataclass
from typing_extensions import Self
from dataclasses import field
import numpy as np
from typing import Optional, Tuple
import warnings
import pickle
from .profile_error import ProfileError


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

    num, denom = ballot_row[break_indices[1] + 1].split("/")
    weight = Fraction(int(num), int(denom))

    if contains_scores:
        scores = {
            c: Fraction(float(ballot_row[i]))
            for i, c in enumerate(candidates)
            if ballot_row[i]
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
        voter_set = set(ballot_row[break_indices[-1] + 1 :])

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
                "to their unique prefixes. For example, (Chris:Ch), (Colleen: Co)."
                + boiler_plate
            )
        )

    if len(header_data[4]) != 1:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 4 should be a single non-negative integer. "
                + boiler_plate
            )
        )
    else:
        try:
            max_ranking_length = int(header_data[4][0])

            if max_ranking_length < 0:
                raise ValueError(
                    (
                        "csv file is improperly formatted. Row 4 should be a single"
                        " non-negative integer. " + boiler_plate
                    )
                )
        except ValueError:
            raise ValueError(
                (
                    "csv file is improperly formatted. Row 4 should be a single"
                    " non-negative integer. " + boiler_plate
                )
            )

    if len(header_data[6]) != 1 or header_data[6][0] not in ["True", "False"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 6 should be 'True' or 'False'. This "
                + boiler_plate
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
                "csv file is improperly formatted. Row 7 should be '=,=,=,=,=,=,=,=,=,='. This "
                + boiler_plate
            )
        )

    if header_data[0] != ["VoteKit PreferenceProfile"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 0 should be 'VoteKit PreferenceProfile'."
                + boiler_plate
            )
        )

    if header_data[1] != ["Candidates"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 1 should be 'Candidates'. This "
                + boiler_plate
            )
        )

    if header_data[3] != ["Max Ranking Length"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 3 should be 'Max Ranking Length'. This "
                + boiler_plate
            )
        )

    if header_data[5] != ["Includes Voter Set"]:
        raise ValueError(
            (
                "csv file is improperly formatted. Row 5 should be 'Includes Voter Set'. This "
                + boiler_plate
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
                "i going from 1 to max ranking length. " + boiler_plate
            )
        )
    elif max_ranking_length == 0 and any("Ranking_" in col for col in header_row):
        raise ValueError(
            (
                "csv file is improperly formatted. Row 8 should not include 'Ranking_'. "
                + boiler_plate
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
                " value given the columns in row 8. " + boiler_plate
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
                "but it should not. " + boiler_plate
            )
        )

    elif contains_scores:
        if len(ballot_row[:break_idx]) != len(candidates):
            raise ValueError(
                (
                    f"csv file is improperly formatted. Ballot in row {row_index} is missing "
                    "some scores. " + boiler_plate
                )
            )

        for score in ballot_row[:break_idx]:
            if score:
                try:
                    float(score)
                except ValueError:
                    raise ValueError(
                        (
                            f"csv file is improperly formatted. Ballot in row {row_index} has "
                            "non-float score value." + boiler_plate
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
                "but it should not. " + boiler_plate
            )
        )
    elif max_ranking_length > 0:
        for ranking_set in ballot_row[break_idxs[0] + 1 : break_idxs[1]]:
            if ranking_set:
                if ranking_set[0] != "{" or ranking_set[-1] != "}":
                    raise ValueError(
                        (
                            f"csv file is improperly formatted. Ballot in row {row_index} has "
                            "improperly formatted ranking sets. " + boiler_plate
                        )
                    )

            if ranking_set not in ["", "{}"]:
                cand_set = ranking_set.strip("{}").split(", ")
                for c in cand_set:
                    if c.strip("'") not in candidate_prefixes:
                        raise ValueError(
                            (
                                f"csv file is improperly formatted. Ballot in row {row_index} has "
                                "undefined candidate prefix. " + boiler_plate
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
    else:
        if "/" not in ballot_row[break_idxs[1] + 1]:
            raise ValueError(
                (
                    f"csv file is improperly formatted. Ballot in row {row_index} has a "
                    "weight entry that is not a fraction. " + boiler_plate
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
                        "weight entry with non-integer numerator or denominator. "
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
                "voter set but it should not. " + boiler_plate
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

    # TODO add try/except and show first 10 row errors
    for i, ballot_row in enumerate(csv_data[9:]):
        _validate_csv_ballot_score(ballot_row, i + 9, candidates, contains_scores)
        _validate_csv_ballot_ranking(
            ballot_row, i + 9, candidate_prefixes, max_ranking_length
        )
        _validate_csv_ballot_weight(ballot_row, i + 9)
        _validate_csv_ballot_voter_set(ballot_row, i + 9, include_voter_set)


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


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
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
        total_ballot_wt (Fraction): Sum of ballot weights.
        num_ballots (int): Length of ballot list.
        contains_rankings (bool): Whether or not the profile contains ballots with
            rankings.
        contains_scores (bool): Whether or not the profile contains ballots with
            scores.

    Raises:
        ProfileError: contains_rankings is set to False but a ballot contains a ranking.
        ProfileError: contains_rankings is set to True but no ballot contains a ranking.
        ProfileError: contains_scores is set to False but a ballot contains a score.
        ProfileError: contains_scores is set to True but no ballot contains a score.
        ProfileError: max_ranking_length is set but a ballot ranking excedes the length.
        ProfileError: a candidate is found on a ballot that is not listed on a provided
            candidate list.
        ProfileError: candidates must be unique.
        ProfileError: candidates must not have names matching ranking columns.

    Warns:
        UserWarning: max_ranking_length is set but contains_rankings is False.
            Sets max_ranking_length to 0.

    """

    ballots: tuple[Ballot, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)
    max_ranking_length: int = 0
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates_cast: tuple[str, ...] = field(default_factory=tuple)
    num_ballots: int = 0
    total_ballot_wt: Fraction = Fraction(0)
    contains_rankings: Optional[bool] = None
    contains_scores: Optional[bool] = None

    @field_validator("candidates")
    @classmethod
    def cands_must_be_unique(
        cls, candidates: Optional[tuple[str, ...]]
    ) -> Optional[tuple[str, ...]]:
        """
        Checks that candidate names are unique.

        Args:
            candidates (tuple[str], optional): Candidate names.

        Returns:
            Optional[tuple[str]]: Candidate names.
        """
        if candidates:
            if not len(set(candidates)) == len(candidates):
                raise ProfileError("All candidates must be unique.")
        return candidates

    @model_validator(mode="after")
    def cands_not_ranking_columns(
        self,
    ) -> Self:
        """
        Ensures that candidate names do not match ranking columns.
        Also added protection for from_csv method.

        """
        for cand in self.candidates:
            if any(f"Ranking_{i}" == cand for i in range(len(self.candidates))):
                raise ProfileError(
                    (
                        f"Candidate {cand} must not share name with"
                        " ranking columns: Ranking_i."
                    )
                )
        return self

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
        if ballot.scores:
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

        if ballot.ranking:
            if self.contains_rankings is False:
                raise ProfileError(
                    (
                        f"Ballot {ballot} has ranking {ballot.ranking} but contains_rankings is"
                        " set to False."
                    )
                )

            for j, cand_set in enumerate(ballot.ranking):
                for c in cand_set:
                    if self.candidates:
                        if c not in self.candidates:
                            raise ProfileError(
                                f"Candidate {c} found in ballot {ballot} but not in "
                                f"candidate list {self.candidates}."
                            )
                    if ballot.weight > 0 and c not in candidates_cast:
                        candidates_cast.append(c)
                if f"Ranking_{j+1}" not in ballot_data:
                    if self.max_ranking_length:
                        raise ProfileError(
                            f"Max ballot length {self.max_ranking_length} given but "
                            "ballot {b} has length at least {j+1}."
                        )
                    ballot_data[f"Ranking_{j+1}"] = [np.nan] * num_ballots

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

        if ballot.voter_set:
            ballot_data["Voter Set"][idx] = ballot.voter_set

        if ballot.scores:
            self.__update_ballot_scores_data(
                ballot_data=ballot_data,
                idx=idx,
                ballot=ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

        if ballot.ranking:
            self.__update_ballot_rankings_data(
                ballot_data=ballot_data,
                idx=idx,
                ballot=ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

    def __init_ballot_data(self) -> Tuple[int, dict[str, list]]:
        """
        Create the ballot data objects.

        Returns:
            Tuple[int, dict[str, list]]: num_ballots, ballot_data

        """
        num_ballots = len(self.ballots)

        ballot_data: dict[str, list] = {
            "Weight": [np.nan] * num_ballots,
            "Voter Set": [set()] * num_ballots,
        }

        if self.candidates:
            ballot_data.update({c: [np.nan] * num_ballots for c in self.candidates})

        if self.max_ranking_length:
            ballot_data.update(
                {
                    f"Ranking_{i+1}": [np.nan] * num_ballots
                    for i in range(self.max_ranking_length)
                }
            )
        return num_ballots, ballot_data

    def __init_formatted_df(
        self,
        ballot_data: dict[str, list],
        contains_scores_indicator: bool,
    ) -> pd.DataFrame:
        """
        Create a pandas dataframe from the ballot data.

        Args:
            ballot_data (dict[str, list]): Dictionary storing ballot data.
            contains_scores_indicator (bool): Whether or not the profile contains ballots
                with scores.

        Returns:
            pd.DataFrame: Dataframe of profile.
        """
        df = pd.DataFrame(ballot_data)
        temp_col_order = [c for c in df.columns if "Ranking_" in c] + [
            "Voter Set",
            "Weight",
        ]

        if self.candidates and contains_scores_indicator:
            col_order = list(self.candidates) + temp_col_order
        elif contains_scores_indicator:
            col_order = (
                sorted([c for c in df.columns if c not in temp_col_order])
                + temp_col_order
            )
        else:
            col_order = temp_col_order
        df = df[col_order]
        df.index.name = "Ballot Index"
        return df

    def __set_class_attrs_from_df(
        self,
        df: pd.DataFrame,
        candidates_cast: list[str],
        contains_rankings_indicator: bool,
        contains_scores_indicator: bool,
    ) -> Self:
        """
        Set various class attributes from the pandas dataframe representation of the profile.

        Args:
            df (pd.DataFrame): The dataframe representation of the profile.
            candidates_cast (list[str]): Candidates who received votes.
            contains_rankings_indicator (bool): Whether or not the profile contains ballots
                with rankings.
            contains_scores_indicator (bool): Whether or not the profile contains ballots
                with scores.
        """
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "candidates_cast", tuple(candidates_cast))
        if not self.candidates:
            object.__setattr__(self, "candidates", tuple(candidates_cast))

        if self.contains_rankings is None:
            object.__setattr__(self, "contains_rankings", contains_rankings_indicator)
        elif self.contains_rankings and not contains_rankings_indicator:
            raise ProfileError(
                "contains_rankings is True but we found no ballots with rankings."
            )

        if self.contains_scores is None:
            object.__setattr__(self, "contains_scores", contains_scores_indicator)
        elif self.contains_scores and not contains_scores_indicator:
            raise ProfileError(
                "contains_scores is True but we found no ballots with scores."
            )

        return self

    @model_validator(mode="after")
    def ballot_list_to_df(self) -> Self:
        """
        Create the pandas dataframe representation of the profile.

        """
        # `ballot_data` sends {Weight, Voter Set} keys to a list to be
        # indexed in the same order as the output df containing information
        # for each ballot. So ballot_data[<weight>][<index>] is the weight value for
        # the ballot at index <index> in the df.
        num_ballots, ballot_data = self.__init_ballot_data()

        candidates_cast: list[str] = []
        contains_rankings_indicator = False
        contains_scores_indicator = False

        for i, b in enumerate(self.ballots):
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
        )

        return self.__set_class_attrs_from_df(
            df=df,
            candidates_cast=candidates_cast,
            contains_rankings_indicator=contains_rankings_indicator,
            contains_scores_indicator=contains_scores_indicator,
        )

    @model_validator(mode="after")
    def find_max_ranking_length(self) -> Self:
        """
        Compute and set the maximum ranking length of the profile.

        Warns:
            UserWarning: If a max_ranking_length is provided but not rankings are in the
                profile, we set the max_ranking_length to 0.
        """
        if self.max_ranking_length > 0 and not self.contains_rankings:
            warnings.warn(
                "Profile does not contain rankings but "
                f"max_ranking_length={self.max_ranking_length}. Setting max_ranking_length"
                " to 0."
            )

            object.__setattr__(self, "max_ranking_length", 0)

        elif not self.max_ranking_length and self.contains_rankings:
            max_ranking_length = len([c for c in self.df.columns if "Ranking_" in c])
            object.__setattr__(self, "max_ranking_length", max_ranking_length)

        return self

    @model_validator(mode="after")
    def find_num_ballots(self) -> Self:
        """
        Compute and set the number of ballots.
        """
        object.__setattr__(self, "num_ballots", len(self.df))
        return self

    @model_validator(mode="after")
    def find_total_ballot_wt(self) -> Self:
        """
        Compute and set the total ballot weight.
        """
        object.__setattr__(self, "total_ballot_wt", self.df["Weight"].sum())

        return self

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

        else:
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

        new_ballots = _df_to_ballot_tuple(
            new_df,
            candidates=self.candidates,
            max_ranking_length=self.max_ranking_length,
        )

        return PreferenceProfile(
            ballots=new_ballots,
            candidates=self.candidates,
            max_ranking_length=self.max_ranking_length,
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
        if ballot.scores:
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
        if ballot.ranking:
            ranking_list = [
                set([candidate_mapping[c] for c in cand_set]) if cand_set else "{}"
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
