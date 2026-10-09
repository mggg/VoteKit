from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from votekit.types import Candidate

EMPTY_VOTER_SET = frozenset()
INDEX_NAME = "Ballot Index"
RANKING_PREFIX = "Ranking_"
TILDE_FROZENSET = frozenset("~")


@dataclass(frozen=True, slots=True)
class ProfileVoterWeightData:
    """
    ProfileVoterWeightData dataclass, holds the weights and voter sets of a profile matrix.



    Parameters:
        weights (np.ndarray): Ballot weights. One entry per ballot. Has dtype float64, or dtype
            object when any weight is a Fraction.
        voter_sets (np.ndarray | None): Voter set per ballot, or None when no ballot has any voters.
        has_fraction_weights (bool): Whether any ballot weight is a Fraction.
    """

    weights: np.ndarray
    voter_sets: np.ndarray | None
    has_fraction_weights: bool

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "ProfileVoterWeightData":
        """
        Extract the weights and voter sets from a profile's df.

        Args:
            df (pd.DataFrame): df with Weight and Voter Set columns.

        Returns:
            ProfileVoterWeightData: Weights and voter sets of the df.
        """
        weights = df["Weight"].to_numpy()
        has_fraction_weights = weights.dtype == object and any(
            isinstance(weight, Fraction) for weight in weights
        )

        voter_set_column = df["Voter Set"]
        voter_sets = (
            voter_set_column.to_numpy() if (voter_set_column != EMPTY_VOTER_SET).any() else None
        )

        return cls(
            weights=weights,
            voter_sets=voter_sets,
            has_fraction_weights=bool(has_fraction_weights),
        )


class ProfileMatrix(Protocol):
    """
    ProfileMatrix Protocol, defines the interface for profile matrices.

    Parameters:
        voter_weight_data (ProfileVoterWeightData): Weights and voter sets of the profile.
    """

    voter_weight_data: ProfileVoterWeightData

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        id_to_cand_set_map: dict[int, frozenset[Candidate]] | dict[int, Candidate],
    ) -> "ProfileMatrix":
        """
        Create a ProfileMatrix from a profile's df and candidate ID mapping.

        Args:
            df (pd.DataFrame): df with Weights and Voter Set columns.
            id_to_cand_set_map (dict[int, frozenset[Candidate]] | dict[int, Candidate]): Maps
                integer IDs to candidate sets or candidates.

        Returns:
            ProfileMatrix: ProfileMatrix with weights and voter sets extracted from the df.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        The number of ballots for this profile.
        """
        raise NotImplementedError

    def to_df(self) -> pd.DataFrame:
        """
        Converts the ProfileMatrix to a df with weights and voter sets.
        """
        raise NotImplementedError


class RankMatrix:
    """
    RankMatrix class, for storing a profile's ballots as arrays.

    Implements the ProfileMatrix Protocol. Constructs a RankMatrix object from a profile's df and
    candidate ID mapping.

    Parameters:
        voter_weight_data (ProfileVoterWeightData): Weights and voter sets of the profile.
        rankings (np.ndarray): num_ballots x max_ranking_length integer array of candidate set IDs.
        id_to_cand_set_map (dict[int, frozenset[Candidate]]): Maps each integer ID to its
            candidate set.
        ranking_columns (list[str]): ``Ranking_i`` labels, in ranking order.
    """

    voter_weight_data: ProfileVoterWeightData

    rankings: np.ndarray
    id_to_cand_set_map: dict[int, frozenset[Candidate]]
    ranking_columns: list[str]

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, id_to_cand_set_map: dict[int, frozenset[Candidate]]
    ) -> "RankMatrix":
        self = cls.__new__(cls)

        self.voter_weight_data = ProfileVoterWeightData.from_df(df)

        self.ranking_columns = sorted(
            [col for col in df.columns if col.startswith(RANKING_PREFIX)],
            key=lambda col: int(col.removeprefix(RANKING_PREFIX)),
        )
        self.rankings = df[self.ranking_columns].to_numpy(dtype=int)
        self.id_to_cand_set_map = id_to_cand_set_map
        return self

    def __len__(self) -> int:
        """
        The number of ballots for this profile.

        Returns:
            int: Number of ballots based on the length of the weights array.
        """
        return len(self.voter_weight_data.weights)

    def max_candidates_ranked(self) -> int:
        """
        The maximum number of candidates ranked on any ballot.

        Counts every candidate in a ranking, so tied candidates each count once. Returns 0 when
        there are no ballots or no ranking columns.

        Returns:
            int: Maximum number of candidates ranked on any ballot.
        """
        if self.rankings.size == 0:
            return 0

        return int(
            np.apply_along_axis(
                lambda row: len(
                    frozenset.union(*(self.id_to_cand_set_map[cand_set_id] for cand_set_id in row))
                    - TILDE_FROZENSET
                ),
                axis=1,
                arr=self.rankings,
            ).max()
        )

    def to_df(self) -> pd.DataFrame:
        """
        Converts the RankMatrix to a df with rankings of candidate sets, not their IDs.

        Returns:
            pd.DataFrame: df with rankings mapped from int IDs to candidate sets.
        """
        # candidate set IDs start at -2 for empty set, -1 for tilde set, and 0+ for candidate sets.
        # Shift by 2 for the reserved negative IDs to index into the lookup array.
        cand_id_lookup = np.empty(max(self.id_to_cand_set_map) + 3, dtype=object)
        for cand_id, cand_set in self.id_to_cand_set_map.items():
            cand_id_lookup[cand_id + 2] = cand_set
        translated_rankings = cand_id_lookup[self.rankings + 2]

        voter_sets = self.voter_weight_data.voter_sets
        if voter_sets is None:
            voter_sets = cast(Any, [EMPTY_VOTER_SET] * len(self))

        df = pd.DataFrame(translated_rankings, columns=self.ranking_columns, copy=False)
        df["Voter Set"] = voter_sets
        df["Weight"] = self.voter_weight_data.weights
        df.index.name = INDEX_NAME
        return df

        def get_all_candidates_ranked_in_row(self, row_index: int) -> list[frozenset[Candidate]]:
            """
            Translate a single row of the rankings array to a list of candidate sets.

            Args:
                row_index (int): Index of the row to translate.

            Returns:
                list[frozenset[Candidate]]: List of candidate sets for the given ballot.
            """
            return [self.id_to_cand_set_map[cand_id] for cand_id in self.rankings[row_index]]


class ScoreMatrix:
    """
    ScoreMatrix class, for storing ``ScoreProfile``'s ballots as arrays.

    Implements the ProfileMatrix Protocol. Constructs a ScoreMatrix object from a profile's df and
    candidate ID mapping.

    Parameters:
        voter_weight_data (ProfileVoterWeightData): Weights and voter sets of the profile.
        scores (np.ndarray): num_ballots x num_candidates float array of scores. Scores are stored
            as floats.
        id_to_cand_set_map (dict[int, Candidate]): Maps each integer ID to its candidate.
        score_columns (list): Candidate IDs, in column order.
    """

    voter_weight_data: ProfileVoterWeightData

    scores: np.ndarray
    id_to_cand_set_map: dict[int, Candidate]
    score_columns: list

    @classmethod
    def from_df(cls, df: pd.DataFrame, id_to_cand_set_map: dict[int, Candidate]) -> "ScoreMatrix":
        self = cls.__new__(cls)

        self.voter_weight_data = ProfileVoterWeightData.from_df(df)

        self.score_columns = [col for col in df.columns if col not in ["Weight", "Voter Set"]]

        self.scores = df[self.score_columns].to_numpy(dtype=float)
        self.id_to_cand_set_map = id_to_cand_set_map
        return self

    def __len__(self) -> int:
        """
        The number of ballots for this profile.

        Returns:
            int: Number of ballots based on the length of the weights array.
        """
        return len(self.voter_weight_data.weights)

    def to_df(self) -> pd.DataFrame:
        """
        Converts the ScoreMatrix to a df with scoring columns with candidate names, not their IDs.

        Returns:
            pd.DataFrame: df with scoring columns mapped from int IDs to candidate names.
        """

        translated_columns = [self.id_to_cand_set_map[col_id] for col_id in self.score_columns]

        voter_sets = self.voter_weight_data.voter_sets
        if voter_sets is None:
            voter_sets = cast(Any, [EMPTY_VOTER_SET] * len(self))

        df = pd.DataFrame(self.scores, columns=translated_columns, copy=False)
        df["Voter Set"] = voter_sets
        df["Weight"] = self.voter_weight_data.weights
        df.index.name = INDEX_NAME
        return df
