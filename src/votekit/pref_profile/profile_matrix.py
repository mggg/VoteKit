from fractions import Fraction
from typing import Any, cast

import numpy as np
import pandas as pd

from votekit.types import Candidate

META_COLUMNS = ("Voter Set", "Weight")


class ProfileMatrix:
    """
    ProfileMatrix parent class, contains the voter sets and weights of a profile's df.

    A profile's ballot data is held as parallel, positionally-aligned arrays rather than a single
    mixed-dtype data frame: a numeric 2D array owned by each subclass, a weight vector, and an
    optional voter set vector. Row ``i`` of every array describes ballot ``i``.

    Parameters:
        weights (np.ndarray): Ballot weights. One entry per ballot. Has dtype as object a weights
            are stored as floats, Fractions, or a mix of both.
        voter_sets (np.ndarray | None): Voter set per ballot, or None when no ballot has any voters.
            None is the common case, e.g. for generated profiles.
        has_fraction_weights (bool): Whether any ballot weight is a Fraction.
        meta_col_order (list): ``Weight`` and ``Voter Set`` in the relative order the data frame
            this matrix was built from had them, so ``to_df`` can return them that way.
    """

    EMPTY_VOTER_SET = set()
    index = "Ballot Index"

    weights: np.ndarray
    voter_sets: np.ndarray | None
    has_fraction_weights: bool
    meta_col_order: list

    def __init__(self, df: pd.DataFrame):
        """
        Extract the weight and voter set columns that are shared across all profile dfs.

        Args:
            df (pd.DataFrame): df with Weights and Voter Set columns.
        """
        self.meta_col_order = [col for col in df.columns if col in META_COLUMNS]
        self.weights = df["Weight"].to_numpy()

        voter_set_column = df["Voter Set"]
        if (voter_set_column != self.EMPTY_VOTER_SET).any():
            self.voter_sets = voter_set_column.to_numpy()
        else:
            self.voter_sets = None

        self.has_fraction_weights = self.weights.dtype == object and any(
            isinstance(weight, Fraction) for weight in self.weights
        )

    def __len__(self) -> int:
        """
        The number of ballots for this profile.

        Based on length of weights which always has a weight for a ballot regardless of whether that
        ballot has a ranking or score.
        """
        return len(self.weights)

    def to_df(self) -> pd.DataFrame:
        raise NotImplementedError


class RankMatrix(ProfileMatrix):
    """
    RankMatrix class, for storing ``RankProfile``'s ballots as arrays.

    Args:
        df (pd.DataFrame): Internal data frame whose ranking columns hold integer candidate set IDs.
        id_candidate_map (dict[int, frozenset[Candidate]]): Maps integer IDs to candidate sets.

    Parameters:
        rankings (np.ndarray): num_ballots x max_ranking_length integer array of candidate set IDs.
        id_cand_set_map (dict[int, frozenset[Candidate]]): Maps each integer ID to its candidate
            set.
        ranking_columns (list[str]): ``Ranking_i`` labels, in ranking order.
    """

    rankings: np.ndarray
    id_cand_set_map: dict[int, frozenset[Candidate]]
    ranking_columns: list[str]

    def __init__(self, df: pd.DataFrame, id_candidate_map: dict[int, frozenset[Candidate]]):
        self.ranking_columns = [col for col in df.columns if "Ranking_" in col]
        self.rankings = df[self.ranking_columns].to_numpy(dtype=int)
        self.id_cand_set_map = id_candidate_map
        super().__init__(df)

    def to_df(self) -> pd.DataFrame:
        """
        Converts the RankMatrix to a df with rankings of candidate sets, not their IDs.

        Returns:
            pd.DataFrame: df with rankings mapped from int IDs to candidate sets.
        """
        # candidate IDs start at -1, need to add 1 to index into lookup array
        cand_id_lookup = np.empty(max(self.id_cand_set_map) + 2, dtype=object)
        for cand_id, cand_set in self.id_cand_set_map.items():
            cand_id_lookup[cand_id + 1] = cand_set
        translated_rankings = cand_id_lookup[self.rankings + 1]

        voter_sets = self.voter_sets
        if voter_sets is None:
            voter_sets = cast(Any, [self.EMPTY_VOTER_SET] * len(self))

        df = pd.DataFrame(translated_rankings, columns=self.ranking_columns, copy=False)
        df["Voter Set"] = voter_sets
        df["Weight"] = self.weights
        df.index.name = self.index
        return df[self.ranking_columns + self.meta_col_order]


class ScoreMatrix(ProfileMatrix):
    """
    ScoreMatrix class, for storing ``ScoreProfile``'s ballots as arrays.

    Args:
        df (pd.DataFrame): Internal data frame whose score columns are labeled with integer
            candidate IDs.
        id_candidate_map (dict[int, Candidate]): Maps integer IDs to candidates.

    Parameters:
        scores (np.ndarray): num_ballots x num_candidates float array of scores. Scores are stored
            as floats.
        id_cand_map (dict[int, Candidate]): Maps each integer ID to its candidate.
        score_columns (list): Candidate IDs, in column order.
    """

    scores: np.ndarray
    id_cand_map: dict[int, Candidate]
    score_columns: list

    def __init__(self, df: pd.DataFrame, id_candidate_map: dict[int, Candidate]):
        self.score_columns = [col for col in df.columns if col not in META_COLUMNS]

        self.scores = df[self.score_columns].to_numpy(dtype=float)
        self.id_cand_map = id_candidate_map
        super().__init__(df)

    def to_df(self) -> pd.DataFrame:
        """
        Converts the ScoreMatrix to a df with scoring columns with candidate names, not their IDs.

        Returns:
            pd.DataFrame: df with scoring columns mapped from int IDs to candidate names.
        """

        translated_columns = [self.id_cand_map[col_id] for col_id in self.score_columns]

        voter_sets = self.voter_sets
        if voter_sets is None:
            voter_sets = cast(Any, [self.EMPTY_VOTER_SET] * len(self))

        df = pd.DataFrame(self.scores, columns=translated_columns, copy=False)
        df["Voter Set"] = voter_sets
        df["Weight"] = self.weights
        df.index.name = self.index
        return df[translated_columns + self.meta_col_order]
