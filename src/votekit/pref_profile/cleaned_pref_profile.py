from __future__ import annotations

import warnings
from dataclasses import replace

import pandas as pd

from votekit.pref_profile.pref_profile import RankProfile, ScoreProfile


class CleanedRankProfile(RankProfile):
    """
    CleanedRankProfile class, keeps track of how ballots are altered from the original profile.

    In addition to a custom __str__ method, this class implements a collection of sets
    that track the indices of the ballot dataframe, and how they are changed by different cleaning
    rules. It also retains the parent profile of the CleanedRankProfile, allowing for full recovery
    of the cleaning steps.

    Construction copies the index-label list and provenance sets supplied by the caller.

    A ballot is considered altered by a cleaning function when its ranking or weight are not
    equivalent to their values prior to cleaning. A ballot is considered empty when its ranking
    consists of only ``frozenset()`` (empty set), ``frozenset({'~'})`` (special character tilde), or
    mix of both. A ballot with an empty, or "no", ranking can still be altered by a cleaning
    function if the position of ``frozenset()`` and ``frozenset({'~'})`` are not identical to before
    cleaning.

    Example:
        profile = RankProfile(
            ballots=[
                RankBallot(ranking=[{"A"}], weight=1),
                RankBallot(ranking=[{"B"}], weight=0),
                RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
                RankBallot(ranking=[], weight=1),
                ],
            )
        cleaned_profile = remove_cand_rank_profile("A", profile)
        cleaned_profile.df_index_column = [0, 2]
        cleaned_profile.no_wt_altr_idxs = set()
        cleaned_profile.no_rank_altr_idxs = {0}
        cleaned_profile.nonempty_altr_idxs = {2}
        cleaned_profile.unaltr_idxs = {1, 3}

    Args:
        ballots (tuple[Ballot], optional): Tuple of ``Ballot`` objects. Defaults to empty tuple.
        candidates (tuple[Candidate], optional): Tuple of candidates. Defaults to empty tuple.
            Candidates can be strings, integers, or mix of both. If empty, computes this from any
            candidate listed on a ballot with positive weight.
        max_ranking_length (int, optional): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election. Defaults to longest observed
            ballot.
        parent_profile (PreferenceProfile | CleanedProfile): The profile that was altered.
            If you apply multiple cleaning functions, the parent is always the profile immediately
            before cleaning, so you need to recurse to get the original, uncleaned profile.
        df_index_column (list[int]): The indices of the ballots in the df from the parent profile.
        no_wt_altr_idxs (set[int], optional): The indices of ballots whose positive weight is set
            to zero as a result of cleaning. Indices are with respect to ``parent_profile.to_df()``.
        no_rank_altr_idxs (set[int], optional): Indices of ballots whose rankings consist of only
            ``frozenset()`` (empty set), ``frozenset({'~'})`` (special character tilde), or mix of
            both after cleaning, and is not a member of the ``unaltr_idxs`` (ballots identical
            before and after cleaning). Indices are with respect to ``parent_profile.to_df()``.
        nonempty_altr_idxs (set[int], optional):  Indices of ballots that are not a member of
            ``unaltr_idxs`` (ballots identical before and after cleaning) and their ranking contains
            at least one ranking position with ``frozenset(<SET_OF_CANDS>``) where ``<SET_OF_CAND>``
            is a non-empty set of valid candidate identifier(s).
        unaltr_idxs (set[int], optional):  Set of indices of ballots that are identical before and
            after cleaning. Includes dropped ballots whose ranking and weight are unaltered by
            cleaning but their index is removed from the cleaned profile. Indices are with respect
            to ``parent_profile.to_df()``.


    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, **kwargs)

    def __init__(
        self,
        parent_profile: RankProfile | CleanedRankProfile = (RankProfile()),
        df_index_column: list[int] = [],
        no_wt_altr_idxs: set[int] = set(),
        no_rank_altr_idxs: set[int] = set(),
        nonempty_altr_idxs: set[int] = set(),
        unaltr_idxs: set[int] = set(),
        **kwargs,
    ):
        self.parent_profile = parent_profile
        self.df_index_column = df_index_column.copy()
        self.no_wt_altr_idxs = no_wt_altr_idxs.copy()
        self.no_rank_altr_idxs = no_rank_altr_idxs.copy()
        self.nonempty_altr_idxs = nonempty_altr_idxs.copy()
        self.unaltr_idxs = unaltr_idxs.copy()

        self._indices_must_match_parent_df()
        super().__init__(**kwargs)
        self._reindex_matrix()

    def _indices_must_match_parent_df(self) -> None:
        """
        Validate the index sets.

        Raises:
            ValueError: index set is not a subset of the parent df index.
            ValueError: union of indices is not equal to the parent df index.

        """
        parent_data = self.parent_profile._matrix.voter_weight_data
        parent_index = (
            parent_data.ballot_index
            if parent_data.ballot_index is not None
            else range(len(parent_data))
        )
        index_sets = [
            self.no_wt_altr_idxs,
            self.no_rank_altr_idxs,
            self.nonempty_altr_idxs,
            self.unaltr_idxs,
        ]

        if not self.no_wt_altr_idxs.issubset(parent_index):
            set_minus = self.no_wt_altr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "no_wt_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in no_wt_altr_idxs"
                    f" but not the parent profile df index column: {set_minus}"
                )
            )

        if not self.no_rank_altr_idxs.issubset(parent_index):
            set_minus = self.no_rank_altr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "no_rank_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " no_rank_altr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if not self.nonempty_altr_idxs.issubset(parent_index):
            set_minus = self.nonempty_altr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "nonempty_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " nonempty_altr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if not self.unaltr_idxs.issubset(parent_index):
            set_minus = self.unaltr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "unaltr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " unaltr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if set().union(*index_sets) != set(parent_index):
            sym_dif = set().union(*index_sets).symmetric_difference(parent_index)
            raise ValueError(
                (
                    "Union of ballot indices must equal the parent profile df index "
                    f"column. Here are the indices in one but not the other: {sym_dif}"
                )
            )

    def _reindex_matrix(self) -> None:
        """
        Store retained parent ballot indices in the authoritative matrix.

        Raises:
            ValueError: If df_index_column does not have the same length as the ballot list.

        """
        if len(self._matrix) != len(self.df_index_column):
            raise ValueError("df_index_column does not have the same length as the ballot list.")

        data = replace(
            self._matrix.voter_weight_data, ballot_index=pd.Index(self.df_index_column).to_numpy()
        )
        matrix = replace(self._matrix, voter_weight_data=data)
        object.__setattr__(self, "_matrix", matrix)

    def group_ballots(self) -> RankProfile:
        warnings.warn(
            (
                "Grouping the ballots of a CleanedRankProfile will return a RankProfile"
                " since this operation resets ballot indices."
            )
        )
        return super().group_ballots()

    def __str__(self) -> str:
        return "Profile has been cleaned\n" + super().__str__()

    __repr__ = __str__


class CleanedScoreProfile(ScoreProfile):
    """
    CleanedScoreProfile class, which is used to keep track of how ballots are altered from the
    original
    profile. In addition to a custom __str__ method, this class implements a collection of sets
    that track the indices of the ballot dataframe, and how they are changed by different cleaning
    rules. It also retains the parent profile of the CleanedScoreProfile, allowing for full recovery
    of the cleaning steps.

    Construction copies the index-label list and provenance sets supplied by the caller.

    Args:
        ballots (tuple[Ballot], optional): Tuple of ``Ballot`` objects. Defaults to empty tuple.
        candidates (tuple[Candidate], optional): Tuple of candidates. Defaults to empty tuple.
            Candidates can be strings, integers, or mix of both. If empty, computes this from any
            candidate listed on a ballot with positive weight.
        parent_profile (ScoreProfile | CleanedScoreProfile): The profile that was altered.
            If you apply multiple cleaning functions, the parent is always the profile immediately
            before cleaning, so you need to recurse to get the original, uncleaned profile.
        df_index_column (list[int]): The indices of the ballots in the df from the parent profile.
        no_wt_altr_idxs (set[int], optional): Set of indices of ballots that have
            0 weight as a result of cleaning. Indices are with respect
            to ``parent_profile.to_df()``.
        no_scores_altr_idxs (set[int], optional): Set of indices of
            ballots that have no scores as a result of cleaning. Indices are with
            respect to ``parent_profile.to_df()``.
        nonempty_altr_idxs (set[int], optional):  Set of indices of ballots that
            have been altered but still have weight and (ranking or score) as a result of cleaning.
            Indices are with respect to ``parent_profile.to_df()``.
        unaltr_idxs (set[int], optional):  Set of indices of ballots that have
            been unaltered by cleaning. Indices are with respect to ``parent_profile.to_df()``.


    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, **kwargs)

    def __init__(
        self,
        parent_profile: ScoreProfile | CleanedRankProfile = (ScoreProfile()),
        df_index_column: list[int] = [],
        no_wt_altr_idxs: set[int] = set(),
        no_scores_altr_idxs: set[int] = set(),
        nonempty_altr_idxs: set[int] = set(),
        unaltr_idxs: set[int] = set(),
        **kwargs,
    ):
        self.parent_profile = parent_profile
        self.df_index_column = df_index_column.copy()
        self.no_wt_altr_idxs = no_wt_altr_idxs.copy()
        self.no_scores_altr_idxs = no_scores_altr_idxs.copy()
        self.nonempty_altr_idxs = nonempty_altr_idxs.copy()
        self.unaltr_idxs = unaltr_idxs.copy()

        self._indices_must_match_parent_df()
        super().__init__(**kwargs)
        self._reindex_matrix()

    def _indices_must_match_parent_df(self) -> None:
        """
        Validate the index sets.

        Raises:
            ValueError: index set is not a subset of the parent df index.
            ValueError: union of indices is not equal to the parent df index.

        """
        parent_data = self.parent_profile._matrix.voter_weight_data
        parent_index = (
            parent_data.ballot_index
            if parent_data.ballot_index is not None
            else range(len(parent_data))
        )
        index_sets = [
            self.no_wt_altr_idxs,
            self.no_scores_altr_idxs,
            self.nonempty_altr_idxs,
            self.unaltr_idxs,
        ]

        if not self.no_wt_altr_idxs.issubset(parent_index):
            set_minus = self.no_wt_altr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "no_wt_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in no_wt_altr_idxs"
                    f" but not the parent profile df index column: {set_minus}"
                )
            )

        if not self.no_scores_altr_idxs.issubset(parent_index):
            set_minus = self.no_scores_altr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "no_scores_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " no_scores_altr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if not self.nonempty_altr_idxs.issubset(parent_index):
            set_minus = self.nonempty_altr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "nonempty_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " nonempty_altr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if not self.unaltr_idxs.issubset(parent_index):
            set_minus = self.unaltr_idxs.difference(parent_index)
            raise ValueError(
                (
                    "unaltr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " unaltr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if set().union(*index_sets) != set(parent_index):
            sym_dif = set().union(*index_sets).symmetric_difference(parent_index)
            raise ValueError(
                (
                    "Union of ballot indices must equal the parent profile df index "
                    f"column. Here are the indices in one but not the other: {sym_dif}"
                )
            )

    def _reindex_matrix(self) -> None:
        """
        Store retained parent ballot indices in the authoritative matrix.

        Raises:
            ValueError: If df_index_column does not have the same length as the ballot list.

        """
        if len(self._matrix) != len(self.df_index_column):
            raise ValueError("df_index_column does not have the same length as the ballot list.")

        data = replace(
            self._matrix.voter_weight_data, ballot_index=pd.Index(self.df_index_column).to_numpy()
        )
        matrix = replace(self._matrix, voter_weight_data=data)
        object.__setattr__(self, "_matrix", matrix)

    def group_ballots(self) -> ScoreProfile:
        warnings.warn(
            (
                "Grouping the ballots of a CleanedScoreProfile will return a ScoreProfile"
                " since this operation resets ballot indices."
            )
        )
        return super().group_ballots()

    def __str__(self) -> str:
        return "Profile has been cleaned\n" + super().__str__()

    __repr__ = __str__
