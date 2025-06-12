from __future__ import annotations
from .pref_profile import PreferenceProfile
import warnings
import pandas as pd


class CleanedProfile(PreferenceProfile):
    """
    CleanedProfile class, which is used to keep track of how ballots are altered from the original
    profile. In addition to a custom __str__ method, this class implements a collection of sets
    that track the indices of the ballot dataframe, and how they are changed by different cleaning
    rules. It also retains the parent profile of the CleanedProfile, allowing for full recovery
    of the cleaning steps.

    Args:
        ballots (tuple[Ballot], optional): Tuple of ``Ballot`` objects. Defaults to empty tuple.
        candidates (tuple[str], optional): Tuple of candidate strings. Defaults to empty tuple.
            If empty, computes this from any candidate listed on a ballot with positive weight.
        max_ranking_length (int, optional): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election. Defaults to longest observed
            ballot.
        parent_profile (PreferenceProfile | CleanedProfile): The profile that was altered.
            If you apply multiple cleaning functions, the parent is always the profile immediately
            before cleaning, so you need to recurse to get the original, uncleaned profile.
        df_index_column (list[int]): The indices of the ballots in the df from the parent profile.
        no_wt_altr_idxs (set[int], optional): Set of indices of ballots that have
            0 weight as a result of cleaning. Indices are with respect
            to ``parent_profile.df``.
        no_rank_no_score_altr_idxs (set[int], optional): Set of indices of
            ballots that have no ranking and no scores as a result of cleaning. Indices are with
            respect to ``parent_profile.df``.
        nonempty_altr_idxs (set[int], optional):  Set of indices of ballots that
            have been altered but still have weight and (ranking or score) as a result of cleaning.
            Indices are with respect to ``parent_profile.df``.
        unaltr_idxs (set[int], optional):  Set of indices of ballots that have
            been unaltered by cleaning. Indices are with respect to ``parent_profile.df``.


    """

    def __init__(
        self,
        parent_profile: PreferenceProfile | CleanedProfile = (PreferenceProfile()),
        df_index_column: list[int] = [],
        no_wt_altr_idxs: set[int] = set(),
        no_rank_no_score_altr_idxs: set[int] = set(),
        nonempty_altr_idxs: set[int] = set(),
        unaltr_idxs: set[int] = set(),
        **kwargs,
    ):

        self.parent_profile = parent_profile
        self.df_index_column = df_index_column
        self.no_wt_altr_idxs = no_wt_altr_idxs
        self.no_rank_no_score_altr_idxs = no_rank_no_score_altr_idxs
        self.nonempty_altr_idxs = nonempty_altr_idxs
        self.unaltr_idxs = unaltr_idxs

        self._indices_must_match_parent_df()
        super().__init__(**kwargs)
        self._reindex_ballot_df()

    def _indices_must_match_parent_df(self) -> None:
        """
        Validate the index sets.

        Raises:
            ValueError: index set is not a subset of the parent df index.
            ValueError: union of indices is not equal to the parent df index.

        """
        index_sets = [
            self.no_wt_altr_idxs,
            self.no_rank_no_score_altr_idxs,
            self.nonempty_altr_idxs,
            self.unaltr_idxs,
        ]

        if not self.no_wt_altr_idxs.issubset(self.parent_profile.df.index):
            set_minus = self.no_wt_altr_idxs.difference(self.parent_profile.df.index)
            raise ValueError(
                (
                    "no_wt_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in no_wt_altr_idxs"
                    f" but not the parent profile df index column: {set_minus}"
                )
            )

        if not self.no_rank_no_score_altr_idxs.issubset(self.parent_profile.df.index):
            set_minus = self.no_rank_no_score_altr_idxs.difference(
                self.parent_profile.df.index
            )
            raise ValueError(
                (
                    "no_rank_no_score_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " no_rank_no_score_altr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if not self.nonempty_altr_idxs.issubset(self.parent_profile.df.index):
            set_minus = self.nonempty_altr_idxs.difference(self.parent_profile.df.index)
            raise ValueError(
                (
                    "nonempty_altr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " nonempty_altr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if not self.unaltr_idxs.issubset(self.parent_profile.df.index):
            set_minus = self.unaltr_idxs.difference(self.parent_profile.df.index)
            raise ValueError(
                (
                    "unaltr_idxs is not a subset of the"
                    " parent profile df index column. Here are the indices found in"
                    " unaltr_idxs but not the parent profile df index column: "
                    f"{set_minus}"
                )
            )

        if set().union(*index_sets) != set(self.parent_profile.df.index):
            sym_dif = (
                set()
                .union(*index_sets)
                .symmetric_difference(self.parent_profile.df.index)
            )
            raise ValueError(
                (
                    "Union of ballot indices must equal the parent profile df index "
                    f"column. Here are the indices in one but not the other: {sym_dif}"
                )
            )

    def _reindex_ballot_df(self) -> None:
        """
        Reindex the df to keep track of which ballots are which.

        Raises:
            ValueError: If df_index_column does not have the same length as the ballot list.

        """
        if len(self.df) != len(self.df_index_column):
            raise ValueError(
                "df_index_column does not have the same length as the ballot list."
            )

        df_copy = self.df.copy()
        df_copy.index = pd.Index(self.df_index_column)
        df_copy.index.name = "Ballot Index"
        object.__setattr__(self, "df", df_copy)

    def group_ballots(self) -> PreferenceProfile:
        warnings.warn(
            (
                "Grouping the ballots of a CleanedProfile will return a PreferenceProfile"
                " since this operation resets ballot indices."
            )
        )
        return super().group_ballots()

    def __str__(self) -> str:

        return "Profile has been cleaned\n" + super().__str__()

    __repr__ = __str__

    def __eq__(self, other):
        return super().__eq__(other)
