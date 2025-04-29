from __future__ import annotations
from .pref_profile import PreferenceProfile
from pydantic import ConfigDict, model_validator, SkipValidation
from pydantic.dataclasses import dataclass
from dataclasses import field
import warnings
from typing_extensions import Self


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class CleanedProfile(PreferenceProfile):
    """
    CleanedProfile class, which is used to keep track of how ballots are altered from the original
    profile.

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
        no_weight_altr_ballot_indices (list[int], optional): List of indices of ballots that have
            0 weight as a result of cleaning. Indices are with respect
            to ``parent_profile.ballots``.
        no_ranking_and_no_scores_altr_ballot_indices (list[int], optional): List of indices of
            ballots that have no ranking and no scores as a result of cleaning. Indices are with
            respect to ``parent_profile.ballots``.
        valid_but_altr_ballot_indices (list[int], optional):  List of indices of ballots that have
            been altered but still have weight and (ranking or score) as a result of cleaning.
            Indices are with respect to ``parent_profile.ballots``.
        unaltr_ballot_indices (list[int], optional):  List of indices of ballots that have
            been unaltered by cleaning. Indices are with respect to ``parent_profile.ballots``.

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
    """

    parent_profile: SkipValidation[PreferenceProfile | CleanedProfile] = (
        PreferenceProfile()
    )
    df_index_column: list[int] = field(default_factory=list)
    no_weight_altr_ballot_indices: set[int] = field(default_factory=set)
    no_ranking_and_no_scores_altr_ballot_indices: set[int] = field(default_factory=set)
    valid_but_altr_ballot_indices: set[int] = field(default_factory=set)
    unaltr_ballot_indices: set[int] = field(default_factory=set)

    @model_validator(mode="after")
    def indices_must_match_parent_df(self) -> Self:
        """
        Validate the index sets.

        Raises:
            ValueError: index set is not a subset of the parent df index.
            ValueError: union of indices is not equal to the parent df index.

        """
        index_sets = [
            self.no_weight_altr_ballot_indices,
            self.no_ranking_and_no_scores_altr_ballot_indices,
            self.valid_but_altr_ballot_indices,
            self.unaltr_ballot_indices,
        ]

        if not self.no_weight_altr_ballot_indices.issubset(
            self.parent_profile.df.index
        ):
            raise ValueError(
                (
                    "no_weight_altr_ballot_indices is not a subset of the"
                    " parent profile df index column."
                )
            )

        if not self.no_ranking_and_no_scores_altr_ballot_indices.issubset(
            self.parent_profile.df.index
        ):
            raise ValueError(
                (
                    "no_ranking_and_no_scores_altr_ballot_indices is not a subset of "
                    "the parent profile df index column."
                )
            )

        if not self.valid_but_altr_ballot_indices.issubset(
            self.parent_profile.df.index
        ):
            raise ValueError(
                (
                    "valid_but_altr_ballot_indices is not a subset of "
                    "the parent profile df index column."
                )
            )

        if not self.unaltr_ballot_indices.issubset(self.parent_profile.df.index):
            raise ValueError(
                (
                    "unaltr_ballot_indices is not a subset of "
                    "the parent profile df index column."
                )
            )

        if set().union(*index_sets) != set(self.parent_profile.df.index):
            raise ValueError(
                (
                    "Union of ballot indices must equal the parent profile df index "
                    "column."
                )
            )
        return self

    @model_validator(mode="after")
    def reindex_ballot_df(self) -> Self:
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
        df_copy.index = self.df_index_column
        df_copy.index.name = "Ballot Index"
        object.__setattr__(self, "df", df_copy)

        return self

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

    def __eq__(self, other):
        return super().__eq__(other)
