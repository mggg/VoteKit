from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import pandas as pd

from votekit.ballot import Ballot, RankBallot, ScoreBallot
from votekit.pref_profile import PreferenceProfile, RankProfile, ScoreProfile
from votekit.types import Candidate

RankBallotFunc = Callable[[RankBallot], RankBallot]
ScoreBallotFunc = Callable[[ScoreBallot], ScoreBallot]
BallotFunc = Callable[[Ballot], Ballot]

ProfileT_contra = TypeVar("ProfileT_contra", RankProfile, ScoreProfile, contravariant=True)


class TransformContractError(Exception):
    """
    A value passed to ``transform()`` broke its contract.
    """

    pass


@runtime_checkable
class ProfileTransform(Protocol[ProfileT_contra]):
    """
    Contract for profile transformations.

    Any object with a ``transform_df`` method can be passed to ``transform()`` and is assignable
    to this protocol. ``transform_df`` consumes a profile and its df and returns a transformed df
    with the candidate id translation.
    """

    def transform_df(
        self, df: pd.DataFrame, profile: ProfileT_contra
    ) -> tuple[pd.DataFrame, dict[int, frozenset[Candidate] | dict[int, Candidate]]]: ...


class _ProfileTransformOperation(ABC):
    """
    Internal abstract class for transform operations.

    Requires all transform operation classes to implement ``transformed_df`` that takes
    a profile and returns the transformed df.
    """

    @abstractmethod
    def transformed_df(
        self, profile: PreferenceProfile, ballot_wt_to_transform_mask: Optional[Any] = None
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class _DataFrameOperation(_ProfileTransformOperation):
    """
    A lifted user-defined ProfileTransform.

    transform() enforces symmetric aggrement on profile type with overloads, so Any is the parameter
    internally.
    An error is thrown if the ProfileTransform operation does not return a pd.DataFrame.

    ``transform_df`` acts upon the internal df where candidate sets are represented as integer IDs.
    A mask is applied to the df prior to transform if given. The mask specifies the weight of each
    ballot within the profile to transform. The untransformed weight of the ballot will be added
    back to the end of the profile's internal df.

    ``transform_df`` can only act upon the profile's df and cannot create a new profile object. The
    candidate integer IDs are not guaranteed to aligned with the original profile.
    """

    transform: ProfileTransform[Any]

    def transformed_df(
        self, profile: PreferenceProfile, ballot_wt_to_transform_mask: Optional[Any] = None
    ) -> pd.DataFrame:
        df_to_transform = profile._df.copy()
        if ballot_wt_to_transform_mask is not None:
            df_untouched = profile._df.copy()
            df_to_transform["Weight"] = ballot_wt_to_transform_mask
            df_untouched["Weight"] = profile._df["Weight"] - ballot_wt_to_transform_mask
        result, id_candidate_map = self.transform.transform_df(df_to_transform, profile)
        class_name = type(self.transform).__name__
        if not isinstance(result, pd.DataFrame):
            raise TransformContractError(
                f"Transform {class_name} must return a pd.DataFrame, received {type(result)}."
            )

        if ballot_wt_to_transform_mask is not None:
            result = pd.concat([result, df_untouched], ignore_index=True)
            result.index.name = "Ballot Index"
        if isinstance(profile, RankProfile):
            id_candidate_map = cast(dict[int, frozenset[Candidate]], id_candidate_map)
            return profile._translate_df_ranking_values(result, id_candidate_map)
        elif isinstance(profile, ScoreProfile):
            id_candidate_map = cast(dict[int, Candidate], id_candidate_map)
            return profile._translate_df_score_values(result, id_candidate_map)
        else:
            raise TransformContractError(
                f"Profile must be a RankProfile or ScoreProfile, got {type(profile)}."
            )


@dataclass(frozen=True)
class _BallotOperation(_ProfileTransformOperation):
    """
    A lifted user-defined ballot transformation function.

    A transformed df is constructed from the transformed ballots using PreferenceProfile's
    initialization. For ballot operations that act upon a RankProfile, max_ranking_length is set to
    the value of the original profile.
    If the transformed ballots are longer than the original profile's max_ranking_length, an error
    will be thrown. If the max_ranking_length is not set, then the profile's max_ranking_length is
    to the length of the longest ranking amongst the transformed ballots. However, an error will be
    thrown if that value is less than the maximum number of unique candidates ranked of any ballot.
    So, the original max_ranking_length is passed to avoid throwing this error that was
    corrected in the original profile but the ballot transformation function cannot return ballots
    with rankings longer than the original max_ranking_length.
    """

    ballot_func: BallotFunc

    def transformed_df(
        self, profile: PreferenceProfile, ballot_wt_to_transform_mask: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Transforms the profile's df via a user-defined ballot transformation operation.

        Args:
            profile (PreferenceProfile): profile to transform.
            ballot_wt_to_transform_mask (Optional(Sequence[list])): ...

        Returns:
            pd.DataFrame: transformed df consisting of the transformed ballots.
        """
        transformed_ballots: list[Ballot] = []
        untouched_ballots: list[Ballot] = []
        for ballot_idx, ballot in enumerate(profile.ballots):
            if ballot_wt_to_transform_mask is not None:
                untouched_ballots.append(
                    _ballot_with_new_weight(
                        ballot, ballot.weight - ballot_wt_to_transform_mask[ballot_idx]
                    )
                )
                ballot = _ballot_with_new_weight(ballot, ballot_wt_to_transform_mask[ballot_idx])
            transformed_ballots.append(self.ballot_func(ballot))
        transformed_ballots.extend(untouched_ballots)
        if isinstance(profile, RankProfile):
            return RankProfile(
                ballots=transformed_ballots,
                max_ranking_length=profile.max_ranking_length,
            ).df
        else:
            return PreferenceProfile(ballots=transformed_ballots).df


def _ballot_with_new_weight(ballot: Ballot, new_weight: Any) -> Ballot:
    kwargs = {
        k: getattr(ballot, k)
        for k in Ballot.__slots__
        if k not in ("weight", "_frozen") and hasattr(ballot, k)
    }
    return Ballot(weight=new_weight, **kwargs)


def _with_return_contract(
    ballot_func: BallotFunc, expected_ballot_type: type[Ballot]
) -> BallotFunc:
    """
    Wraps ballot_func to verify its result adheres to the expected ballot type.

    Args:
        ballot_func (BallotFunc): ballot transformation function.
        expected_ballot_type (type[Ballot]): expected type of ballot returned by ballot_func.

    Returns:
        BallotFunc: wrapped ballot_func.

    Raises:
        TransformContractError: ballot_func did not return the expected ballot type.
    """

    func_name = getattr(ballot_func, "__name__", repr(ballot_func))

    def checked(ballot: Ballot) -> Ballot:
        returned_ballot = ballot_func(ballot)
        if not isinstance(returned_ballot, expected_ballot_type):
            raise TransformContractError(
                f"transform function {func_name} returned {type(returned_ballot).__name__},"
                f" expected {expected_ballot_type.__name__}. Offending input ballot: {ballot}"
                f" returned {returned_ballot}."
            )
        return returned_ballot

    return checked


def _lift_to_operation(
    transformation: object, profile: PreferenceProfile
) -> _ProfileTransformOperation:
    """
    Lifts the user-defined transformation to _ProfileTransformOperation.

    If the transformation object has a `transform_df` method, then its assignable to
    ProfileTransform and lifted to _DataFrameOperation.
    If the transformation object is callable, then its lifted to _BallotTransformOperation after
    being wrapped in method that verifies its return type is equivalent to the expected ballot type
    per the profile passed.

    Args:
        transformation (object): ProfileTransform object or ballot transformation function.
        profile (PreferenceProfile): profile to transform.

    Returns:
        _ProfileTransformOperation: object with transformed_df method.

    Raises:
        TransformContractError: transformation is not a valid ballot transformation function nor a
            ProfileTransform instance.
    """
    if isinstance(transformation, type):
        raise TransformContractError(
            f"Transformation was given the class {transformation.__name__}, not an instance."
            f" Did you mean to construct it like {transformation.__name__}(...)?"
        )
    if isinstance(transformation, ProfileTransform):
        return _DataFrameOperation(transform=transformation)
    elif callable(transformation):  # function
        expected_ballot_type: type[Ballot]
        if isinstance(profile, RankProfile):
            expected_ballot_type = RankBallot
        elif isinstance(profile, ScoreProfile):
            expected_ballot_type = ScoreBallot
        else:
            raise TransformContractError(
                f"profile must be a RankProfile or ScoreProfile, received {type(profile)}."
            )
        ballot_func = cast(BallotFunc, transformation)
        return _BallotOperation(_with_return_contract(ballot_func, expected_ballot_type))
    else:
        raise TypeError("Transformation must be a ballot function or ProfileTransform instance.")


ProbabilityFunctions = Callable[[Any], bool] | Callable[[pd.DataFrame, dict], Sequence[bool]]


@runtime_checkable
class DataFrameProbability(Protocol):
    def transform_mask(
        self,
        df: pd.DataFrame,
        id_cand_set_map: dict[int, frozenset[Candidate]] | dict[int, Candidate],
    ) -> Any: ...


def _lift_to_transform_mask(probability: object, profile: PreferenceProfile) -> Callable:
    if isinstance(probability, type):
        raise TransformContractError(
            f"Probability was given the class {probability.__class__}"
            " Did you mean to construct it like"
            f" {probability.__class__}(...)?"
        )
    if isinstance(probability, DataFrameProbability):
        # TODO: check the output is an array of booleans
        def dataframe_probability(profile: PreferenceProfile):
            # TODO: PreferenceProfile does not have mapping, add as an attribute?
            assert isinstance(profile, (RankProfile, ScoreProfile))
            result = probability.transform_mask(profile._df.copy(), profile.id_candidate_map.copy())
            if any(not isinstance(ballot_result, (bool, np.bool_)) for ballot_result in result):
                raise TransformContractError(
                    f"Probability {probability.__class__} must return a list of boolean values."
                )
            if len(result) != profile.total_ballot_wt:
                raise TransformContractError(
                    f"Probability {probability.__class__} must return"
                    " a list with length equal to the total ballot weight"
                    f" of the profile. Expected {profile.total_ballot_wt},"
                    f" got {len(result)}."
                )
            return result

        return dataframe_probability
    elif isinstance(probability, Callable):
        prob_func = cast(Callable[[Ballot], bool], probability)

        def ballot_probability(profile: PreferenceProfile):
            result = []
            for ballot in profile.ballots:
                for _ in range(int(ballot.weight)):
                    ballot_result = prob_func(ballot)
                    if not isinstance(ballot_result, bool):
                        raise TransformContractError(
                            f"Probability {probability.__class__}"
                            " must return a bool, got"
                            f" {type(ballot_result)}."
                        )
                    result.append(ballot_result)
            return result

        return ballot_probability
    else:
        raise TransformContractError()


@overload
def transform(
    profile: RankProfile,
    transformation: object,
    *,
    probability: Optional[object] = None,
    group_ballots_first: bool = True,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    reduce_max_ranking_length: bool = False,
) -> RankProfile: ...


@overload
def transform(
    profile: ScoreProfile,
    transformation: object,
    *,
    probability: Optional[object] = None,
    group_ballots_first: bool = True,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    reduce_max_ranking_length: bool = False,
) -> ScoreProfile: ...


def transform(
    profile: PreferenceProfile,
    transformation: object,
    *,
    probability: Optional[object] = None,
    group_ballots_first: bool = True,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    reduce_max_ranking_length: bool = False,
) -> Any:
    """
    Transformas a profile.

    Args:
        profile (PreferenceProfile): profile to transform.
        transformation (object): Object with ``transform_df`` func assignable to ProfileTransform or
            a ballot transform function assignable to Callable[[Ballot], Ballot]. Must return the
            ballot type it consumes.
        group_ballots_first (bool, optional): whether to group profile's ballots by unique ballot
            type. True by default.
        remove_empty_ballots (bool, optional): whether to remove ballots with an empty ranking. An
            empty ranking means all positions contain ``frozenset("~")``. True by default.
        remove_zero_weight_ballots (bool, optional): whether to remove ballots with zero weight.
            True by default.
        retain_original_candidate_list (bool, optional): whether to keep the original profile's
            candidate list or set it to the ``candidates_cast`` of the transformed profile if False.
            True by default.
        reduce_max_ranking_length (bool, optional): whether to keep the original profile's
            ``max_ranking_length`` or if True, set it to the minimum number of df columns to
            represent any of the transformed profile's rankings, given its at least the maximum
            number of unique candidates ranked amongst all ballots. Only applicable to RankProfiles.
            False by default.

        Returns:
            Any: transformed profile.

        Raises:
            TypeError: profile to transform must be a RankProfile or ScoreProfile.

    """
    if not isinstance(profile, (RankProfile, ScoreProfile)):
        raise TypeError(f"profile must be a RankProfile or ScoreProfile, received {type(profile)}.")
    operation = _lift_to_operation(transformation, profile)
    profile = profile.group_ballots() if group_ballots_first else profile
    ballot_wt_transform_mask = None
    if probability is not None:
        probability_func = _lift_to_transform_mask(probability, profile)
        transform_mask = probability_func(profile)
        ballot_wt_transform_mask = _ballot_weight_to_transform(transform_mask, profile)

    transformed_df = operation.transformed_df(profile, ballot_wt_transform_mask)

    if remove_empty_ballots and isinstance(profile, RankProfile):
        ranking_cols = [col for col in transformed_df.columns if "Ranking_" in col]
        empty_ballot_mask = (
            transformed_df[ranking_cols].map(lambda x: x == frozenset({"~"})).all(axis=1)
        )
        transformed_df = transformed_df[~empty_ballot_mask]
    if remove_zero_weight_ballots:
        transformed_df = transformed_df[transformed_df["Weight"] != 0]

    assert profile.max_ranking_length is not None
    return PreferenceProfile(
        df=transformed_df,
        candidates=profile.candidates if retain_original_candidate_list else tuple(),
        max_ranking_length=profile.max_ranking_length
        if not reduce_max_ranking_length
        else _reduced_max_ranking_length(transformed_df),
    )


def _ballot_weight_to_transform(per_wt_transform_mask: Any, profile) -> Any:
    """
    Determines the weight per ballot of a profile to apply transformation.

    Ballots with zero weight stay at zero weight.
    Args:
        per_wt_transform_mask (Any): list or numpy array of boolean values. Length of the total
            weight of the profile. Indicates the amount of weight to transform per ballot.
        profile (PreferenceProfile): profile to transform
    Returns:
        np.NDArray(bool): 1d array of amount of weight to transform per ballot in profile.
            Index of array maps to the index of the ballot within the profile.
    """
    mask_counts = np.cumsum(np.asarray(per_wt_transform_mask))
    ballot_wt_to_transform = np.zeros(profile.num_ballots, dtype=int)
    bin_start = 0
    prev_bin_count = 0
    for ballot_idx, ballot_weight in enumerate(profile._df["Weight"]):
        if ballot_weight == 0:
            continue
        else:
            bin_end = bin_start + int(ballot_weight) - 1
            ballot_wt_to_transform[ballot_idx] = (
                mask_counts[bin_end] - prev_bin_count if ballot_weight > 0 else 0
            )
            prev_bin_count = mask_counts[bin_end]
            bin_start = bin_end + 1
    return ballot_wt_to_transform


# TODO: target the feat/reduce-max-ranking-length-clean branch and move these functions
# into a shared utility
def _max_candidates_ranked(profile: RankProfile | pd.DataFrame) -> int:
    """
    The maximum number of unique candidates ranked on any ballot in the profile.

    Can be longer than the number of ranking columns in df if candidates are tied.

    Args:
        profile (RankProfile | pd.DataFrame): rank profile or the profile's df.

    Returns:
        int: Maximum number of candidates ranked amongst all ballots within profile.
    """
    df = profile.df if isinstance(profile, RankProfile) else profile

    if df.empty:
        return 0

    tilde = frozenset("~")

    ranking_cols = [col for col in df.columns if "Ranking_" in col]
    return (
        df[ranking_cols]
        .apply(
            lambda row: len(frozenset.union(*row) - tilde),
            axis=1,
        )
        .max()
    )


def _reduced_max_ranking_length(profile: RankProfile | pd.DataFrame) -> int:
    """
    The minimum max_ranking_length to represent all ballots of the profile after cleaning.

    After cleaning, the max_ranking_length can be reduced to the minimum number of columns necessary
    to represent all rankings. A profile's max_ranking_length has a lower bound: maximum number of
    unique candidates ranked on any ballot. When there are tied candidates within a ballot, the
    number of candidates ranked can be greater than the minimum number of ranking columns to
    represent all ballots. This function returns the maximum of these two values.

    Args:
        profile (RankProfile | pd.DataFrame): rank profile or the profile's df.

    Returns:
        int: Maximum ranking length.
    """
    df = profile.df if isinstance(profile, RankProfile) else profile
    tilde = frozenset("~")

    ranking_cols = [col for col in df.columns if "Ranking_" in col]
    last_col_with_cands = max(
        (i + 1 for i, col in enumerate(ranking_cols) if (df[col] != tilde).any()),
        default=0,
    )
    return max(last_col_with_cands, _max_candidates_ranked(profile))


# TODO: add a probability function and wire throughout
# provide a probability function that returns 0 and 1s or true and falses when provided a list


@dataclass
class SwapCandidates:
    candidate_a: Candidate
    candidate_b: Candidate
    max_distance: Optional[int] = None
    min_distance: Optional[int] = None
    strict_order: bool = False
    swap_ties: bool = False  # TODO: need to implement

    # TODO: swap

    def transform_df(
        self, df: pd.DataFrame, profile: RankProfile
    ) -> tuple[pd.DataFrame, dict[int, frozenset[Candidate]]]:
        if self.swap_ties:
            cand_a_ids = [
                id
                for id, cand_set in profile.id_candidate_map.items()
                if self.candidate_a in cand_set
            ]
            cand_b_ids = [
                id
                for id, cand_set in profile.id_candidate_map.items()
                if self.candidate_b in cand_set
            ]
        else:
            cand_a_ids = [
                id
                for id, cand_set in profile.id_candidate_map.items()
                if frozenset({self.candidate_a}) == cand_set
            ]
            cand_b_ids = [
                id
                for id, cand_set in profile.id_candidate_map.items()
                if frozenset({self.candidate_b}) == cand_set
            ]

        ranking_cols = [col for col in df.columns if "Ranking_" in col]
        ranking_arr = df[ranking_cols].to_numpy().copy()

        cand_a_positions = np.isin(ranking_arr, cand_a_ids)
        cand_b_positions = np.isin(ranking_arr, cand_b_ids)
        cand_a_counts = np.count_nonzero(cand_a_positions, axis=1)
        cand_b_counts = np.count_nonzero(cand_b_positions, axis=1)
        swapable_row_mask = (cand_a_counts == 1) & (cand_b_counts == 1)
        if np.any(cand_a_counts > 1):  # check only rows with a and b?
            raise ValueError(
                f"Profile contains rankings with candidate {str(self.candidate_a)}"
                " Cannot deterministically swap."
            )
        if np.any(cand_b_counts > 1):
            raise ValueError(
                f"Profile contains rankings with candidate {str(self.candidate_b)}"
                " Cannot deterministically swap."
            )

        cand_a_rank_idxs = np.argmax(cand_a_positions, axis=1)
        cand_b_rank_idxs = np.argmax(cand_b_positions, axis=1)

        cand_dists = cand_a_rank_idxs - cand_b_rank_idxs
        swap_cand_mask = np.ones(len(cand_a_rank_idxs), dtype=bool) & swapable_row_mask
        if self.strict_order:
            swap_cand_mask &= cand_dists < 0
        abs_cand_dists = np.abs(cand_dists)
        if self.min_distance is not None:
            swap_cand_mask &= abs_cand_dists >= (self.min_distance + 1)
        if self.max_distance is not None:
            swap_cand_mask &= abs_cand_dists <= (self.max_distance + 1)

        swap_rows = np.nonzero(swap_cand_mask)[0]
        cand_a_idxs, cand_b_idxs = cand_a_rank_idxs[swap_rows], cand_b_rank_idxs[swap_rows]
        cand_a_id_vals = ranking_arr[swap_rows, cand_a_idxs]
        cand_b_id_vals = ranking_arr[swap_rows, cand_b_idxs]
        ranking_arr[swap_rows, cand_a_idxs] = cand_b_id_vals
        ranking_arr[swap_rows, cand_b_idxs] = cand_a_id_vals

        df[ranking_cols] = ranking_arr

        return df, profile.id_candidate_map


@dataclass
class SwapRankPositions:
    # Index refers to Ranking_{i}
    ranking_col_a_idx: int
    ranking_col_b_idx: int

    def transform_df(self, df: pd.DataFrame, profile: RankProfile):
        ranking_cols = [col for col in df.columns if "Ranking_" in col]
        ranking_arr = df[ranking_cols].to_numpy().copy()
        # ranking columns are 1-based
        ranking_arr_a_idx = self.ranking_col_a_idx - 1
        ranking_arr_b_idx = self.ranking_col_b_idx - 1
        ranking_arr[:, [ranking_arr_a_idx, ranking_arr_b_idx]] = ranking_arr[
            :, [ranking_arr_b_idx, ranking_arr_a_idx]
        ]

        df[ranking_cols] = ranking_arr
        return df, profile.id_candidate_map


@dataclass
class RemoveCandidate:
    removed: Candidate

    def transform_df(self, df: pd.DataFrame, profile: RankProfile):
        removed_ids_dict = {}
        removed_set = frozenset({self.removed})

        orig_ids, orig_cand_sets = zip(*profile.id_candidate_map.items())

        candidate_id_map_copy = profile.candidate_id_map.copy()
        id_candidate_map_copy = profile.id_candidate_map.copy()
        for id, cand_set in zip(orig_ids, orig_cand_sets):
            if removed_set == cand_set:
                removed_ids_dict[id] = candidate_id_map_copy.get(
                    frozenset(), len(candidate_id_map_copy)
                )
                if removed_ids_dict[id] == len(candidate_id_map_copy):
                    id_candidate_map_copy[len(candidate_id_map_copy)] = frozenset()
                    candidate_id_map_copy[frozenset()] = len(candidate_id_map_copy)
            elif self.removed in cand_set:
                new_cand_set = cand_set - removed_set
                removed_ids_dict[id] = candidate_id_map_copy.get(
                    new_cand_set, len(candidate_id_map_copy)
                )
                if removed_ids_dict[id] == len(candidate_id_map_copy):
                    id_candidate_map_copy[len(candidate_id_map_copy)] = new_cand_set
                    candidate_id_map_copy[new_cand_set] = len(candidate_id_map_copy)
            else:
                removed_ids_dict[id] = id

        ranking_cols = [col for col in df.columns if "Ranking_" in col]
        ranking_arr = df[ranking_cols].to_numpy().copy()
        min_id = min(removed_ids_dict)  # always -1 for frozenset({'~'})
        id_mapping = np.zeros(len(removed_ids_dict) + 1)
        for old_id, new_id in removed_ids_dict.items():
            id_mapping[old_id - min_id] = new_id

        ranking_arr_zero_idx = ranking_arr - min_id
        mapped_ranking_arr = id_mapping[ranking_arr_zero_idx]
        df[ranking_cols] = mapped_ranking_arr
        return df, id_candidate_map_copy
