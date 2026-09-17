from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar, cast, overload, runtime_checkable

import pandas as pd

from votekit.ballot import Ballot, RankBallot, ScoreBallot
from votekit.pref_profile import PreferenceProfile, RankProfile, ScoreProfile

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
    to this protocol. ``transform_df`` consumes a profile and its df and returns a transformed df.
    """

    def transform_df(self, df: pd.DataFrame, profile: ProfileT_contra) -> pd.DataFrame: ...


class _ProfileTransformOperation(ABC):
    """
    Internal abstract class for transform operations.

    Requires all transform operation classes to implement ``transformed_df`` that takes
    a profile and returns the transformed df.
    """

    @abstractmethod
    def transformed_df(self, profile: PreferenceProfile) -> pd.DataFrame: ...


@dataclass(frozen=True)
class _DataFrameOperation(_ProfileTransformOperation):
    """
    A lifted user-defined ProfileTransform.

    transform() enforces symmetric aggrement with overloads, so Any is the parameter internally.
    An error is thrown if the ProfileTransform operation does not return a pd.DataFrame.
    """

    transform: ProfileTransform[Any]

    def transformed_df(self, profile: PreferenceProfile) -> pd.DataFrame:
        result = self.transform.transform_df(profile.df.copy(), profile)
        class_name = type(self.transform).__name__
        if not isinstance(result, pd.DataFrame):
            raise TransformContractError(
                f"Transform {class_name} must return a pd.DataFrame, received {type(result)}."
            )
        return result


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

    def transformed_df(self, profile: PreferenceProfile) -> pd.DataFrame:
        """
        Transforms the profile's df via a user-defined ballot transformation operation.

        Args:
            profile (PreferenceProfile): profile to transform.

        Returns:
            pd.DataFrame: transformed df consisting of the transformed ballots.
        """
        transformed_ballots = []
        for ballot in profile.ballots:
            transformed_ballots.append(self.ballot_func(ballot))

        if isinstance(profile, RankProfile):
            return RankProfile(
                ballots=transformed_ballots,
                max_ranking_length=profile.max_ranking_length,
            ).df
        else:
            return PreferenceProfile(ballots=transformed_ballots).df


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


@overload
def transform(
    profile: RankProfile,
    transformation: object,
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
    group_ballots_first: bool = True,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    reduce_max_ranking_length: bool = False,
) -> ScoreProfile: ...


def transform(
    profile: PreferenceProfile,
    transformation: object,  #
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
    transformed_df = operation.transformed_df(profile)

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
