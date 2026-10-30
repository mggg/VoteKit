from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from votekit.pref_profile.pref_profile import PreferenceProfile, RankProfile, ScoreProfile

from dataclasses import replace
from fractions import Fraction
from functools import partial
from math import fsum
from typing import Optional, Sequence, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import (
    RANKING_COLUMN_PREFIX,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot import Ballot, RankBallot, ScoreBallot
from votekit.constants import TOL_EPSILON
from votekit.pref_profile.profile_matrix import (
    EMPTY_SET_ID,
    END_OF_RANKING_ID,
    ProfileVoterWeightData,
    RankMatrix,
    ScoreMatrix,
    WeightKind,
)
from votekit.pref_profile.profile_matrix.rank_matrix import _concatenate_rank_matrices
from votekit.pref_profile.profile_matrix.score_matrix import _concatenate_score_matrices
from votekit.pref_profile.profile_matrix.shared_data import _as_fraction
from votekit.types import Candidate
from votekit.utils import sort_candidates_pseudo_lexicographically

MatrixT = TypeVar("MatrixT", RankMatrix, ScoreMatrix)


def _convert_ranking_cols_to_ranking(
    row: pd.Series, max_ranking_length: int
) -> Optional[tuple[frozenset, ...]]:
    """
    Convert the ranking cols to a ranking tuple in profile.to_df().

    Args:
        row (pd.Series): Row of a profile.to_df().
        max_ranking_length (int, optional): The maximum length of a ranking.

    Returns:
        Optional[tuple[frozenset, ...]]: Ranking of ballot.

    Raises:
        ValueError: NaN values can only trail on a ranking.

    """
    ranking_cols_idxs = [f"{RANKING_COLUMN_PREFIX}{i + 1}" for i in range(max_ranking_length)]

    if any(idx not in row.index for idx in ranking_cols_idxs):
        raise ValueError(f"Row has improper ranking columns: {row.index}.")

    if any(
        row[col_idx] == frozenset({"~"})
        and not all(row[idx] == frozenset({"~"}) for idx in ranking_cols_idxs[i:])
        for i, col_idx in enumerate(ranking_cols_idxs)
    ):
        raise ValueError(
            f"Row {row} has '~' between valid ranking positions. "
            "'~' values can only trail on a ranking."
        )

    ranking = [row[col_idx] for col_idx in ranking_cols_idxs if row[col_idx] != frozenset("~")]

    return tuple(ranking) if len(ranking) > 0 else None


def convert_row_to_rank_ballot(row: pd.Series, max_ranking_length: int = 0) -> RankBallot:
    """
    Convert a row of a properly formatted profile.to_df() to a Ballot.

    Args:
        row (pd.Series): Row of a profile.to_df().
        max_ranking_length (int, optional): The maximum length of a ranking. Defaults to 0, which
            is used for ballots with no ranking.

    Returns:
        RankBallot: Ballot corresponding to the row of the df.
    """
    ranking = None
    if max_ranking_length > 0:
        ranking = _convert_ranking_cols_to_ranking(row, max_ranking_length)
    voter_set = row[VOTER_SET_COLUMN]
    weight = row[WEIGHT_COLUMN]

    return RankBallot(
        ranking=ranking,
        weight=weight,
        voter_set=voter_set,
    )


def convert_row_to_score_ballot(row: pd.Series, candidates: tuple[Candidate, ...]) -> ScoreBallot:
    """
    Convert a row of a properly formatted profile.to_df() to a Ballot.

    Args:
        row (pd.Series): Row of a profile.to_df().
        candidates (tuple[Candidate,...]): The name of the candidates.
            Candidates can be strings, integers, or mix of both.

    Returns:
        ScoreBallot: Ballot corresponding to the row of the df.
    """
    scores = {c: row[c] for c in candidates if c in row and not pd.isna(row[c])}
    voter_set = row[VOTER_SET_COLUMN]
    weight = row[WEIGHT_COLUMN]

    return ScoreBallot(
        scores=scores if scores != dict() else None,
        weight=weight,
        voter_set=voter_set,
    )


def _df_to_rank_ballot_tuple(
    df: pd.DataFrame, candidates: tuple[Candidate, ...], max_ranking_length: int = 0
) -> tuple[RankBallot, ...]:
    """
    Convert a properly formatted profile.to_df() into a list of ballots.

    Args:
        df (pd.DataFrame): A profile.to_df().
        candidates (tuple[Candidate,...]): The candidates.
            Candidates can be strings, integers, or mix of both.
        max_ranking_length (int, optional): The maximum length of a ranking. Defaults to 0, which
            is used for ballots with no ranking.

    Returns:
        tuple[RankBallot]: The tuple of ballots.
    """
    if df.empty:
        return tuple()

    return tuple(
        df.apply(
            partial(
                convert_row_to_rank_ballot,
                max_ranking_length=max_ranking_length,
            ),
            axis="columns",
        )
    )


def rank_profile_to_ballot_dict(
    rank_profile: RankProfile, standardize: bool = False
) -> dict[RankBallot, float]:
    """
    Converts profile to dictionary with keys = ballots and
    values = corresponding total weights.

    Args:
        rank_profile (RankProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[Ballot, float]:
            A dictionary with ballots (keys) and corresponding total weights (values).
    """
    tot_weight = float(rank_profile.total_ballot_wt)
    di: dict = {}
    for ballot in rank_profile.ballots:
        weightless_ballot = Ballot(
            ranking=ballot.ranking,
            voter_set=ballot.voter_set,
        )
        weight = float(ballot.weight)
        if standardize:
            weight /= tot_weight

        if weightless_ballot not in di.keys():
            di[weightless_ballot] = weight
        else:
            di[weightless_ballot] += weight
    return di


def score_profile_to_ballot_dict(
    score_profile: ScoreProfile, standardize: bool = False
) -> dict[ScoreBallot, float]:
    """
    Converts profile to dictionary with keys = ballots and
    values = corresponding total weights.

    Args:
        score_profile (ScoreProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[Ballot, float]:
            A dictionary with ballots (keys) and corresponding total weights (values).
    """
    tot_weight = float(score_profile.total_ballot_wt)
    di: dict = {}
    for ballot in score_profile.ballots:
        weightless_ballot = Ballot(
            scores=ballot.scores,
            voter_set=ballot.voter_set,
        )
        weight = ballot.weight
        if standardize:
            weight /= tot_weight

        if weightless_ballot not in di.keys():
            di[weightless_ballot] = weight
        else:
            di[weightless_ballot] += weight
    return di


def rank_profile_to_ranking_dict(
    rank_profile: RankProfile, standardize: bool = False
) -> dict[tuple[frozenset[Candidate], ...], float]:
    """
    Converts profile to dictionary with keys = rankings and
    values = corresponding total weights.

    Args:
        rank_profile (RankProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[tuple[frozenset[Candidate],...], float]:
            A dictionary with candidate rankings (keys) and corresponding total weights (values).
            Candidates can be strings, integers, or mix of both.

    Raises:
        TypeError: Profile must be a RankProfile.
    """
    from votekit.pref_profile.pref_profile import RankProfile

    if not isinstance(rank_profile, RankProfile):
        raise TypeError(("Profile must be a RankProfile."))
    tot_weight = float(rank_profile.total_ballot_wt)
    di: dict = {}
    for ballot in rank_profile.ballots:
        ranking = ballot.ranking
        weight = float(ballot.weight)
        if standardize:
            weight /= tot_weight
        di[ranking] = di.get(ranking, 0) + weight

    return di


def score_profile_to_scores_dict(
    score_profile: ScoreProfile, standardize: bool = False
) -> dict[tuple[tuple[Candidate, float], ...] | None, float]:
    """
    Converts profile to dictionary with keys = scores and
    values = corresponding total weights.

    Args:
        score_profile (ScoreProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[tuple[tuple[Candidate, float], ...] | None, float]:
            A dictionary with candidate scores (keys) and corresponding total weights (values).
            Candidates can be strings, integers, or mix of both.

    Raises:
        TypeError: Profile must be a ScoreProfile.
    """
    from votekit.pref_profile.pref_profile import ScoreProfile

    if not isinstance(score_profile, ScoreProfile):
        raise TypeError(("Profile must be a ScoreProfile."))

    tot_weight = float(score_profile.total_ballot_wt)
    di: dict[tuple[tuple[Candidate, float], ...] | None, float] = {}
    for ballot in score_profile.ballots:
        scores = tuple(ballot.scores.items()) if ballot.scores else None
        weight = float(ballot.weight)
        if standardize:
            weight /= tot_weight

        di[scores] = di.get(scores, 0) + weight
    return di


def _export_profile_rows(profile: PreferenceProfile, row_indices: NDArray[np.intp]) -> pd.DataFrame:
    """
    Export selected matrix rows, retaining their source ballot labels.

    Args:
        profile (PreferenceProfile): Profile whose rows are selected.
        row_indices (NDArray[np.intp]): Positional row indices in the requested output order.

    Returns:
        pd.DataFrame: Independent export of the selected rows, including repeated index labels.
    """
    matrix = profile._matrix
    if matrix is None:
        raise NotImplementedError
    return matrix.to_df(rows=row_indices)


def _ordered_profile_rows(
    profile: PreferenceProfile, sort_by_weight: bool | None
) -> NDArray[np.intp]:
    """
    Select source row order or descending weight order, preserving source order for ties.
    """
    matrix = profile._matrix
    if matrix is None:
        raise NotImplementedError
    weights = matrix.voter_weight_data.weights
    return np.argsort(-weights, kind="stable") if sort_by_weight else np.arange(len(weights))


def profile_df_head(
    profile: PreferenceProfile,
    n: int,
    sort_by_weight: Optional[bool] = True,
    percents: Optional[bool] = False,
    totals: Optional[bool] = False,
    n_decimals: int = 1,
) -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the top-n ballots in profile.

    Args:
        n (int): Number of ballots to view.
        sort_by_weight (bool, optional): If True, rank ballot from most to least votes.
            Equal weights retain source order. Source index labels are preserved. Defaults to True.
        percents (bool, optional): If True, show voter share for a given ballot.
            Defaults to False.
        totals (bool, optional): If True, show total values for Percent and Weight.
            Defaults to False.
        n_decimals (int, optional): Number of decimals to round to. Defaults to 1.

    Returns:
        pandas.DataFrame: A dataframe with top-n ballots.

    Raises:
        ZeroDivisionError: Profile has 0 total ballot weight; cannot show percentages.
    """

    rows = _ordered_profile_rows(profile, sort_by_weight)[:n]
    df = _export_profile_rows(profile, rows)

    df_col_num = len(df.columns)
    if percents:
        if profile.total_ballot_wt == 0:
            raise ZeroDivisionError("Profile has 0 total ballot weight; cannot show percentages.")
        df["Percent"] = df[WEIGHT_COLUMN] / float(profile.total_ballot_wt)

    if totals:
        total_row = [""] * (df_col_num - 1) + [df[WEIGHT_COLUMN].sum()]
        if percents:
            total_row += [df["Percent"].sum()]
        df.loc["Total"] = total_row

    if percents:
        df["Percent"] = df["Percent"].apply(lambda x: f"{float(x):.{n_decimals}%}")

    return df


def profile_df_tail(
    profile: PreferenceProfile,
    n: int,
    sort_by_weight: Optional[bool] = True,
    percents: Optional[bool] = False,
    totals: Optional[bool] = False,
    n_decimals: int = 1,
) -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the bottom-n ballots in profile.

    Args:
        n (int): Number of ballots to view.
        sort_by_weight (bool, optional): If True, select the last rows in descending weight order.
            Equal weights retain source order. Source index labels are preserved. Defaults to True.
        percents (bool, optional): If True, show voter share for a given ballot.
            Defaults to False.
        totals (bool, optional): If True, show total values for Percent and Weight.
            Defaults to False.
        n_decimals (int, optional): Number of decimals to round to. Defaults to 1.

    Returns:
        pandas.DataFrame: A data frame with bottom-n ballots.

    Raises:
        ZeroDivisionError: Profile has 0 total ballot weight; cannot show percentages.
    """
    ordered_rows = _ordered_profile_rows(profile, sort_by_weight)
    rows = ordered_rows[-n:] if n else ordered_rows[:0]
    df = _export_profile_rows(profile, rows)

    df_col_num = len(df.columns)
    if percents:
        if profile.total_ballot_wt == 0:
            raise ZeroDivisionError("Profile has 0 total ballot weight; cannot show percentages.")
        df["Percent"] = df[WEIGHT_COLUMN] / float(profile.total_ballot_wt)

    if totals:
        total_row = [""] * (df_col_num - 1) + [df[WEIGHT_COLUMN].sum()]
        if percents:
            total_row += [df["Percent"].sum()]
        df.loc["Total"] = total_row

    if percents:
        df["Percent"] = df["Percent"].apply(lambda x: f"{float(x):.{n_decimals}%}")

    return df


def convert_rank_profile_to_score_profile_via_score_vector(
    rank_profile: RankProfile,
    score_vector: Sequence[float],
) -> ScoreProfile:
    """
    Convert a rank profile to a score profile using a score vector. Ballots must
    not contain ties. Score vector
    should be non-increasing and non-negative.

    Args:
        rank_profile (RankProfile): Rank profile to convert.
        score_vector (Sequence[float]): Score vector to use.

    Returns:
        ScoreProfile: Score profile retaining ballot row order, index labels, voters, and weights
            converted to floats.

    Raises:
        ValueError: Ballots must not contain ties.
        ValueError: Score vector must be non-increasing and non-negative.
    """
    # here to prevent circular import
    from votekit.pref_profile.pref_profile import ScoreProfile
    from votekit.utils import validate_score_vector

    validate_score_vector(score_vector)
    score_vector = list(score_vector)

    assert rank_profile.max_ranking_length is not None
    if len(score_vector) < rank_profile.max_ranking_length:
        score_vector += [0] * (rank_profile.max_ranking_length - len(score_vector))

    matrix = rank_profile._matrix
    candidate_columns = {
        candidate: column for column, candidate in enumerate(rank_profile.candidates)
    }
    score_columns_by_set = np.full(len(matrix.cand_sets), -1, dtype=np.intp)
    referenced_ids = np.unique(matrix.rankings)
    if any(len(matrix.get_cand_set_from_id(int(set_id))) > 1 for set_id in referenced_ids):
        raise ValueError("Ballots must not contain ties.")
    for set_id in referenced_ids:
        if set_id == END_OF_RANKING_ID:
            continue
        candidate_set = matrix.get_cand_set_from_id(int(set_id))
        candidate = next(iter(candidate_set))
        score_columns_by_set[set_id - EMPTY_SET_ID] = candidate_columns[candidate]

    scores = np.full((len(matrix), len(rank_profile.candidates)), np.nan)
    for position, set_ids in enumerate(matrix.rankings.T):
        columns = score_columns_by_set[set_ids - EMPTY_SET_ID]
        rows = np.flatnonzero(columns >= 0)
        scores[rows, columns[rows]] = (
            score_vector[position] if score_vector[position] > 0 else np.nan
        )

    data = matrix.voter_weight_data.with_weights(matrix.voter_weight_data.weights.astype(float))
    return ScoreProfile._from_matrix(
        ScoreMatrix(
            voter_weight_data=data,
            scores=scores,
            score_columns=rank_profile.candidates,
            candidates=rank_profile.candidates,
        )
    )


def _sum_rank_profiles(rank_profiles: Sequence[PreferenceProfile]) -> RankProfile:
    """
    Helper function for sum_profiles that sums RankProfiles.

    Args:
        rank_profiles (Sequence[PreferenceProfile]): List of profiles to sum.

    Raises:
        TypeError: Each profile must be of RankProfile type
    """

    from votekit.pref_profile.pref_profile import RankProfile

    if len(rank_profiles) == 1 and isinstance(rank_profiles[0], RankProfile):
        return rank_profiles[0].copy()

    if not (all(isinstance(p, RankProfile) for p in rank_profiles)):
        invalid_profiles = [
            (i, type(p).__name__)
            for i, p in enumerate(rank_profiles)
            if not isinstance(p, RankProfile)
        ]
        invalid_profiles_str = ", ".join(f"index {i} ({t})" for i, t in invalid_profiles)
        raise TypeError(
            "All profiles must be of the same type, RankProfile. "
            f"non-RankProfiles found at: {invalid_profiles_str}"
        )

    candidates = list(set().union(*[set(profile.candidates) for profile in rank_profiles]))
    max_ranking_length = max([profile.max_ranking_length for profile in rank_profiles])

    validated_profiles = cast(Sequence[RankProfile], rank_profiles)
    matrices = [profile._matrix for profile in validated_profiles]
    matrix = _concatenate_rank_matrices(matrices, tuple(candidates), max_ranking_length)
    return RankProfile._from_matrix(matrix, max_ranking_length=max_ranking_length)


def _sum_score_profiles(score_profiles: Sequence[PreferenceProfile]) -> ScoreProfile:
    """
    Helper function for sum_profiles that sums ScoreProfiles.

    Args:
        score_profiles (Sequence[PreferenceProfile]): The profiles to sum.

    Raises:
        TypeError: Each profile must be of ScoreProfile type
    """

    from votekit.pref_profile.pref_profile import ScoreProfile

    if len(score_profiles) == 1 and isinstance(score_profiles[0], ScoreProfile):
        return score_profiles[0].copy()

    if not (all(isinstance(p, ScoreProfile) for p in score_profiles)):
        invalid_profiles = [
            (i, type(p).__name__)
            for i, p in enumerate(score_profiles)
            if not isinstance(p, ScoreProfile)
        ]
        invalid_profiles_str = ", ".join(f"index {i} ({t})" for i, t in invalid_profiles)
        raise TypeError(
            "All profiles must be of the same type, ScoreProfile. "
            f"non-ScoreProfiles found at: {invalid_profiles_str}"
        )

    candidates = tuple(
        sort_candidates_pseudo_lexicographically(
            set().union(*(set(profile.candidates) for profile in score_profiles))
        )
    )
    validated_profiles = cast(Sequence[ScoreProfile], score_profiles)
    matrices = [profile._matrix for profile in validated_profiles]
    return ScoreProfile._from_matrix(_concatenate_score_matrices(matrices, candidates))


def sum_profiles(profiles: Sequence[PreferenceProfile]) -> PreferenceProfile:
    """
    Combine profiles of the same type by concatenating their matrix rows.

    Args:
        profiles (Sequence[PreferenceProfile]): The profiles to sum.

    Returns:
        PreferenceProfile: A new PreferenceProfile object containing the combined profiles.

    Raises:
        ValueError: Cannot sum an empty list of profiles.
        TypeError: Can only sum profiles of type RankProfile or ScoreProfile.
    """

    from votekit.pref_profile.pref_profile import RankProfile, ScoreProfile

    if len(profiles) == 0:
        raise ValueError("Cannot sum an empty list of profiles.")

    if isinstance(profiles[0], RankProfile):
        return _sum_rank_profiles(profiles)

    elif isinstance(profiles[0], ScoreProfile):
        return _sum_score_profiles(profiles)

    else:
        raise TypeError(
            f"Cannot sum profiles of type {type(profiles[0]).__name__}. "
            "List can only contain RankProfiles or ScoreProfiles."
        )


def subtract_profiles(
    base_profile: PreferenceProfile,
    subtracted_profiles: PreferenceProfile | Sequence[PreferenceProfile],
) -> PreferenceProfile:
    """
    Subtracts the ballots of one or more profiles from a base profile, matched by identical
    rankings or scores.

    The base profile is grouped, then each group loses the combined weight of the matching
    subtracted ballots. A ballot is exhausted when its weight goes to 0. A ballot group whose weight
    is exhausted is removed; any weight left over from the subtracted ballots is ignored, as are
    subtracted ballots with no match in the base. Rankings match when they are equal after padding
    with ``frozenset({'~'})``, so profiles with different ranking lengths can be subtracted. The
    result keeps the base profile's candidates, ranking length, voter sets, and weight
    representation.

    Args:
        base_profile (PreferenceProfile): The profile being subtracted from.
        subtracted_profiles (PreferenceProfile | Sequence[PreferenceProfile]): The profile or
            profiles whose ballots are removed from ``base_profile``.

    Returns:
        PreferenceProfile: The grouped base profile with the subtracted ballots removed.

    Raises:
        TypeError: The profiles are not all RankProfiles or all ScoreProfiles.
        ValueError: ``subtracted_profiles`` is an empty sequence, or a subtracted weight cannot
            be represented exactly in the base profile's weight representation, e.g. a
            non-integral float subtracted from Fraction weights, or ``Fraction(1, 3)``
            subtracted from float weights.
        ProfileError: The subtracted profiles mix Fraction weights with non-integral float
            weights, so they cannot be summed.
    """

    from votekit.pref_profile.pref_profile import PreferenceProfile, RankProfile, ScoreProfile

    if not isinstance(base_profile, (RankProfile, ScoreProfile)):
        raise TypeError(
            f"Cannot subtract profiles of type {type(base_profile).__name__}. "
            "Only RankProfiles or ScoreProfiles are supported."
        )

    if isinstance(subtracted_profiles, PreferenceProfile):
        subtracted_profile = subtracted_profiles
    else:
        subtracted_profile = sum_profiles(subtracted_profiles)

    if isinstance(base_profile, RankProfile):
        return _subtract_rank_profiles(base_profile, subtracted_profile)

    return _subtract_score_profiles(base_profile, subtracted_profile)


def _as_weight_kind(subtracted_matrix: MatrixT, base_kind: WeightKind) -> MatrixT:
    """
    Convert a subtracted matrix's weights to the base profile's representation without rounding.

    Args:
        subtracted_matrix (RankMatrix | ScoreMatrix): Matrix of a subtracted profile.
        base_kind (WeightKind): Weight representation of the base profile. Can be FRACTION or FLOAT.

    Returns:
        RankMatrix | ScoreMatrix: Matrix with weights stored as the base profile's weights kind.

    Raises:
        ValueError: A weight cannot be represented exactly as the base profile's weights kind.
    """
    data = subtracted_matrix.voter_weight_data
    if data.weight_kind is base_kind:
        return subtracted_matrix

    if base_kind is WeightKind.FRACTION:
        weights = np.empty(len(data), dtype=object)
        weights[:] = [_as_fraction(weight) for weight in data.weights]
    else:
        weights = data.weights.astype(np.float64)
        for weight, float_weight in zip(data.weights, weights):
            if Fraction(float_weight) != weight:
                raise ValueError(
                    f"Cannot subtract weight {weight} from float weights without rounding. "
                    "Use Fraction weights in the base profile to subtract it exactly."
                )

    return replace(subtracted_matrix, voter_weight_data=data.with_weights(weights))


def _subtract_grouped_weights(
    base_data: ProfileVoterWeightData,
    first_rows: NDArray[np.intp],
    group_ids: NDArray[np.intp],
    subtracted_weights: NDArray,
) -> tuple[NDArray[np.intp], ProfileVoterWeightData]:
    """
    Group the base profile rows and subtract the weight of matching rows in subtracted_weights.

    Group IDs follow first-appearance order with base rows preceding subtracted rows. Base groups
    occupy IDs ``0..k-1``; subtracted-only groups have IDs at least ``k``. Only base groups can
    survive, and their voter sets come solely from the base profile.

    Args:
        base_data (ProfileVoterWeightData): Shared data of the base profile.
        first_rows (NDArray[np.intp]): Representative row of each group of a combined matrix
            whose leading rows are the base profile's rows.
        group_ids (NDArray[np.intp]): Group of each combined row. Groups are numbered in
            first-appearance order, so the base rows use exactly groups ``0..k-1``.
        subtracted_weights (NDArray): Weights of the combined rows after the base rows, in the
            base profile's weight representation.

    Returns:
        tuple[NDArray[np.intp], ProfileVoterWeightData]: Representative base row of each
            retained group, and the grouped shared data with the subtracted weight removed.
    """
    num_base_rows = len(base_data)
    base_group_ids = group_ids[:num_base_rows]
    num_base_groups = int(base_group_ids.max(initial=-1)) + 1
    grouped_base_data = base_data.group_rows(first_rows[:num_base_groups], base_group_ids)

    subtracted_group_ids = group_ids[num_base_rows:]
    matched_groups = subtracted_group_ids < num_base_groups
    weights_to_subtract = np.zeros(num_base_groups, dtype=grouped_base_data.weights.dtype)
    np.add.at(
        weights_to_subtract,
        subtracted_group_ids[matched_groups],
        subtracted_weights[matched_groups],
    )
    if grouped_base_data.weight_kind is WeightKind.FRACTION:
        remaining_group_weights = np.subtract(grouped_base_data.weights, weights_to_subtract)
    else:
        # use compensated summation for floating point weights
        remaining_group_weights = grouped_base_data.weights.copy()

        signed_weights = np.concatenate((base_data.weights, -subtracted_weights))
        signed_weights = signed_weights[np.argsort(group_ids, kind="stable")]

        group_offsets = np.concatenate(([0], np.cumsum(np.bincount(group_ids))))
        for group in np.flatnonzero(weights_to_subtract):
            start, end = group_offsets[group : group + 2]
            remaining_group_weights[group] = fsum(signed_weights[start:end])

    # Floating-point subtraction can leave a residue of an exhausted ballot. Drop with tolerance.
    zero_tolerance = 0 if grouped_base_data.weight_kind is WeightKind.FRACTION else TOL_EPSILON
    keep_groups = np.array(
        (weights_to_subtract == 0) | (remaining_group_weights > zero_tolerance), dtype=np.bool_
    )
    voter_sets = (
        None if grouped_base_data.voter_sets is None else grouped_base_data.voter_sets[keep_groups]
    )

    return first_rows[:num_base_groups][keep_groups], ProfileVoterWeightData(
        remaining_group_weights[keep_groups], voter_sets, grouped_base_data.weight_kind
    )


def _subtract_rank_profiles(
    base_profile: PreferenceProfile, subtracted_profile: PreferenceProfile
) -> RankProfile:
    """
    Helper function for subtract_profiles that subtracts RankProfiles.

    Ballots in the subtracted profile are removed from the grouped base profile in accordance with
    their weight. If a ballot in the subtracted profile does not exist in the base profile, the base
    profile is unaltered. If the weight of a base profile's ballot is exhausted, the ballot is
    removed entirely and any remaining weight of that ballot in the subtracted profile is ignored.

    Args:
        base_profile (PreferenceProfile): Base profile to remove ballots from.
        subtracted_profile (PreferenceProfile): Profile with ballots to subtract from the base
            profile.

    Returns:
        RankProfile: Grouped base profile with the subtracted profile's ballots removed.

    Raises:
        TypeError: Both profiles must be of type RankProfile.
        ValueError: A subtracted weight cannot be represented exactly in the base's weights.
    """
    from votekit.pref_profile.pref_profile import RankProfile

    if not isinstance(base_profile, RankProfile) or not isinstance(subtracted_profile, RankProfile):
        raise TypeError(
            f"Both profiles must be of the same type, RankProfile. Got "
            f"{type(base_profile).__name__} and {type(subtracted_profile).__name__}."
        )

    base_matrix = base_profile._matrix
    subtract_matrix = _as_weight_kind(
        subtracted_profile._matrix, base_matrix.voter_weight_data.weight_kind
    )

    # Concatenation maps every profile onto one candidate-set table and pads rankings with
    # end-of-ranking markers, so equal rankings fall into one group across profiles.
    candidates = tuple(set(base_matrix.candidates).union(subtract_matrix.candidates))
    width = max(base_matrix.num_ranking_positions, subtract_matrix.num_ranking_positions)
    combined = _concatenate_rank_matrices([base_matrix, subtract_matrix], candidates, width)
    first_rows, group_ids = combined._group_ranking_rows()

    rows, data = _subtract_grouped_weights(
        base_matrix.voter_weight_data,
        first_rows,
        group_ids,
        combined.voter_weight_data.weights[len(base_matrix) :],  # subtract_matrix weights
    )
    matrix = RankMatrix(
        voter_weight_data=data,
        rankings=base_matrix.rankings[rows],
        cand_sets=base_matrix.cand_sets,
        ranking_columns=base_matrix.ranking_columns,
        candidates=base_matrix.candidates,
    )
    return RankProfile._from_matrix(matrix, max_ranking_length=base_profile.max_ranking_length)


def _subtract_score_profiles(
    base_profile: PreferenceProfile, subtracted_profile: PreferenceProfile
) -> ScoreProfile:
    """
    Helper function for subtract_profiles that subtracts ScoreProfiles.

    Ballots in the subtracted profile are removed from the grouped base profile in accordance with
    their weight. If a ballot in the subtracted profile does not exist in the base profile, the base
    profile is unaltered. If the weight of a base profile's ballot is exhausted, the ballot is
    removed entirely and any remaining weight of that ballot in the subtracted profile is ignored.

    Args:
        base_profile (PreferenceProfile): Base profile to remove ballots from.
        subtracted_profile (PreferenceProfile): Profile with ballots to subtract from the base
            profile.

    Returns:
        ScoreProfile: Grouped base profile with the subtracted profile's ballots removed.

    Raises:
        TypeError: Both profiles must be of type ScoreProfile.
        ValueError: A subtracted weight cannot be represented exactly in the base's weights.
    """
    from votekit.pref_profile.pref_profile import ScoreProfile

    if not isinstance(base_profile, ScoreProfile) or not isinstance(
        subtracted_profile, ScoreProfile
    ):
        raise TypeError(
            f"Both profiles must be of the same type, ScoreProfile. Got "
            f"{type(base_profile).__name__} and {type(subtracted_profile).__name__}."
        )

    base_matrix = base_profile._matrix
    other_matrix = _as_weight_kind(
        subtracted_profile._matrix, base_matrix.voter_weight_data.weight_kind
    )

    # Concatenation aligns score columns by candidate, leaving unscored entries as NaN, which
    # grouping treats as equal.
    comparison_matrices = [
        replace(matrix, candidates=matrix.score_columns) for matrix in (base_matrix, other_matrix)
    ]
    candidates = tuple(
        sort_candidates_pseudo_lexicographically(
            set(base_matrix.score_columns).union(other_matrix.score_columns)
        )
    )
    combined = _concatenate_score_matrices(comparison_matrices, candidates)
    first_rows, group_ids = combined._group_score_rows()

    rows, data = _subtract_grouped_weights(
        base_matrix.voter_weight_data,
        first_rows,
        group_ids,
        combined.voter_weight_data.weights[len(base_matrix) :],
    )
    matrix = ScoreMatrix(
        voter_weight_data=data,
        scores=base_matrix.scores[rows],
        score_columns=base_matrix.score_columns,
        candidates=base_matrix.candidates,
    )
    return ScoreProfile._from_matrix(matrix)
