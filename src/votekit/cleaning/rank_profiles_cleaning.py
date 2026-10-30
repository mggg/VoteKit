from functools import partial
from typing import Callable

import numpy as np

from votekit.pref_profile import (
    CleanedRankProfile,
    RankProfile,
)
from votekit.pref_profile.profile_matrix import EMPTY_SET_ID, END_OF_RANKING_ID, RankMatrix
from votekit.types import Candidate, CandidateList
from votekit.utils import _validate_candidate_names, sort_candidates_pseudo_lexicographically


def _iterate_and_clean_ranking_tuples(
    profile: RankProfile,
    clean_ranking_func: Callable[[tuple[frozenset, ...]], tuple[frozenset, ...]],
) -> tuple[RankMatrix, set[int], set[int], set[int], set[int]]:
    """
    Clean the ranking tuples of the profile according to the cleaning rule.

    Rankings are decoded to candidate-set tuples for ``clean_ranking_func`` and the results are
    encoded back to candidate-set IDs. Sets already in the profile keep their IDs and new sets are
    appended, so a ranking is unaltered exactly when its ID row is unchanged. Each distinct ranking
    is cleaned once and the result is shared by every ballot with that ranking.

    Args:
        profile (RankProfile): The original profile of ballots.
        clean_ranking_func (Callable[[tuple[frozenset, ...]], tuple[frozenset, ...]]): Function that
            takes the ranking portion of a row of the profile df and returns an altered ranking.

    Returns:
        tuple[RankMatrix, set[int], set[int], set[int], set[int]]: cleaned RankMatrix, unaltr_idxs,
            nonempty_altr_idxs, no_wt_altr_idxs, no_rank_altr_idxs

    Raises:
        ValueError: A cleaned ranking does not have one entry per ranking position, or the cleaned
            rankings are not a valid rank matrix.
    """
    matrix = profile._matrix
    num_positions = matrix.num_ranking_positions

    lookup = np.empty(len(matrix.cand_sets), dtype=object)
    lookup[:] = matrix.cand_sets
    cand_set_to_id = {
        cand_set: cand_set_id for cand_set_id, cand_set in matrix.iter_candidate_sets()
    }

    # Cleaning depends only on the ranking, so clean each distinct ranking once.
    first_rows_idx, group_ids = matrix._group_ranking_rows()
    cleaned_groups = np.empty((len(first_rows_idx), num_positions), dtype=np.intp)

    for group, ranking in enumerate(lookup[matrix.rankings[first_rows_idx] - EMPTY_SET_ID]):
        cleaned_ranking = clean_ranking_func(tuple(ranking))
        if len(cleaned_ranking) != num_positions:
            raise ValueError(
                f"Cleaned rankings must have {num_positions} positions, one per ranking column. "
                f"Got {len(cleaned_ranking)}: {cleaned_ranking}"
            )
        cleaned_groups[group] = [
            cand_set_to_id.setdefault(cand_set, len(cand_set_to_id) + EMPTY_SET_ID)
            for cand_set in cleaned_ranking
        ]

    cleaned_rankings = cleaned_groups[group_ids]

    labels = matrix.voter_weight_data.ballot_index
    labels = np.arange(len(matrix)) if labels is None else labels

    unaltr_idxs = set(labels[np.all(cleaned_rankings == matrix.rankings, axis=1)].tolist())
    no_rank_altr_idxs = set(labels[~np.any(cleaned_rankings >= 0, axis=1)].tolist()) - unaltr_idxs
    nonempty_altr_idxs = set(labels.tolist()) - unaltr_idxs - no_rank_altr_idxs
    no_wt_altr_idxs: set[int] = set()

    cleaned_matrix = RankMatrix(
        voter_weight_data=matrix.voter_weight_data,
        rankings=cleaned_rankings,
        cand_sets=tuple(cand_set_to_id),
        ranking_columns=matrix.ranking_columns,
        candidates=profile.candidates,
    )

    return (
        cleaned_matrix,
        unaltr_idxs,
        nonempty_altr_idxs,
        no_wt_altr_idxs,
        no_rank_altr_idxs,
    )


def clean_rank_profile(
    profile: RankProfile,
    clean_ranking_func: Callable[[tuple[frozenset, ...]], tuple[frozenset, ...]],
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    *,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Allows user-defined cleaning rules for RankProfile. Input function
    that applies modification to ranking portion of the df, not weight or voterset.

    Args:
        profile (RankProfile): A RankProfile to clean.
        clean_ranking_func (Callable[[tuple[frozenset, ...]], tuple[frozenset, ...]]): Function that
            takes the ranking portion of a row of the profile df and returns an altered ranking.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of the cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    Raises:
        TypeError: Profile must only contain ranked ballots.
    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be a RankProfile.")

    return _clean_rank_profile_with_candidates(
        profile,
        clean_ranking_func,
        candidates=profile.candidates if retain_original_candidate_list else (),
        remove_null_ballots=remove_null_ballots,
        remove_zero_weight_ballots=remove_zero_weight_ballots,
        reduce_max_ranking_length=reduce_max_ranking_length,
    )


def _clean_rank_profile_with_candidates(
    profile: RankProfile,
    clean_ranking_func: Callable[[tuple[frozenset, ...]], tuple[frozenset, ...]],
    *,
    candidates: tuple[Candidate, ...],
    remove_null_ballots: bool,
    remove_zero_weight_ballots: bool,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Clean and filter rankings, then construct one profile with the final candidate universe.

    Args:
        profile (RankProfile): Source profile whose labels and cleaning provenance are retained.
        clean_ranking_func (Callable[[tuple[frozenset, ...]], tuple[frozenset, ...]]):
            Transformation applied to each ranking row.
        candidates (tuple[Candidate, ...]): Candidate universe for the result. An empty tuple
            infers candidates from retained positive-weight ballots.
        remove_null_ballots (bool): Whether to discard rankings containing only end markers.
        remove_zero_weight_ballots (bool): Whether to discard rows with zero weight.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: Cleaned profile with the source as its immediate parent.
    """
    (
        cleaned_matrix,
        unaltr_idxs,
        nonempty_altr_idxs,
        no_wt_altr_idxs,
        no_rank_altr_idxs,
    ) = _iterate_and_clean_ranking_tuples(profile, clean_ranking_func)

    keep_rows = np.ones(len(cleaned_matrix), dtype=np.bool_)
    if remove_null_ballots:
        keep_rows &= cleaned_matrix.has_rankings()

    if remove_zero_weight_ballots:
        keep_rows &= cleaned_matrix.voter_weight_data.weights > 0

    cleaned_matrix = cleaned_matrix.apply_mask(keep_rows)

    max_ranking_length = profile.max_ranking_length
    if reduce_max_ranking_length:
        max_ranking_length = min(
            profile.max_ranking_length, _reduced_max_ranking_length(cleaned_matrix)
        )

    if not candidates:
        candidates = tuple(
            sort_candidates_pseudo_lexicographically(cleaned_matrix.get_candidates_cast())
        )

    # Validates the final candidate universe and the truncated ranking width.
    cleaned_matrix = RankMatrix(
        voter_weight_data=cleaned_matrix.voter_weight_data,
        rankings=cleaned_matrix.rankings[:, :max_ranking_length],
        cand_sets=cleaned_matrix.cand_sets,
        ranking_columns=cleaned_matrix.ranking_columns[:max_ranking_length],
        candidates=candidates,
    )

    return CleanedRankProfile._from_cleaned_matrix(
        cleaned_matrix,
        max_ranking_length=max_ranking_length,
        parent_profile=profile,
        no_wt_altr_idxs=no_wt_altr_idxs,
        no_rank_altr_idxs=no_rank_altr_idxs,
        nonempty_altr_idxs=nonempty_altr_idxs,
        unaltr_idxs=unaltr_idxs,
    )


def _remove_repeat_cands_from_ranking_row(
    ranking_tup: tuple[frozenset, ...],
) -> tuple[frozenset, ...]:
    """
    Given a ranking tuple, if a candidate appears multiple times, keep the first instance,
    and remove any further instances. Does not condense the ranking.

    Args:
        ranking_tup (tuple[frozenset, ...]): Ranking to remove repeated candidates from.

    Returns:
        tuple[frozenset, ...]: Ranking with duplicate candidate(s) removed.
    """

    dedup_ranking: list[frozenset] = []
    seen_cands = []

    for cand_set in ranking_tup:
        if cand_set == frozenset({"~"}):
            dedup_ranking.append(frozenset({"~"}))
            continue
        if not isinstance(cand_set, frozenset):
            dedup_ranking.append(np.nan)
            continue

        new_position = []
        for cand in cand_set:
            if cand not in seen_cands:
                new_position.append(cand)
                seen_cands.append(cand)

        dedup_ranking.append(frozenset(new_position))

    return tuple(dedup_ranking)


def remove_repeat_cands_rank_profile(
    profile: RankProfile,
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    *,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Given a profile, if a candidate appears multiple times on a ballot, keep the first instance and
    remove any further instances.

    The repeated candidate is replaced with an empty set if ranked alone or
    removed from the set if tied. Can condense the ballots after removing repeated candidate(s)
    by calling ``condense_rank_profile`` on the result.

    Wrapper for clean_rank_profile.

    Args:
        profile (RankProfile): Profile to remove repeated candidates from.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    Raises:
        TypeError: Profile must only contain ranked ballots.
    """

    return clean_rank_profile(
        profile,
        _remove_repeat_cands_from_ranking_row,
        remove_null_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list=retain_original_candidate_list,
        reduce_max_ranking_length=reduce_max_ranking_length,
    )


def _remove_cand_from_ranking_row(
    removed: Candidate | CandidateList,
    ranking_tup: tuple[frozenset, ...],
) -> tuple[frozenset, ...]:
    """
    Removes specified candidate(s) from ranking. Does not condense the resulting ranking.

    Args:
        removed (Candidate | list[Candidate] | list[str] | list[int]):
            Candidate or list of candidates to be removed.
            Candidates can be strings, integers, or mix of both.
        ranking_tup (tuple[frozenset, ...]): Ranking to remove candidates from.

    Returns:
        tuple[frozenset, ...]: Ranking with candidate(s) removed.
    """
    if isinstance(removed, Candidate):
        removed = [removed]

    removed_set = set(removed)

    out = []
    for s in ranking_tup:
        if s.isdisjoint(removed_set):
            out.append(s)
        else:
            out.append(frozenset(s - removed_set))
    return tuple(out)


def remove_cand_rank_profile(
    removed: Candidate | list[Candidate],
    profile: RankProfile,
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
    *,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Given a ranked profile, remove the given candidate(s) from the ballots. Does not condense the
    resulting ballots.

    Wrapper for clean_rank_profile that does some extra processing to ensure the candidate list
    is handled correctly. A removed candidate is replaced with an empty slot if not tied. The
    profile's max_ranking_length stays the same unless ``reduce_max_ranking_length`` is True. Can
    condense the ballots after removing candidate(s) by calling ``condense_rank_profile`` on the
    result.

    Args:
        removed (Candidate | list[Candidate]): Candidate or list of candidates to be removed.
            Candidates can be strings, integers, or mix of both.
        profile (RankProfile): Profile to remove candidates from.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    Raises:
        TypeError: Profile must only contain ranked ballots.
    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be a RankProfile.")

    if isinstance(removed, Candidate):
        removed = [removed]

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    return _clean_rank_profile_with_candidates(
        profile,
        partial(_remove_cand_from_ranking_row, removed),
        candidates=new_candidates,
        remove_null_ballots=remove_null_ballots,
        remove_zero_weight_ballots=remove_zero_weight_ballots,
        reduce_max_ranking_length=reduce_max_ranking_length,
    )


def _truncate_ranking_row_at_cand(
    truncated: Candidate | CandidateList,
    ranking_tup: tuple[frozenset, ...],
) -> tuple[frozenset, ...]:
    """
    Truncate a ranking at the first position containing the specified candidate(s).

    The matching position and every position below it are replaced with trailing ``~``
    placeholders so that the ranking keeps its original ranking length.

    Args:
        truncated (Candidate | list[Candidate]): Candidate or list of candidates at
            which to truncate. Must already have passed candidate-name validation; this helper does
            not validate identifiers.
        ranking_tup (tuple[frozenset, ...]): Ranking to truncate.

    Returns:
        tuple[frozenset, ...]: Ranking truncated at the first matching position.
    """
    if isinstance(truncated, Candidate):
        truncated = [truncated]

    truncated_set = set(truncated)
    truncated_ranking: list[frozenset] = []

    for cand_set in ranking_tup:
        if not cand_set.isdisjoint(truncated_set):
            truncated_ranking.extend([frozenset("~")] * (len(ranking_tup) - len(truncated_ranking)))
            break
        truncated_ranking.append(cand_set)

    return tuple(truncated_ranking)


def truncate_at_cand_rank_profile(
    truncated: Candidate | CandidateList,
    profile: RankProfile,
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
    *,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Truncate ranked ballots at the first position containing the specified candidate(s).

    The ballots will be truncated at the first position containing any of the specified candidates
    even if tied with non-truncated candidates. The matching position and all lower-ranked positions
    are removed. Ballots without any of the truncated candidates remain unchanged.

    Args:
        truncated (Candidate | list[Candidate]): Candidate or list of candidates at which to
            truncate the profile's rankings.
        profile (RankProfile): Profile to truncate.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove zero-weight ballots.
            Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to retain the original
            candidate list. If False, takes the original candidate list and removes the candidate(s)
            given in ``removed``, but preserves all others. Defaults to False.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    Raises:
        ProfileError: Profile must only contain ranked ballots.
    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be a RankProfile.")

    if isinstance(truncated, Candidate):
        truncated = [truncated]

    _validate_candidate_names(truncated, "truncated")

    candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(truncated))
    )

    return _clean_rank_profile_with_candidates(
        profile,
        partial(_truncate_ranking_row_at_cand, truncated),
        candidates=candidates,
        remove_null_ballots=remove_null_ballots,
        remove_zero_weight_ballots=remove_zero_weight_ballots,
        reduce_max_ranking_length=reduce_max_ranking_length,
    )


def _condense_ranking_row(
    ranking_tup: tuple[frozenset, ...],
) -> tuple[frozenset, ...]:
    """
    Given a ranking, removes any empty ranking positions and moves up any lower ranked candidates.

    Args:
        ranking_tup (tuple[frozenset, ...]): Ranking to condense.

    Returns:
        tuple[frozenset, ...]: Condensed ranking.

    """
    max_ranking_length = len(ranking_tup)
    condensed_ranking = [cand_set for cand_set in ranking_tup if cand_set != frozenset()]

    if len(condensed_ranking) < max_ranking_length:
        condensed_ranking += [frozenset("~")] * (max_ranking_length - len(condensed_ranking))

    return tuple(condensed_ranking)


def condense_rank_profile(
    profile: RankProfile,
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    *,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Given a ranked profile, removes any empty frozensets from the rankings and condenses the
    resulting ranking.

    Rankings with empty frozensets (``frozenset()``) will be handled as follows:
    - Empty frozensets between ranking positions with candidate sets are removed and remaining
    ranking positions are shifted up to fill the gaps. These ranking's indices will be recorded in
    ``nonempty_altr_idxs``.
    - Trailing empty frozensets, that have no ranking positions with candidate sets following them,
    are replaced with tilde frozensets, ``frozenset('~')``. If the ranking had at least one ranking
    position with a candidate set followed by trailing empty frozenset(s), then the ranking's index
    will be recorded in ``nonempty_altr_idxs``.
    - Rankings with only empty frozensets or a mix of empty frozensets with tilde frozensets will be
    replaced entirely with tilde frozensets and recorded in ``no_rank_altr_idxs``.

    Wrapper for clean_rank_profile.

    Args:
        profile (RankProfile): Profile to condense rankings.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    """
    return clean_rank_profile(
        profile,
        _condense_ranking_row,
        remove_null_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list,
        reduce_max_ranking_length=reduce_max_ranking_length,
    )


def remove_and_condense_rank_profile(
    removed: Candidate | CandidateList,
    profile: RankProfile,
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
    *,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Given a ranked profile, remove the given candidate(s) and condense the resulting rankings.

    Rankings with empty frozensets (``frozenset()``) will be handled as follows:
    - Empty frozensets between ranking positions with candidate sets are removed and remaining
    ranking positions are shifted up to fill the gaps. These ranking's indices will be recorded in
    ``nonempty_altr_idxs``.
    - Trailing empty frozensets, that have no ranking positions with candidate sets following them,
    are replaced with tilde frozensets, ``frozenset('~')``. If the ranking had at least one ranking
    position with a candidate set followed by trailing empty frozenset(s), then the ranking's index
    will be recorded in ``nonempty_altr_idxs``.
    - Rankings with only empty frozensets or a mix of empty frozensets with tilde frozensets will be
    replaced entirely with tilde frozensets and recorded in ``no_rank_altr_idxs``.

    This function is intended to save computational time in election methods, where removing
    and condensing happen frequently. Researches interested in the difference between
    removing and condensing should use ``remove_cand`` and ``condense_profile`` in series.

    Cleans and filters rankings before constructing one profile with the final candidate list.

    Args:
        removed (Candidate | list[Candidate] | list[str] | list[int]):
            Candidate or list of candidates to be removed.
            Candidates can be strings, integers, or mix of both.
        profile (RankProfile): Profile to remove candidates from and condense rankings.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.
    """

    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be a RankProfile.")

    if isinstance(removed, Candidate):
        removed = [removed]

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    return _clean_rank_profile_with_candidates(
        profile,
        lambda ranking: _condense_ranking_row(_remove_cand_from_ranking_row(removed, ranking)),
        candidates=new_candidates,
        remove_null_ballots=remove_null_ballots,
        remove_zero_weight_ballots=remove_zero_weight_ballots,
        reduce_max_ranking_length=reduce_max_ranking_length,
    )


def remove_ballots_with_cand_rank_profile(
    removed: Candidate | list[Candidate],
    profile: RankProfile,
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
    *,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Given a ranked profile, remove the ballots that contain the given candidate(s).

    A removed ballot is not considered altered after cleaning and is accounted for in the difference
    between the parent profile's index versus the indices of the returned
    ``CleanedRankProfile``. The same goes for removed empty ranking and zero weight ballots.

    Args:
        removed (Candidate | list[Candidate]): Candidate or list of candidates to remove their
            ballots. Candidates can be strings, integers, or mix of both.
        profile (RankProfile): Profile to remove ballots from.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the minimum
            number of columns necessary to represent all rankings, or to the maximum number of
            unique candidates ranked in any ballot within the profile. Whichever is higher.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile`` with ballots containing the specified
            candidate(s) removed.

    Raises:
        TypeError: Profile must contain ranked ballots.
        TypeError: Candidates to be removed must be strings or integers. A boolean or float
            candidate of the same value as an integer candidate would result in removing the ballots
            with that integer candidate.
    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be a RankProfile.")

    if isinstance(removed, Candidate):
        removed = [removed]
    _validate_candidate_names(removed, "removed")

    cand_set_ids = []
    for cand in removed:
        cand_set_ids.extend(
            [
                cand_set_id
                for cand_set_id, cand_set in profile._matrix.iter_candidate_sets()
                if cand in cand_set
            ]
        )
    ballots_to_remove = np.any(np.isin(profile._matrix.rankings, cand_set_ids), axis=1)

    keep_rows = np.ones(len(profile._matrix), dtype=np.bool_)
    keep_rows &= ~ballots_to_remove
    if remove_null_ballots:
        keep_rows &= profile._matrix.has_rankings()

    if remove_zero_weight_ballots:
        keep_rows &= profile._matrix.voter_weight_data.weights > 0

    cleaned_matrix = profile._matrix.apply_mask(keep_rows)

    max_ranking_length = profile.max_ranking_length
    if reduce_max_ranking_length:
        max_ranking_length = min(
            profile.max_ranking_length, _reduced_max_ranking_length(cleaned_matrix)
        )

    candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    # Validates the final candidate universe and the truncated ranking width.
    cleaned_matrix = RankMatrix(
        voter_weight_data=cleaned_matrix.voter_weight_data,
        rankings=cleaned_matrix.rankings[:, :max_ranking_length],
        cand_sets=cleaned_matrix.cand_sets,
        ranking_columns=cleaned_matrix.ranking_columns[:max_ranking_length],
        candidates=candidates,
    )

    orig_idxs = profile._matrix.voter_weight_data.ballot_index

    return CleanedRankProfile._from_cleaned_matrix(
        cleaned_matrix,
        max_ranking_length=max_ranking_length,
        parent_profile=profile,
        no_wt_altr_idxs=set(),
        no_rank_altr_idxs=set(),
        nonempty_altr_idxs=set(),
        unaltr_idxs=set(range(len(profile._matrix)) if orig_idxs is None else orig_idxs.tolist()),
    )


def _reduced_max_ranking_length(cleaned_matrix: RankMatrix) -> int:
    """
    The minimum max_ranking_length to represent all ballots of the profile after cleaning.

    After cleaning, the max_ranking_length can be reduced to the minimum number of columns necessary
    to represent all rankings. A profile's max_ranking_length has a lower bound: maximum number of
    unique candidates ranked on any ballot. When there are tied candidates within a ballot, the
    number of candidates ranked can be greater than the minimum number of ranking columns to
    represent all ballots. This function returns the maximum of these two values.

    Args:
        rankings (RankMatrix): Cleaned rank matrix.

    Returns:
        int: Maximum ranking length.
    """
    min_num_ranking_columns = int(
        (cleaned_matrix.rankings != END_OF_RANKING_ID).sum(axis=1).max(initial=0)
    )
    return int(max(min_num_ranking_columns, cleaned_matrix.max_candidates_ranked()))
