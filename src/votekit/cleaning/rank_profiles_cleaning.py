from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from votekit._df_schema import WEIGHT_COLUMN
from votekit.pref_profile import (
    CleanedRankProfile,
    RankProfile,
)
from votekit.types import Candidate, CandidateList
from votekit.utils import _validate_candidate_names


def _iterate_and_clean_ranking_tuples(
    profile: RankProfile,
    clean_ranking_func: Callable[[tuple], tuple],
) -> tuple[pd.DataFrame, set[int], set[int], set[int], set[int]]:
    """
    Clean the ranking tuples of the df according to the cleaning rule.

    Args:
        profile (RankProfile): The original profile of ballots.
        clean_ranking_func (Callable[[tuple], tuple]): Function that
            takes the ranking portion of a row of the profile df and returns an altered ranking.

    Returns:
        tuple[pd.DataFrame, set[int], set[int], set[int], set[int]]: cleaned_df, unaltr_idxs,
            nonempty_altr_idxs, no_wt_altr_idxs, no_rank_altr_idxs
    """
    cleaned_df = profile.to_df()
    assert profile.max_ranking_length is not None
    ranking_cols = list(profile._matrix.ranking_columns)

    orig_rows = [
        tuple(vals) for vals in cleaned_df[ranking_cols].itertuples(index=False, name=None)
    ]
    cleaned_rows = [clean_ranking_func(row) for row in orig_rows]

    cleaned_df[ranking_cols] = pd.DataFrame(cleaned_rows, index=cleaned_df.index)

    tilde = frozenset({"~"})
    empty = frozenset()
    idxs = cleaned_df.index

    unaltr_idxs = {
        idx
        for idx, (orig_row, cleaned_row) in zip(idxs, zip(orig_rows, cleaned_rows))
        if orig_row == cleaned_row
    }
    no_rank_altr_idxs = {
        idx
        for idx, cleaned_row in zip(idxs, cleaned_rows)
        if all(cand_set == tilde or cand_set == empty for cand_set in cleaned_row)
    }
    no_rank_altr_idxs = no_rank_altr_idxs - unaltr_idxs
    nonempty_altr_idxs = set(idxs) - unaltr_idxs - no_rank_altr_idxs
    no_wt_altr_idxs: set[int] = set()

    return (
        cleaned_df,
        unaltr_idxs,
        nonempty_altr_idxs,
        no_wt_altr_idxs,
        no_rank_altr_idxs,
    )


def clean_rank_profile(
    profile: RankProfile,
    clean_ranking_func: Callable[[tuple], tuple],
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Allows user-defined cleaning rules for RankProfile. Input function
    that applies modification to ranking portion of the df, not weight or voterset.

    Args:
        profile (RankProfile): A RankProfile to clean.
        clean_ranking_func (Callable[[tuple], tuple]): Function that
            takes the ranking portion of a row of the profile df and returns an altered ranking.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of the cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce to the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the max
            candidates ranked of any ballot within the profile which is the minimum allowed.
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
    clean_ranking_func: Callable[[tuple], tuple],
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
        clean_ranking_func (Callable[[tuple], tuple]): Transformation applied to each ranking row.
        candidates (tuple[Candidate, ...]): Candidate universe for the result. An empty tuple
            infers candidates from retained positive-weight ballots.
        remove_null_ballots (bool): Whether to discard rankings containing only end markers.
        remove_zero_weight_ballots (bool): Whether to discard rows with zero weight.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce to the profile's
        max_ranking_length.

    Returns:
        CleanedRankProfile: Cleaned profile with the source as its immediate parent.
    """
    (
        cleaned_df,
        unaltr_idxs,
        nonempty_altr_idxs,
        no_wt_altr_idxs,
        no_rank_altr_idxs,
    ) = _iterate_and_clean_ranking_tuples(profile, clean_ranking_func)

    if remove_null_ballots:
        assert profile.max_ranking_length is not None
        ranking_cols = list(profile._matrix.ranking_columns)

        mask = cleaned_df[ranking_cols].map(lambda x: x == frozenset({"~"})).all(axis=1)

        cleaned_df = cleaned_df[~mask]

    if remove_zero_weight_ballots:
        cleaned_df = cleaned_df[cleaned_df[WEIGHT_COLUMN] > 0]

    return CleanedRankProfile(
        df=cleaned_df,
        candidates=candidates,
        max_ranking_length=profile.max_ranking_length
        if not reduce_max_ranking_length
        else profile.max_candidates_ranked,
        no_wt_altr_idxs=no_wt_altr_idxs,
        no_rank_altr_idxs=no_rank_altr_idxs,
        nonempty_altr_idxs=nonempty_altr_idxs,
        unaltr_idxs=unaltr_idxs,
        parent_profile=profile,
        df_index_column=list(cleaned_df.index),
    )


def remove_repeat_cands_from_ranking_row(
    ranking_tup: tuple,
) -> tuple:
    """
    Given a ranking tuple, if a candidate appears multiple times, keep the first instance,
    and remove any further instances. Does not condense the ranking.

    Args:
        ranking_tup (tuple): Ranking to remove repeated candidates from.

    Returns:
        tuple: Ranking with duplicate candidate(s) removed.
    """

    dedup_ranking: list[float | frozenset] = []
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
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Given a profile, if a candidate appears multiple times on a ballot, keep the first instance and
    remove any further instances. Does not condense any empty rankings as as result.
    Only works on ranking ballots, not score ballots.

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
        reduce_max_ranking_length (bool, optional): Whether or not to reduce to the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the max
            candidates ranked of any ballot within the profile which is the minimum allowed.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    Raises:
        TypeError: Profile must only contain ranked ballots.
    """

    return clean_rank_profile(
        profile,
        remove_repeat_cands_from_ranking_row,
        remove_null_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list,
        reduce_max_ranking_length,
    )


def remove_cand_from_ranking_row(
    removed: Candidate | CandidateList,
    ranking_tup: tuple[frozenset, ...],
) -> tuple[frozenset, ...]:
    """
    Removes specified candidate(s) from ranking. Does not condense the resulting ranking.

    Args:
        removed (Candidate | list[Candidate] | list[str] | list[int]):
            Candidate or list of candidates to be removed.
            Candidates can be strings, integers, or mix of both.
        ranking_tup (tuple): Ranking to remove candidates from.

    Returns:
        tuple: Ranking with candidate(s) removed.
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
    reduce_max_ranking_length: bool = False,
) -> CleanedRankProfile:
    """
    Given a ranked profile, remove the given candidate(s) from the ballots. Does not condense the
    resulting ballots.

    Wrapper for clean_rank_profile that does some extra processing to ensure the candidate list
    is handled correctly.

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
        reduce_max_ranking_length (bool, optional): Whether or not to reduce to the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the max
            candidates ranked of any ballot within the profile which is the minimum allowed.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    Raises:
        TypeError: Profile must only contain ranked ballots.
    """
    if isinstance(removed, Candidate):
        removed = [removed]

    cleaned_profile = clean_rank_profile(
        profile,
        partial(remove_cand_from_ranking_row, removed),
        remove_null_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list=True,
    )

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    return CleanedRankProfile(
        df=cleaned_profile.to_df(),
        candidates=new_candidates,
        max_ranking_length=cleaned_profile.max_ranking_length
        if not reduce_max_ranking_length
        else profile.max_candidates_ranked,
        parent_profile=cleaned_profile.parent_profile,
        df_index_column=cleaned_profile.df_index_column,
        no_wt_altr_idxs=cleaned_profile.no_wt_altr_idxs,
        no_rank_altr_idxs=cleaned_profile.no_rank_altr_idxs,
        nonempty_altr_idxs=cleaned_profile.nonempty_altr_idxs,
        unaltr_idxs=cleaned_profile.unaltr_idxs,
    )


def condense_ranking_row(
    ranking_tup: tuple,
) -> tuple:
    """
    Given a ranking, removes any empty ranking positions and moves up any lower ranked candidates.

    Args:
        ranking_tup (tuple): Ranking to condense.

    Returns:
        tuple: Condensed ranking.

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

    Wrapper for clean_rank_profile that does some extra processing to ensure condensed ranking
    equivalence is handled correctly.

    Args:
        profile (RankProfile): Profile to condense rankings.
        remove_null_ballots (bool, optional): Whether or not to remove ballots whose rankings are
            made of only tilde frozensets (``frozenset('~')``). Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.
        reduce_max_ranking_length (bool, optional): Whether or not to reduce to the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the max
            candidates ranked of any ballot within the profile which is the minimum allowed.
            Defaults to False.

    Returns:
        CleanedRankProfile: A cleaned ``RankProfile``.

    """
    condensed_profile = clean_rank_profile(
        profile,
        condense_ranking_row,
        remove_null_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list,
    )

    return CleanedRankProfile(
        df=condensed_profile.to_df(),
        candidates=condensed_profile.candidates,
        max_ranking_length=condensed_profile.max_ranking_length
        if not reduce_max_ranking_length
        else profile.max_candidates_ranked,
        parent_profile=profile,
        df_index_column=condensed_profile.df_index_column,
        no_wt_altr_idxs=condensed_profile.no_wt_altr_idxs,
        no_rank_altr_idxs=condensed_profile.no_rank_altr_idxs,
        nonempty_altr_idxs=condensed_profile.nonempty_altr_idxs,
        unaltr_idxs=condensed_profile.unaltr_idxs,
    )


def remove_and_condense_rank_profile(
    removed: Candidate | CandidateList,
    profile: RankProfile,
    remove_null_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
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
        reduce_max_ranking_length (bool, optional): Whether or not to reduce to the profile's
            max_ranking_length after cleaning. If True, the max_ranking_length is set to the max
            candidates ranked of any ballot within the profile which is the minimum allowed.
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
        lambda ranking: condense_ranking_row(remove_cand_from_ranking_row(removed, ranking)),
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
    ranking_cols = list(profile._matrix.ranking_columns)
    ballots_to_remove = np.any(np.isin(profile._matrix.rankings, cand_set_ids), axis=1)
    profile_df = profile.to_df()
    cleaned_df = profile_df[~ballots_to_remove]

    if remove_null_ballots:
        mask = cleaned_df[ranking_cols].map(lambda x: x == frozenset({"~"})).all(axis=1)
        cleaned_df = cleaned_df[~mask]

    if remove_zero_weight_ballots:
        cleaned_df = cleaned_df[cleaned_df[WEIGHT_COLUMN] > 0]

    candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    return CleanedRankProfile(
        df=cleaned_df,
        candidates=candidates,
        max_ranking_length=profile.max_ranking_length,
        parent_profile=profile,
        df_index_column=list(cleaned_df.index),
        no_wt_altr_idxs=set(),
        no_rank_altr_idxs=set(),
        nonempty_altr_idxs=set(),
        unaltr_idxs=set(profile_df.index),  # ballots are only dropped, never altered
    )
