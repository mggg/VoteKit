from functools import partial
from typing import Callable, Union
from ...pref_profile import (
    PreferenceProfile,
    CleanedProfile,
    ProfileError,
)
import pandas as pd
import numpy as np


def _compute_altered_indices(
    cleaned_df: pd.DataFrame,
    profile: PreferenceProfile,
) -> tuple[set[int], set[int], set[int], set[int]]:
    """
    Compute which ballots were altered by the cleaning rule.

    Args:
        cleaned_df (pd.DataFrame): The cleaned df of ballots.
        profile (PreferenceProfile): The original profile of ballots.

    Returns:
        tuple[set[int], set[int], set[int], set[int]]: unaltr_idxs, nonempty_altr_idxs,
            no_wt_altr_idxs, no_rank_no_score_altr_idxs
    """

    no_wt_altr_idxs: set[int] = set()
    no_rank_no_score_altr_idxs: set[int] = set()
    nonempty_altr_idxs: set[int] = set()
    unaltr_idxs: set[int] = set()

    mask = ~(cleaned_df.eq(profile.df) | (cleaned_df.isna() & profile.df.isna()))
    disagree_rows = mask.any(axis=1)

    for i, row in disagree_rows.items():
        assert isinstance(i, int)
        if row is False:
            unaltr_idxs.add(i)
            continue

        ballot = cleaned_df.loc[i]
        if ballot["Weight"] == 0:
            no_wt_altr_idxs.add(i)

        if ballot.drop(["Weight", "Voter Set"]).isna().all():
            no_rank_no_score_altr_idxs.add(i)

        else:
            nonempty_altr_idxs.add(i)

    return unaltr_idxs, nonempty_altr_idxs, no_wt_altr_idxs, no_rank_no_score_altr_idxs


def clean_ranked_profile(
    profile: PreferenceProfile,
    clean_ranking_func: Callable[[tuple], tuple],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
) -> CleanedProfile:
    """
    Allows user-defined cleaning rules for ranked PreferenceProfile. Input function
    that applies modification to a single ballot.

    By using the underlying dataframe, we make speed improvements as compared to the cleaning
    functions that rely on the ballot list.

    Args:
        profile (PreferenceProfile): A PreferenceProfile to clean.
        clean_ranking_func (Callable[[tuple], tuple]): Function that
            takes the ranking portion of a row of the profile df and returns an altered ranking.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking and no scores as a result of the cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of the cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.

    Raises:
        ProfileError: Profile must only contain ranked ballots.
    """
    if profile.contains_scores is True:
        raise ProfileError("Profile must only contain ranked ballots.")

    cleaned_df = profile.df.copy()
    ranking_cols = [f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)]
    new_ranking_cols: list[tuple] = [
        tuple(None for _ in range(profile.max_ranking_length))
    ] * len(cleaned_df)

    for idx, row in enumerate(cleaned_df[ranking_cols].itertuples(index=False)):
        new_ranking_cols[idx] = clean_ranking_func(row)

    cleaned_df[ranking_cols] = pd.DataFrame(new_ranking_cols, index=cleaned_df.index)
    unaltr_idxs, nonempty_altr_idxs, no_wt_altr_idxs, no_rank_no_score_altr_idxs = (
        _compute_altered_indices(cleaned_df, profile)
    )

    if remove_empty_ballots:
        cleaned_df = cleaned_df[
            ~cleaned_df.drop(["Weight", "Voter Set"], axis=1).isna().all(axis=1)
        ]

    if remove_zero_weight_ballots:
        cleaned_df = cleaned_df[cleaned_df["Weight"] > 0]

    return CleanedProfile(
        df=cleaned_df,
        contains_rankings=profile.contains_rankings,
        contains_scores=profile.contains_scores,
        candidates=(profile.candidates if retain_original_candidate_list else tuple()),
        max_ranking_length=profile.max_ranking_length,
        no_wt_altr_idxs=no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=no_rank_no_score_altr_idxs,
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
        new_position = []
        if not isinstance(cand_set, frozenset):
            dedup_ranking.append(np.nan)
            continue

        for cand in cand_set:
            if cand not in seen_cands:
                new_position.append(cand)
                seen_cands.append(cand)

        dedup_ranking.append(frozenset(new_position))

    return tuple(dedup_ranking)


def remove_repeat_cands_ranked_profile(
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
) -> CleanedProfile:
    """
    Given a profile, if a candidate appears multiple times on a ballot, keep the first instance and
    remove any further instances. Does not condense any empty rankings as as result.
    Only works on ranking ballots, not score ballots.

    Wrapper for clean_ranked_profile.

    Args:
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.

    Raises:
        ProfileError: Profile must only contain ranked ballots.
    """

    return clean_ranked_profile(
        profile,
        remove_repeat_cands_from_ranking_row,
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list,
    )


def remove_cand_from_ranking_row(
    removed: Union[str, list],
    ranking_tup: tuple,
) -> tuple:
    """
    Removes specified candidate(s) from ranking. Does not condense the resulting ranking.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        ranking_tup (tuple): Ranking to remove candidates from.

    Returns:
        tuple: Ranking with candidate(s) removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    new_ranking: list[float | frozenset] = []
    for s in ranking_tup:
        new_s = []
        if not isinstance(s, frozenset):
            new_ranking.append(np.nan)
            continue

        for c in s:
            if c not in removed:
                new_s.append(c)
        new_ranking.append(frozenset(new_s))

    return tuple(new_ranking)


def remove_cand_ranked_profile(
    removed: Union[str, list],
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
) -> CleanedProfile:
    """
    Given a ranked profile, remove the given candidate(s) from the ballots. Does not condense the
    resulting ballots.

    Wrapper for clean_ranked_profile that does some extra processing to ensure the candidate list
    is handled correctly.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the orginal profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.

    Raises:
        ProfileError: Profile must only contain ranked ballots.
    """
    if isinstance(removed, str):
        removed = [removed]

    cleaned_profile = clean_ranked_profile(
        profile,
        partial(remove_cand_from_ranking_row, removed),
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list=True,
    )

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    return CleanedProfile(
        df=cleaned_profile.df,
        candidates=new_candidates,
        contains_rankings=cleaned_profile.contains_rankings,
        max_ranking_length=cleaned_profile.max_ranking_length,
        parent_profile=cleaned_profile.parent_profile,
        df_index_column=cleaned_profile.df_index_column,
        no_wt_altr_idxs=cleaned_profile.no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=cleaned_profile.no_rank_no_score_altr_idxs,
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
        tuple: Condensed tanking.

    """
    max_ranking_length = len(ranking_tup)
    condensed_ranking = [
        cand_set for cand_set in ranking_tup if cand_set != frozenset()
    ]

    if len(condensed_ranking) < max_ranking_length:
        condensed_ranking += [np.nan] * (max_ranking_length - len(condensed_ranking))

    return tuple(condensed_ranking)


def _is_equiv_to_condensed(ranking: pd.Series) -> bool:
    """
    Returns True if the given ranking is equivalent to its condensed form. It is equivalent
    if the rankings are identical, or if the original ranking only has trailing empty frozensets
    in its ranking after some listed candidate.

    Args:
        ranking (pd.Series): Ranking to check.

    Returns:
        bool: True if the given ranking is equivalent to its condensed form.
    """
    if all(cs == frozenset() for cs in ranking):
        return False

    for i, cand_set in enumerate(ranking):
        if cand_set != frozenset():
            continue

        if all(cs in [frozenset(), np.nan] for cs in ranking[i:]):
            return True

        return False

    return True


def condense_ranked_profile(
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
) -> CleanedProfile:
    """
    Given a ranked profile, removes any empty frozensets from the rankings and condenses the
    resulting ranking. If a ranking only has trailing empty positions, the condensed ranking is
    considered equivalent. For example, (A,B,{},{}) is mapped to (A,B) but considered unaltered
    since the ranking did not change.

    Wrapper for clean_ranked_profile that does some extra processing to ensure condensed ranking
    equivalence is handled correctly.

    Args:
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.

    """
    condensed_profile = clean_ranked_profile(
        profile,
        condense_ranking_row,
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list,
    )

    ranking_cols = [f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)]
    ranking_df = profile.df[ranking_cols]
    additional_unaltr_idxs = set(
        [
            i
            for i in condensed_profile.nonempty_altr_idxs
            if _is_equiv_to_condensed(ranking_df.loc[i])
        ]
    )

    new_unaltr_idxs = condensed_profile.unaltr_idxs | additional_unaltr_idxs
    new_nonempty_altr_idxs = condensed_profile.nonempty_altr_idxs.difference(
        additional_unaltr_idxs
    )

    return CleanedProfile(
        df=condensed_profile.df,
        contains_rankings=condensed_profile.contains_rankings,
        candidates=condensed_profile.candidates,
        max_ranking_length=condensed_profile.max_ranking_length,
        parent_profile=profile,
        df_index_column=condensed_profile.df_index_column,
        no_wt_altr_idxs=condensed_profile.no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=condensed_profile.no_rank_no_score_altr_idxs,
        nonempty_altr_idxs=new_nonempty_altr_idxs,
        unaltr_idxs=new_unaltr_idxs,
    )


def _is_equiv_for_remove_and_condense(removed: list[str], ranking: pd.Series) -> bool:
    """
    Returns True if the given ranking is equivalent to its removed and condensed form.
    It is equivalent if the ranking has no candidate in the removed list and either no empty
    frozensets or only trailing ones. If its has internal empty frozensets or any candidate
    in the removed list, it is not equivalent.

    Args:
        removed (list[str]): Candidates to be removed.
        ranking (pd.Series): Ranking to check.

    Returns:
        bool: True if the given ranking is equivalent to its remove and condensed form.
    """

    if any(
        c_remove == cand for c_remove in removed for c_set in ranking for cand in c_set
    ):
        return False

    if all(c_set != frozenset() for c_set in ranking):
        return True

    for i, cand_set in enumerate(ranking):
        if cand_set != frozenset():
            continue

        if all(cs == frozenset() for cs in ranking[i:]):
            return True

        return False

    return True


def remove_and_condense_ranked_profile(
    removed: Union[str, list],
    profile: PreferenceProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
) -> CleanedProfile:
    """
    Given a ranked profile, remove the given candidate(s) and condense the
    resulting rankings. If a ranking only has trailing empty positions, the condensed ranking is
    considered equivalent. For example, (A,B,{},{}) is mapped to (A,B) but considered unaltered
    since the ranking did not change.

    This function is intended to save computational time in election methods, where removing
    and condensing happen frequently. Researches interested in the difference between
    removing and condensing should use ``remove_cand`` and ``condense_profile`` in series.

    Wrapper for clean_ranked_profile that does some extra processing to ensure the candidate list
    is handled correctly, and that ballot equivalence is checked.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the orginal profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.

    Returns:
        CleanedProfile: A cleaned ``PreferenceProfile``.
    """

    if isinstance(removed, str):
        removed = [removed]

    cleaned_profile = clean_ranked_profile(
        profile,
        lambda b: condense_ranking_row(remove_cand_from_ranking_row(removed, b)),
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list=True,
    )

    new_candidates = (
        profile.candidates
        if retain_original_candidate_list
        else tuple(set(profile.candidates) - set(removed))
    )

    ranking_cols = [f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)]
    ranking_df = profile.df[ranking_cols]

    additional_unaltr_idxs = set(
        [
            i
            for i in cleaned_profile.nonempty_altr_idxs
            if _is_equiv_for_remove_and_condense(
                removed,
                ranking_df.loc[i],
            )
        ]
    )

    new_unaltr_idxs = cleaned_profile.unaltr_idxs | additional_unaltr_idxs
    new_nonempty_altr_idxs = cleaned_profile.nonempty_altr_idxs.difference(
        additional_unaltr_idxs
    )

    return CleanedProfile(
        df=cleaned_profile.df,
        candidates=new_candidates,
        contains_rankings=cleaned_profile.contains_rankings,
        max_ranking_length=cleaned_profile.max_ranking_length,
        parent_profile=cleaned_profile.parent_profile,
        df_index_column=cleaned_profile.df_index_column,
        no_wt_altr_idxs=cleaned_profile.no_wt_altr_idxs,
        no_rank_no_score_altr_idxs=cleaned_profile.no_rank_no_score_altr_idxs,
        nonempty_altr_idxs=new_nonempty_altr_idxs,
        unaltr_idxs=new_unaltr_idxs,
    )
