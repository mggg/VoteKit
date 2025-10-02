from votekit.pref_profile import ScoreProfile, CleanedScoreProfile, ProfileError
from typing import Callable
import pandas as pd
from typing import Union
from functools import partial
import numpy as np


def _iterate_and_clean_score_tuples(
    profile: ScoreProfile,
    clean_score_func: Callable[[tuple], tuple],
) -> tuple[pd.DataFrame, set[int], set[int], set[int], set[int]]:
    """
    Clean the rows of the df according to the cleaning rule. Note that this function
    only allows you to edit the score tuple, not any other part of the ballot.

    Args:
        profile (ScoreProfile): The original profile of ballots.
        clean_score_func (Callable[[tuple], tuple]): Function that
            takes the score portion of a row of the profile df and returns an altered score tuple.

    Returns:
        tuple[pd.DataFrame, set[int], set[int], set[int], set[int]]: cleaned_df, unaltr_idxs,
            nonempty_altr_idxs, no_wt_altr_idxs, no_scores_altr_idxs
    """
    cleaned_df = profile.df.copy()
    candidate_cols = [c for c in cleaned_df.columns if c not in ["Weight", "Voter Set"]]

    orig_rows = [
        tuple(vals)
        for vals in cleaned_df[candidate_cols].itertuples(index=False, name=None)
    ]
    cleaned_rows = [clean_score_func(row) for row in orig_rows]

    cleaned_df[candidate_cols] = pd.DataFrame(cleaned_rows, index=cleaned_df.index)
    idxs = cleaned_df.index

    unaltr_idxs = {
        idx
        for idx, (o, c) in zip(idxs, zip(orig_rows, cleaned_rows))
        if np.array_equal(o, c, equal_nan=True)
    }
    no_scores_altr_idxs = {
        idx
        for idx, score_tuple in zip(idxs, cleaned_rows)
        if all(score == 0 or pd.isna(score) for score in score_tuple)
    }
    nonempty_altr_idxs = set(idxs) - unaltr_idxs - no_scores_altr_idxs
    no_wt_altr_idxs: set[int] = set()

    return (
        cleaned_df,
        unaltr_idxs,
        nonempty_altr_idxs,
        no_wt_altr_idxs,
        no_scores_altr_idxs,
    )


def clean_score_profile(
    profile: ScoreProfile,
    clean_score_func: Callable[[tuple], tuple],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = True,
) -> CleanedScoreProfile:
    """
    Allows user-defined cleaning rules for ScoreProfile. Cleaning function can only
    be applied to the score section of the dataframe, not the weight or voter set.


    Args:
        profile (ScoreProfile): A ScoreProfile to clean.
        clean_score_func (Callable[[tuple], tuple]): Function that
            takes the score portion of a row of the profile df and returns an altered score.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            no scores as a result of the cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of the cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the original profile in the new profile. If False, uses only candidates who receive
            votes. Defaults to True.

    Returns:
        CleanedScoreProfile: A cleaned ``ScoreProfile``.

    Raises:
        ProfileError: Profile must only contain ranked ballots.
    """
    if not isinstance(profile, ScoreProfile):
        raise ProfileError("Profile must be a ScoreProfile.")

    (
        cleaned_df,
        unaltr_idxs,
        nonempty_altr_idxs,
        no_wt_altr_idxs,
        no_scores_altr_idxs,
    ) = _iterate_and_clean_score_tuples(profile, clean_score_func)

    if remove_empty_ballots:
        candidate_cols = [
            c for c in cleaned_df.columns if c not in ["Weight", "Voter Set"]
        ]
        mask = (
            cleaned_df[candidate_cols]
            .map(lambda score: score == 0 or pd.isna(score))  # type: ignore[operator]
            .all(axis=1)
        )

        cleaned_df = cleaned_df[~mask]

    if remove_zero_weight_ballots:
        cleaned_df = cleaned_df[cleaned_df["Weight"] > 0]

    return CleanedScoreProfile(
        df=cleaned_df,
        candidates=(profile.candidates if retain_original_candidate_list else tuple()),
        no_wt_altr_idxs=no_wt_altr_idxs,
        no_scores_altr_idxs=no_scores_altr_idxs,
        nonempty_altr_idxs=nonempty_altr_idxs,
        unaltr_idxs=unaltr_idxs,
        parent_profile=profile,
        df_index_column=list(cleaned_df.index),
    )


def remove_cand_from_score_tuple(
    removed_idxs: list[int],
    score_tup: tuple[float, ...],
) -> tuple[float, ...]:
    """
    Removes specified candidate(s) from score tuple.

    Args:
        removed_idxs (list[int]): Indices of candidates to be removed.
        ranking_tup (tuple): Ranking to remove candidates from.

    Returns:
        tuple: Ranking with candidate(s) removed.
    """
    return tuple(
        s if i not in removed_idxs else np.nan for i, s in enumerate(score_tup)
    )


def remove_cand_score_profile(
    removed: Union[str, list],
    profile: ScoreProfile,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
    retain_original_candidate_list: bool = False,
) -> CleanedScoreProfile:
    """
    Given a score profile, remove the given candidate(s) from the ballots.

    Wrapper for clean_score_profile that does some extra processing to ensure the candidate list
    is handled correctly.

    Not as fast as removing the candidate columns directly, but tracks more information
    about which ballots were adjusted.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile (ScoreProfile): Profile to remove candidates from.
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.
        retain_original_candidate_list (bool, optional): Whether or not to use the candidate list
            from the orginal profile in the new profile. If False, takes the original candidate
            list and removes the candidate(s) given in ``removed``, but preserves all others.
            Defaults to False.

    Returns:
        CleanedScoreProfile: A cleaned ``ScoreProfile``.

    Raises:
        ProfileError: Profile must only contain score ballots.
    """
    if isinstance(removed, str):
        removed = [removed]
    removed_idxs = [i for i, c in enumerate(profile.df.columns) if c in removed]

    cleaned_profile = clean_score_profile(
        profile,
        partial(remove_cand_from_score_tuple, removed_idxs),
        remove_empty_ballots,
        remove_zero_weight_ballots,
        retain_original_candidate_list=True,
    )

    new_candidates = (
        [c for c in profile.candidates if c not in removed]
        if not retain_original_candidate_list
        else list(profile.candidates)
    )
    new_df = (
        cleaned_profile.df[new_candidates + ["Voter Set", "Weight"]]
        if not retain_original_candidate_list
        else cleaned_profile.df
    )

    return CleanedScoreProfile(
        df=new_df,
        candidates=new_candidates,
        parent_profile=cleaned_profile.parent_profile,
        df_index_column=cleaned_profile.df_index_column,
        no_wt_altr_idxs=cleaned_profile.no_wt_altr_idxs,
        no_scores_altr_idxs=cleaned_profile.no_scores_altr_idxs,
        nonempty_altr_idxs=cleaned_profile.nonempty_altr_idxs,
        unaltr_idxs=cleaned_profile.unaltr_idxs,
    )
