from votekit.pref_profile import ScoreProfile, CleanedScoreProfile, ProfileError
from typing import Callable
import pandas as pd


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
        idx for idx, (o, c) in zip(idxs, zip(orig_rows, cleaned_rows)) if o == c
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
        profile (RankProfile): A RankProfile to clean.
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


# TODO
def remove_cand_from_score_profile():
    pass
