from typing import List, Tuple
from ...pref_profile import (
    PreferenceProfile,
)
import numpy as np

def remove_cand_scored(removed: List[str] | str, profile: PreferenceProfile, remove_empty_ballots: bool = True, remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile:
    """
    Faster version of remove_cand for score profiles.

    Args:
        removed (List[str] or str): List of candidates to be removed from the profile.
        profile (PreferenceProfile): The original preference profile.
        remove_empty_ballots (bool, optional): If True, removes ballots with no votes.
        remove_zero_weight_ballots (bool, optional): If True, removes ballots with zero weight.

    Returns:
        PreferenceProfile: A new profile with the specified candidates removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    # pull out candidate list, df, and weight vector
    all_cands_list = list(profile.candidates_cast)
    kept_cands_list = [c for c in all_cands_list if c not in removed]
    kept_cands_tuple: Tuple[str, ...] = tuple(kept_cands_list) # preferred type for PreferenceProfile
    df = profile.df.drop(columns=removed)

    # Remove zero-weight ballots
    if remove_zero_weight_ballots:
        df = df[df["Weight"] > 0]

    if remove_empty_ballots:
        candidate_matrix = df[kept_cands_list].to_numpy()
        mask = (np.nansum(candidate_matrix, axis=1) > 0)
        df = df[mask]

    return PreferenceProfile(df=df, 
                            candidates=kept_cands_tuple, 
                            contains_scores=profile.contains_scores,
                            contains_rankings=False)