from ...pref_profile import PreferenceProfile
from .comentions import comention
import numpy as np
from typing import Tuple
from .._utils import _convert_dict_to_matrix


def boost_prob(i: str, j: str, pref_profile: PreferenceProfile) -> Tuple[float, float]:
    """
    Takes candidates i,j and a preference profile and computes the conditional
    P(mention i | mention j) and P(mention i). If candidate j is never mentioned,
    or if the profile has 0 total ballot weight, it returns
    numpy.nan for the respective probability.

    Args:
      i (str): Candidate.
      j (str): Candidate.
      pref_profile (PreferenceProfile): Profile.

    Returns:
      tuple[float, float]: P(mention i | mention j), P(mention i)
    """

    i_mentions = 0.0
    j_mentions = 0.0
    both_mentions = 0.0

    for ballot in pref_profile.ballots:
        if comention(i, ballot):
            i_mentions += ballot.weight

        if comention(j, ballot):
            j_mentions += ballot.weight

        if comention([i, j], ballot):
            both_mentions += ballot.weight

    return (
        float(both_mentions) / j_mentions if (j_mentions != 0) else np.nan,
        (
            float(i_mentions) / pref_profile.total_ballot_wt
            if (pref_profile.total_ballot_wt != 0)
            else np.nan
        ),
    )


def boost_matrix(pref_profile: PreferenceProfile, candidates: list[str]) -> np.ndarray:
    """
    Takes a profile and converts to a matrix
    where the i,j entry shows P(mention i | mention j) - P(mention i).
    Thus, the i,j entry shows the boost given to i by j.
    Computations use ballot weight. Non-symmetric matrix.
    Undefined entries are denoted with numpy.nan values.

    Args:
      pref_profile (PreferenceProfile): Profile.
      candidates (list[str]): List of candidates to use. Indexing of this list matches indexing of
          output array.

    Returns:
        np.ndarray: Numpy array of boosts.
    """
    boost_matrix = {c: {c: 0.0 for c in candidates} for c in candidates}

    for i in candidates:
        for j in candidates:
            if i != j:
                cond, uncond = boost_prob(i, j, pref_profile)
                boost_matrix[i][j] = cond - uncond
            else:
                boost_matrix[i][j] = np.nan

    return _convert_dict_to_matrix(boost_matrix)
