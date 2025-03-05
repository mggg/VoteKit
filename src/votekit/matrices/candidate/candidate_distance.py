from ...ballot import Ballot
from ...pref_profile import PreferenceProfile
import numpy as np
import itertools as it
from .._utils import _convert_dict_to_matrix


def candidate_distance(i: str, j: str, ballot: Ballot) -> float:
    """
    Takes candidates i,j and returns distance r(j)-r(i) in ranking.
    Returns numpy.nan if a candidate is not on ballot. Note that this is non-symmetric,
    and that a positive value indicates that i is ranked higher than j.

    Args:
      i (str): Candidate.
      j (str): Candidate.
      ballot (Ballot): Ballot.

    Returns:
      float: Distance r(j)-r(i) in ranking.
    """
    if not ballot.ranking:
        raise TypeError("Ballot must have a ranking.")

    positions = {i: -1, j: -1}

    for position, s in enumerate(ballot.ranking):
        if i in s:
            positions[i] = position
        if j in s:
            positions[j] = position

    if -1 in positions.values():
        return np.nan
    else:
        return positions[j] - positions[i]


def candidate_distance_matrix(
    pref_profile: PreferenceProfile, candidates: list[str]
) -> np.ndarray:
    """
    Takes a preference profile and converts to a matrix
    where the i,j entry shows the average distance between i and j when i >= j on the same
    ballot. Computations use ballot weight. Non-symmetric.
    Uses numpy.nan for undefined entries.

    Args:
      pref_profile (PreferenceProfile): Profile.
      candidates (list[str]): List of candidates to use. Indexing of this list matches indexing of
          output array.

    Returns:
        np.ndarray: Numpy array of average distances.
    """

    dist_matrix = {c: {c: 0.0 for c in candidates} for c in candidates}
    weight_matrix = {c: {c: 0.0 for c in candidates} for c in candidates}
    avg_dist_matrix = {c: {c: 0.0 for c in candidates} for c in candidates}

    for i, j in it.combinations_with_replacement(candidates, 2):
        for ballot in pref_profile.ballots:
            d = candidate_distance(i, j, ballot)

            # i >= j
            if d >= 0:
                dist_matrix[i][j] += d * ballot.weight
                weight_matrix[i][j] += ballot.weight

            # i < j
            elif d < 0:
                dist_matrix[j][i] += (-d) * ballot.weight
                weight_matrix[j][i] += ballot.weight

    for c, row in dist_matrix.items():
        for k, v in row.items():
            avg_dist_matrix[c][k] = (
                float(v / weight_matrix[c][k]) if weight_matrix[c][k] > 0 else np.nan
            )

    return _convert_dict_to_matrix(avg_dist_matrix)
