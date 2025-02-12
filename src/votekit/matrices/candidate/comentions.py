from ...ballot import Ballot
from ...pref_profile import PreferenceProfile
from .._utils import _convert_dict_to_matrix
import numpy as np
from fractions import Fraction
from typing import Union


def comention(cands: Union[str, list[str]], ballot: Ballot):
    """
    Takes cands and returns true if they all appear on the ballot, either in the ranking
    or the scoring. Candidates who receive 0 points are not counted as mentioned.

    Args:
      cands (Union[str, list[str]]): Candidate name or list of candidate names.
      ballot (Ballot): Ballot.

    Returns:
      bool: True if all candidates appear in ballot.
    """

    all_cands: set[str] = set()
    if ballot.scores:
        all_cands = all_cands.union(ballot.scores.keys())

    if ballot.ranking:
        all_cands = all_cands.union(c for s in ballot.ranking for c in s)

    if isinstance(cands, str):
        cands = [cands]

    return set(cands).issubset(all_cands)


def comention_above(i: str, j: str, ballot: Ballot) -> bool:
    """
    Takes candidates i,j and returns True if i >= j in the ranking.
    Requires that the ballot has a ranking.


    Args:
      i (str): Candidate name.
      j (str): Candidate name.
      ballot (Ballot): Ballot.

    Returns:
      bool: True if both i and j appear in ballot and i >= j.
    """

    if not ballot.ranking:
        raise TypeError(f"Ballot must have a ranking: {ballot}")
    i_index, j_index = (-1, -1)

    for rank, s in enumerate(ballot.ranking):
        if i in s:
            i_index = rank
        if j in s:
            j_index = rank

    return (i_index >= 0 and j_index >= 0) and (i_index <= j_index)


def comentions_matrix(
    pref_profile: PreferenceProfile, candidates: list[str], symmetric: bool = False
) -> np.ndarray:
    """
    Takes a preference profile and converts to a matrix
    where the i,j entry shows the number of times candidates i,j were mentioned on the same
    ballot with i >= j. There is an option to make it symmetric so that the i,j entry is just
    the number of times candidates i and j were mentioned on the same ballot.

    Args:
      pref_profile (PreferenceProfile): Profile.
      candidates (list[str]): List of candidates to use. Indexing of this list matches indexing of
        output array.
      symmetric (bool, optional): Whether or not to make the matrix symmetric. Defaults to False
        in which case the i,j entry is comentions where i >= j. True means the i,j entry is
        comentions of i,j.

    Returns:
      np.ndarray: Numpy array of comentions.
    """
    comentions_matrix = {c: {c: Fraction(0) for c in candidates} for c in candidates}

    for i in candidates:
        for j in candidates:
            for ballot in pref_profile.ballots:

                if symmetric:
                    if comention([i, j], ballot):
                        comentions_matrix[i][j] += ballot.weight
                else:
                    if comention_above(i, j, ballot):
                        comentions_matrix[i][j] += ballot.weight

    return _convert_dict_to_matrix(comentions_matrix)
