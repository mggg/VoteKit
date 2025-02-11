from ..ballot import Ballot
from ..pref_profile import PreferenceProfile
from ._convert_dict_to_mat import _convert_dict_to_matrix
import numpy as np
from fractions import Fraction


def mention(i: str, ballot: Ballot) -> bool:
    """
    Returns true if candidate i is on the ballot, either in the ranking or the scoring.
    Candidates who receive 0 points are not counted as mentioned.

    Args:
      i (str): Candidate name.
      ballot (Ballot): Ballot.

    Returns:
      bool: True if i appears anywhere in ballot.
    """
    cands: set[str] = set()
    if ballot.scores:
        cands = cands.union(ballot.scores.keys())

    if ballot.ranking:
        cands = cands.union(c for s in ballot.ranking for c in s)

    return i in cands


def comention(i: str, j: str, ballot: Ballot) -> bool:
    """
    Takes candidates i,j and returns true if they both appear on the ballot, either in the ranking
    or the scoring. Candidates who receive 0 points are not counted as mentioned.

    Args:
      i (str): Candidate name.
      j (str): Candidate name.
      ballot (Ballot): Ballot.

    Returns:
      bool: True if both i and j appear in ballot.
    """

    cands: set[str] = set()
    if ballot.scores:
        cands = cands.union(ballot.scores.keys())

    if ballot.ranking:
        cands = cands.union(c for s in ballot.ranking for c in s)

    return i in cands and j in cands


def comention_above(i: str, j: str, ballot: Ballot) -> bool:
    """
    Takes candidates i,j and returns true if i appears tied with or before j in the ranking.
    Requires that the ballot has a ranking.


    Args:
      i (str): Candidate name.
      j (str): Candidate name.
      ballot (Ballot): Ballot.

    Returns:
      bool: True if both i and j appear in ballot and i is above j.
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
    Takes a preference profile and converts to a numpy array
    where the i,j entry shows the number of times candidates i,j were mentioned on the same
    ballot with i above j. There is an option to make it symmetric so that the i,j entry is just
    the comentions of i,j. Comentions are counted using ballot weight. The indexing of the
    matrix matches the indexing of ``candidates``.

    Args:
      pref_profile (PreferenceProfile): Profile.
      candidates (list[str]): List of candidates to use. Indexing of this list matches indexing of
        output array.
      symmetric (bool, optional): Whether or not to make the matrix symmetric. Defaults to False
        in which case the i,j entry is comentions where i>j. True means the i,j entry is comentions
        of i,j.

    Returns:
      np.ndarray: Numpy array of comentions.
    """
    comentions_matrix = {c: {c: Fraction(0) for c in candidates} for c in candidates}

    for i in candidates:
        for j in candidates:
            for ballot in pref_profile.ballots:

                if symmetric:
                    if comention(i, j, ballot):
                        comentions_matrix[i][j] += ballot.weight
                else:
                    if comention_above(i, j, ballot):
                        comentions_matrix[i][j] += ballot.weight

    return _convert_dict_to_matrix(comentions_matrix)
