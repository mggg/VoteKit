import numpy as np

from votekit.ballot import RankBallot
from votekit.matrices._utils import _convert_dict_to_matrix
from votekit.pref_profile import RankProfile
from votekit.types import Candidate, CandidateListLike


def comention(cands: Candidate | CandidateListLike, ballot: RankBallot):
    """
    Takes cands and returns true if they all appear on the ballot in the ranking.

    Args:
      cands (Candidate | list[Candidate] | list[str] | list[int]):
        Candidate name or list of candidate names.
        Candidates can be strings, integers, or mix of both.
      ballot (RankBallot): RankBallot.

    Returns:
      bool: True if all candidates appear in ballot.
    """
    all_cands: set[Candidate] = set()

    if ballot.ranking:
        all_cands = all_cands.union(c for s in ballot.ranking for c in s)

    if isinstance(cands, Candidate):
        cands = [cands]

    return set(cands).issubset(all_cands)


def comention_above(cand_a: Candidate, cand_b: Candidate, ballot: RankBallot) -> bool:
    """
    Takes two candidates and returns True if cand_a >= cand_b in the ranking.
    Requires that the ballot has a ranking.


    Args:
      above_cand (Candidate): Candidate to check as ranked at or above cand_b.
        Candidates can be strings, integers, or mix of both.
      below_cand (Candidate): Candidate to check as ranked at or below cand_a.
        Candidates can be strings, integers, or mix of both.
      ballot (RankBallot): RankBallot.

    Returns:
      bool: True if both cand_a and cand_b appear in ballot and cand_a >= cand_b.
    """
    if not isinstance(ballot, RankBallot):
        raise TypeError("Ballot must be of type RankBallot.")
    if ballot.ranking is None:
        raise TypeError(f"RankBallot must have a ranking: {ballot}")
    cand_a_index, cand_b_index = (-1, -1)

    for rank, s in enumerate(ballot.ranking):
        if cand_a in s:
            cand_a_index = rank
        if cand_b in s:
            cand_b_index = rank

    return (cand_a_index >= 0 and cand_b_index >= 0) and (cand_a_index <= cand_b_index)


def comentions_matrix(
    pref_profile: RankProfile, candidates: list[Candidate], symmetric: bool = False
) -> np.ndarray:
    """
    Takes a preference profile and converts to a matrix
    where the i,j entry shows the number of times candidates i,j were mentioned on the same
    ballot with i >= j. There is an option to make it symmetric so that the i,j entry is just
    the number of times candidates i and j were mentioned on the same ballot.

    Args:
      pref_profile (RankProfile): Profile.
      candidates (list[Candidate]): List of candidates to use. Indexing of this list matches
        indexing of output array. Candidates can be strings, integers, or mix of both.
      symmetric (bool, optional): Whether or not to make the matrix symmetric. Defaults to False
        in which case the i,j entry is comentions where i >= j. True means the i,j entry is
        comentions of i,j.

    Returns:
      np.ndarray: Numpy array of comentions.
    """
    comentions_matrix = {c: {c: 0.0 for c in candidates} for c in candidates}

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
