from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.matrices import candidate_distance_matrix
import numpy as np


ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"}))
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})))
ballot_3 = Ballot(ranking=(frozenset({"Chris"}), frozenset({"Moon"})))
ballot_4 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"}))
)


pref_profile = PreferenceProfile(
    ballots=tuple(
        [ballot_1 for _ in range(5)]
        + [ballot_2 for _ in range(2)]
        + [ballot_3 for _ in range(1)]
        + [ballot_4 for _ in range(3)]
    )
)


def test_candidate_dist_matrix():
    mat = candidate_distance_matrix(pref_profile, ["Chris", "Moon", "Peter"])

    assert mat[0][0] == 0
    assert mat[0][1] == 17 / 9
    assert mat[0][2] == 1
    assert np.isnan(mat[1][0])


def test_candidate_dist_matrix_nan_values():
    mat = candidate_distance_matrix(pref_profile, ["Chris", "Mala"])

    assert mat[0][0] == 0
    assert np.isnan(mat[0][1])
    assert np.isnan(mat[1][0])
    assert np.isnan(mat[1][1])
