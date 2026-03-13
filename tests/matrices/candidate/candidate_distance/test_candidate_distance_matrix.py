import numpy as np

from votekit.ballot import RankBallot
from votekit.matrices import candidate_distance_matrix
from votekit.pref_profile import RankProfile

ballot_1 = RankBallot(ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"})))
ballot_2 = RankBallot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})))
ballot_3 = RankBallot(ranking=(frozenset({"Chris"}), frozenset({"Moon"})))
ballot_4 = RankBallot(ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"})))


pref_profile = RankProfile(
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
