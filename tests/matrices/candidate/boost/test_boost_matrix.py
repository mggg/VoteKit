from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.matrices import boost_matrix
import numpy as np


ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"}))
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})))
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),))
pref_profile = PreferenceProfile(
    ballots=tuple(
        [ballot_1 for _ in range(5)]
        + [ballot_2 for _ in range(2)]
        + [ballot_3 for _ in range(1)]
    )
)


def test_boost_matrix():
    mat = boost_matrix(pref_profile, ["Chris", "Peter", "Moon"])

    assert np.isnan(mat[0][0])
    assert mat[0][1] == 5 / 7 - 3 / 4
    assert mat[2][0] == 5 / 6 - 7 / 8


def test_boost_matrix_nan():
    mat = boost_matrix(pref_profile, ["Chris", "Mala"])

    assert np.isnan(mat[0][0])
    assert np.isnan(mat[0][1])
    assert mat[1][0] == 0
