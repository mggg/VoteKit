from matplotlib.axes import Axes

from votekit.ballot import RankBallot
from votekit.matrices import candidate_distance_matrix, matrix_heatmap
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


def test_heatmap():
    mat = candidate_distance_matrix(pref_profile, ["Chris", "Peter", "Moon"])

    assert isinstance(matrix_heatmap(mat), Axes)
