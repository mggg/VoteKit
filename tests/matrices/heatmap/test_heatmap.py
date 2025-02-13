from votekit.matrices import matrix_heatmap, candidate_distance_matrix
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from matplotlib.axes import Axes

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


def test_heatmap():
    mat = candidate_distance_matrix(pref_profile, ["Chris", "Peter", "Moon"])

    assert isinstance(matrix_heatmap(mat), Axes)
