import matplotlib.pyplot as plt
import pytest

from votekit.ballot import RankBallot
from votekit.plots.profiles import multi_profile_bar_plot
from votekit.pref_profile import RankProfile
from votekit.utils import first_place_votes

ballot_1 = RankBallot(ranking=(1, "A", "B"), weight=1)
ballot_2 = RankBallot(ranking=(1, 2), weight=4)
ballot_3 = RankBallot(ranking=("B",), weight=1)
ballot_4 = RankBallot(ranking=(2,), weight=1)

profile_1 = RankProfile(ballots=(ballot_1, ballot_2, ballot_3, ballot_4))
profile_2 = RankProfile(ballots=(ballot_1, ballot_2, ballot_4))

profile_dict = {"1": profile_1, "2": profile_2}


def _fig_for_plot(profile_dict, stat_function, **kwargs):
    fig, ax = plt.subplots()
    multi_profile_bar_plot(profile_dict, stat_function, ax=ax, **kwargs)
    fig.tight_layout()
    return fig


CASES = [
    pytest.param(
        profile_dict,
        first_place_votes,
        {},
        id="multi_profile_bar_plot_with_mixed_candidates.png",
        marks=pytest.mark.mpl_image_compare(
            baseline_dir="../../../snapshots/bar_plot",
            filename="multi_profile_bar_plot_with_mixed_candidates.png",
            tolerance=2,
        ),
    ),
]


@pytest.mark.parametrize("profile_dict,stat_function,kwargs", CASES)
def test_multi_profile_bar_plot_snapshots(profile_dict, stat_function, kwargs):
    fig = _fig_for_plot(profile_dict, stat_function, **kwargs)
    return fig
