from votekit.plots import profile_bar_plot
from votekit import Ballot, PreferenceProfile
from votekit.utils import borda_scores
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"})), weight=1
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})), weight=4)
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),), weight=1)
ballot_4 = Ballot(ranking=(frozenset({"Peter"}),), weight=1)

profile = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_3, ballot_4))


def test_profile_barplot_with_defaults():
    ax = profile_bar_plot(profile, borda_scores)

    assert isinstance(ax, Axes)
    plt.close()


def test_profile_barplot_with_kwds():
    ax = profile_bar_plot(
        profile,
        stat_function=borda_scores,
        profile_label="Profile Portland",
        normalize=True,
        profile_color="red",
        bar_width=1,
        category_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        categories_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()
