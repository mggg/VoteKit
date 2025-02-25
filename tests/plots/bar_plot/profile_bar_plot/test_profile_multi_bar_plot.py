from votekit.plots.profiles import multi_profile_bar_plot
from votekit.pref_profile import PreferenceProfile
from votekit.utils import COLOR_LIST
from votekit.ballot import Ballot
from matplotlib.axes import Axes
import pytest
import matplotlib.pyplot as plt

ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"})), weight=1
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})), weight=4)
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),), weight=1)
ballot_4 = Ballot(ranking=(frozenset({"Peter"}),), weight=1)

profile_1 = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_3, ballot_4))
profile_2 = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_4))


def test_barplot_with_defaults():
    ax = multi_profile_bar_plot(
        {"Profile 1": profile_1, "Profile 2": profile_2},
        stat_function="first place votes",
    )
    assert isinstance(ax, Axes)
    plt.close()

    ax = multi_profile_bar_plot(
        {"Profile 1": profile_1, "Profile 2": profile_2}, stat_function="mentions"
    )
    assert isinstance(ax, Axes)
    plt.close()

    ax = multi_profile_bar_plot(
        {"Profile 1": profile_1, "Profile 2": profile_2}, stat_function="borda"
    )
    assert isinstance(ax, Axes)
    plt.close()

    ax = multi_profile_bar_plot(
        {"Profile 1": profile_1, "Profile 2": profile_2}, stat_function="ballot lengths"
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_barplot_with_callable():
    ax = multi_profile_bar_plot(
        {"Profile 1": profile_1, "Profile 2": profile_2},
        stat_function=lambda x: {"1": 1},
    )
    assert isinstance(ax, Axes)
    plt.close()


def test_barplot_with_no_defaults():
    ax = multi_profile_bar_plot(
        {"Profile 1": profile_1, "Profile 2": profile_2},
        stat_function="first place votes",
        stat_function_kwds={"to_float": True},
        normalize=True,
        profile_colors={"Profile 1": "red"},
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


def test_wrapped_errors():
    with pytest.raises(
        ValueError, match=f"Cannot plot more than {len(COLOR_LIST)} profiles."
    ):
        multi_profile_bar_plot(
            {f"Profile_{i}": profile_1 for i in range(len(COLOR_LIST) + 2)},
            stat_function="borda",
        )

    with pytest.raises(
        ValueError, match="category_ordering must be the same length as the dictionary"
    ):
        multi_profile_bar_plot(
            {f"Profile_{i}": profile_1 for i in range(3)},
            stat_function="borda",
            category_ordering=["Chris"],
        )
