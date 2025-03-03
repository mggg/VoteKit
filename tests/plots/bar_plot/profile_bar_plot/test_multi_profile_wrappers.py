from votekit.plots import (
    multi_profile_borda_plot,
    multi_profile_fpv_plot,
    multi_profile_mentions_plot,
    multi_profile_ballot_lengths_plot,
)
from votekit import Ballot, PreferenceProfile
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"})), weight=1
)
ballot_2 = Ballot(ranking=(frozenset({"Moon"}), frozenset({"Peter"})), weight=4)
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),), weight=1)
ballot_4 = Ballot(ranking=(frozenset({"Peter"}),), weight=1)

profile_1 = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_3, ballot_4))
profile_2 = PreferenceProfile(ballots=(ballot_1, ballot_2, ballot_4))

profile_dict = {"1": profile_1, "2": profile_2}


def test_multi_profile_borda_plot_with_defaults():
    ax = multi_profile_borda_plot(profile_dict)

    assert isinstance(ax, Axes)
    plt.close()


def test_multi_profile_borda_plot_with_kwds():
    ax = multi_profile_borda_plot(
        profile_dict,
        normalize=True,
        profile_colors={"1": "red", "2": "green"},
        bar_width=1,
        candidate_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        candidate_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        relabel_candidates_with_int=True,
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()


def test_multi_profile_mentions_plot_with_defaults():
    ax = multi_profile_mentions_plot(profile_dict)

    assert isinstance(ax, Axes)
    plt.close()


def test_multi_profile_mentions_plot_with_kwds():
    ax = multi_profile_mentions_plot(
        profile_dict,
        normalize=True,
        profile_colors={"1": "red", "2": "green"},
        bar_width=1,
        candidate_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        candidate_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        relabel_candidates_with_int=True,
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()


def test_multi_profile_fpv_plot_with_defaults():
    ax = multi_profile_fpv_plot(profile_dict)

    assert isinstance(ax, Axes)
    plt.close()


def test_multi_profile_fpv_plot_with_kwds():
    ax = multi_profile_fpv_plot(
        profile_dict,
        normalize=True,
        profile_colors={"1": "red", "2": "green"},
        bar_width=1,
        candidate_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        candidate_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        relabel_candidates_with_int=True,
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()


def test_multi_profile_ballot_lengths_plot_with_defaults():
    ax = multi_profile_ballot_lengths_plot(profile_dict)

    assert isinstance(ax, Axes)
    plt.close()


def test_multi_profile_ballot_lengths_plot_with_kwds():
    ax = multi_profile_ballot_lengths_plot(
        profile_dict,
        normalize=True,
        profile_colors={"1": "red", "2": "green"},
        bar_width=1,
        lengths_ordering=[3, 1, 2],
        x_axis_name="lengths",
        y_axis_name="fpv",
        title="plot",
        show_profile_legend=True,
        lengths_legend={3: 5, 1: 6, 2: 7},
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()
